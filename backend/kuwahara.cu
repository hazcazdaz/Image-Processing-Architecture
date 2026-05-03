#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define BLOCK_SIZE 16
#define TILE_WIDTH 16
#define KUWAHARA_RADIUS 3         // each quadrant is (R+1)×(R+1) = 4×4
#define QUAD_SIZE ((KUWAHARA_RADIUS+1)*(KUWAHARA_RADIUS+1))  // 16
#define BLOCK_WIDTH (TILE_WIDTH + 2*KUWAHARA_RADIUS)  // 22

// Luminance weights (BT.601)
#define W_R 0.299f
#define W_G 0.587f
#define W_B 0.114f

__device__ __host__ inline float toGray(unsigned char r, unsigned char g, unsigned char b) {
    return W_R * r + W_G * g + W_B * b;
}

// Compute mean and variance of a quadrant in global memory
__device__ __host__ inline void quadrantStats(
    unsigned char* input, int width, int height, int channels,
    int cx, int cy, int x0, int y0, int x1, int y1,
    float* meanR, float* meanG, float* meanB, float* variance)
{
    float sumR = 0, sumG = 0, sumB = 0, sumLum = 0, sumLum2 = 0;
    int count = 0;

    for (int y = y0; y <= y1; y++) {
        for (int x = x0; x <= x1; x++) {
            int px = min(max(x, 0), width - 1);
            int py = min(max(y, 0), height - 1);
            int idx = (py * width + px) * channels;
            float r = input[idx];
            float g = input[idx + 1];
            float b = input[idx + 2];
            float lum = W_R * r + W_G * g + W_B * b;
            sumR += r;
            sumG += g;
            sumB += b;
            sumLum += lum;
            sumLum2 += lum * lum;
            count++;
        }
    }

    float inv = 1.0f / count;
    *meanR = sumR * inv;
    *meanG = sumG * inv;
    *meanB = sumB * inv;
    float mean = sumLum * inv;
    *variance = sumLum2 * inv - mean * mean;
}

// GPU kernel — global memory
__global__ void kuwaharaGlobal(unsigned char* input, unsigned char* output,
                               int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int R = KUWAHARA_RADIUS;
        float bestVar = 1e30f;
        float bestR = 0, bestG = 0, bestB = 0;

        // Four quadrants: top-left, top-right, bottom-left, bottom-right
        int regions[4][4] = {
            { x - R, y - R, x,     y     },  // top-left
            { x,     y - R, x + R, y     },  // top-right
            { x - R, y,     x,     y + R },  // bottom-left
            { x,     y,     x + R, y + R },  // bottom-right
        };

        for (int q = 0; q < 4; q++) {
            float mR, mG, mB, var;
            quadrantStats(input, width, height, channels,
                          x, y, regions[q][0], regions[q][1], regions[q][2], regions[q][3],
                          &mR, &mG, &mB, &var);
            if (var < bestVar) {
                bestVar = var;
                bestR = mR;
                bestG = mG;
                bestB = mB;
            }
        }

        int outIdx = (y * width + x) * channels;
        output[outIdx]     = (unsigned char)min(max(bestR, 0.0f), 255.0f);
        output[outIdx + 1] = (unsigned char)min(max(bestG, 0.0f), 255.0f);
        output[outIdx + 2] = (unsigned char)min(max(bestB, 0.0f), 255.0f);
        if (channels == 4) {
            output[outIdx + 3] = input[outIdx + 3];
        }
    }
}

// GPU kernel — shared memory with tiled loading
__global__ void kuwaharaShared(unsigned char* input, unsigned char* output,
                               int width, int height, int channels) {
    __shared__ unsigned char tileR[BLOCK_WIDTH][BLOCK_WIDTH];
    __shared__ unsigned char tileG[BLOCK_WIDTH][BLOCK_WIDTH];
    __shared__ unsigned char tileB[BLOCK_WIDTH][BLOCK_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * TILE_WIDTH + tx;
    int y = blockIdx.y * TILE_WIDTH + ty;

    // Cooperatively load tile with halo
    for (int dy = ty; dy < BLOCK_WIDTH; dy += TILE_WIDTH) {
        for (int dx = tx; dx < BLOCK_WIDTH; dx += TILE_WIDTH) {
            int gx = blockIdx.x * TILE_WIDTH + dx - KUWAHARA_RADIUS;
            int gy = blockIdx.y * TILE_WIDTH + dy - KUWAHARA_RADIUS;
            gx = min(max(gx, 0), width - 1);
            gy = min(max(gy, 0), height - 1);
            int idx = (gy * width + gx) * channels;
            tileR[dy][dx] = input[idx];
            tileG[dy][dx] = input[idx + 1];
            tileB[dy][dx] = input[idx + 2];
        }
    }
    __syncthreads();

    if (x < width && y < height) {
        int R = KUWAHARA_RADIUS;
        float bestVar = 1e30f;
        float bestR = 0, bestG = 0, bestB = 0;

        // Center of this pixel in shared memory coordinates
        int sx = tx + R;
        int sy = ty + R;

        // Four quadrants defined by offsets from center in shared mem
        int regions[4][4] = {
            { sx - R, sy - R, sx,     sy     },
            { sx,     sy - R, sx + R, sy     },
            { sx - R, sy,     sx,     sy + R },
            { sx,     sy,     sx + R, sy + R },
        };

        for (int q = 0; q < 4; q++) {
            float sumR = 0, sumG = 0, sumB = 0, sumLum = 0, sumLum2 = 0;
            int count = 0;

            for (int j = regions[q][1]; j <= regions[q][3]; j++) {
                for (int i = regions[q][0]; i <= regions[q][2]; i++) {
                    float r = tileR[j][i];
                    float g = tileG[j][i];
                    float b = tileB[j][i];
                    float lum = W_R * r + W_G * g + W_B * b;
                    sumR += r;
                    sumG += g;
                    sumB += b;
                    sumLum += lum;
                    sumLum2 += lum * lum;
                    count++;
                }
            }

            float inv = 1.0f / count;
            float mR = sumR * inv;
            float mG = sumG * inv;
            float mB = sumB * inv;
            float mean = sumLum * inv;
            float var = sumLum2 * inv - mean * mean;

            if (var < bestVar) {
                bestVar = var;
                bestR = mR;
                bestG = mG;
                bestB = mB;
            }
        }

        int outIdx = (y * width + x) * channels;
        output[outIdx]     = (unsigned char)min(max(bestR, 0.0f), 255.0f);
        output[outIdx + 1] = (unsigned char)min(max(bestG, 0.0f), 255.0f);
        output[outIdx + 2] = (unsigned char)min(max(bestB, 0.0f), 255.0f);
        if (channels == 4) {
            output[outIdx + 3] = input[outIdx + 3];
        }
    }
}

// CPU implementation
void kuwaharaCPU(unsigned char* input, unsigned char* output,
                 int width, int height, int channels) {
    int R = KUWAHARA_RADIUS;

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float bestVar = 1e30f;
            float bestR = 0, bestG = 0, bestB = 0;

            int regions[4][4] = {
                { x - R, y - R, x,     y     },
                { x,     y - R, x + R, y     },
                { x - R, y,     x,     y + R },
                { x,     y,     x + R, y + R },
            };

            for (int q = 0; q < 4; q++) {
                float mR, mG, mB, var;
                quadrantStats(input, width, height, channels,
                              x, y, regions[q][0], regions[q][1], regions[q][2], regions[q][3],
                              &mR, &mG, &mB, &var);
                if (var < bestVar) {
                    bestVar = var;
                    bestR = mR;
                    bestG = mG;
                    bestB = mB;
                }
            }

            int outIdx = (y * width + x) * channels;
            output[outIdx]     = (unsigned char)(bestR < 0 ? 0 : (bestR > 255 ? 255 : bestR));
            output[outIdx + 1] = (unsigned char)(bestG < 0 ? 0 : (bestG > 255 ? 255 : bestG));
            output[outIdx + 2] = (unsigned char)(bestB < 0 ? 0 : (bestB > 255 ? 255 : bestB));
            if (channels == 4) {
                output[outIdx + 3] = input[outIdx + 3];
            }
        }
    }
}

// Usage: kuwahara <input_image> <output_image> <mode>
int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <input_image> <output_image> <mode>\n", argv[0]);
        fprintf(stderr, "  mode: CPU | GLOBAL | SHARED | ALL\n");
        return 1;
    }

    char* inputPath  = argv[1];
    char* outputPath = argv[2];
    char* mode       = argv[3];

    int width, height, channels;
    unsigned char* h_input = stbi_load(inputPath, &width, &height, &channels, 0);
    if (!h_input) {
        fprintf(stderr, "Failed to load image: %s\n", inputPath);
        return 1;
    }

    if (channels < 3) {
        fprintf(stderr, "Image must have at least 3 channels (RGB). Got %d\n", channels);
        stbi_image_free(h_input);
        return 1;
    }

    int imageBytes = width * height * channels * sizeof(unsigned char);
    unsigned char* h_output = (unsigned char*)malloc(imageBytes);

    double cpuTimeMs   = 0;
    float globalTimeMs = 0;
    float sharedTimeMs = 0;
    float transferTimeMs = 0;

    int runCPU    = (strcmp(mode, "CPU") == 0 || strcmp(mode, "ALL") == 0);
    int runGlobal = (strcmp(mode, "GLOBAL") == 0 || strcmp(mode, "ALL") == 0);
    int runShared = (strcmp(mode, "SHARED") == 0 || strcmp(mode, "ALL") == 0);

    // --- CPU ---
    if (runCPU) {
        double start = omp_get_wtime();
        kuwaharaCPU(h_input, h_output, width, height, channels);
        double end = omp_get_wtime();
        cpuTimeMs = (end - start) * 1000.0;
    }

    // --- GPU setup ---
    if (runGlobal || runShared) {
        unsigned char *d_input, *d_output;
        cudaMalloc(&d_input, imageBytes);
        cudaMalloc(&d_output, imageBytes);

        // Time the host-to-device transfer
        cudaEvent_t tStart, tStop;
        cudaEventCreate(&tStart);
        cudaEventCreate(&tStop);
        cudaEventRecord(tStart);
        cudaMemcpy(d_input, h_input, imageBytes, cudaMemcpyHostToDevice);
        cudaEventRecord(tStop);
        cudaEventSynchronize(tStop);
        cudaEventElapsedTime(&transferTimeMs, tStart, tStop);
        cudaEventDestroy(tStart);
        cudaEventDestroy(tStop);

        dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridSize((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                      (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

        // --- Global memory ---
        if (runGlobal) {
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
            kuwaharaGlobal<<<gridSize, blockSize>>>(d_input, d_output, width, height, channels);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&globalTimeMs, start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) fprintf(stderr, "kuwaharaGlobal: %s\n", cudaGetErrorString(err));
        }

        // --- Shared memory ---
        if (runShared) {
            dim3 sharedBlock(TILE_WIDTH, TILE_WIDTH);
            dim3 sharedGrid((width + TILE_WIDTH - 1) / TILE_WIDTH,
                            (height + TILE_WIDTH - 1) / TILE_WIDTH);

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
            kuwaharaShared<<<sharedGrid, sharedBlock>>>(d_input, d_output, width, height, channels);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&sharedTimeMs, start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) fprintf(stderr, "kuwaharaShared: %s\n", cudaGetErrorString(err));
        }

        cudaMemcpy(h_output, d_output, imageBytes, cudaMemcpyDeviceToHost);
        cudaFree(d_input);
        cudaFree(d_output);
    }

    // Save output image
    stbi_write_png(outputPath, width, height, channels, h_output, width * channels);

    // Print results as JSON for the backend to parse
    int megapixels_per_sec = 0;
    float bestGpu = (sharedTimeMs > 0) ? sharedTimeMs : globalTimeMs;
    if (bestGpu > 0) {
        megapixels_per_sec = (int)((width * height) / (bestGpu / 1000.0) / 1e6);
    }

    printf("{\"width\":%d,\"height\":%d,\"channels\":%d,", width, height, channels);
    printf("\"cpuTime\":%.3f,\"globalTime\":%.3f,\"sharedTime\":%.3f,", cpuTimeMs, globalTimeMs, sharedTimeMs);
    printf("\"transferTime\":%.3f,\"throughput\":%d}\n", transferTimeMs, megapixels_per_sec);

    stbi_image_free(h_input);
    free(h_output);

    return 0;
}
