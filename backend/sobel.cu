#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define BLOCK_SIZE 16
#define TILE_WIDTH 16
#define BLOCK_WIDTH (TILE_WIDTH + 2)  // +2 for 1-pixel radius on each side

// Sobel kernels (constant memory)
__constant__ int c_Gx[3][3] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
};

__constant__ int c_Gy[3][3] = {
    {-1, -2, -1},
    { 0,  0,  0},
    { 1,  2,  1}
};

// Luminance weights for grayscale conversion (BT.601)
#define W_R 0.299f
#define W_G 0.587f
#define W_B 0.114f

// Helper: convert RGB pixel to grayscale
__device__ __host__ inline float toGray(unsigned char r, unsigned char g, unsigned char b) {
    return W_R * r + W_G * g + W_B * b;
}

// GPU kernel — global memory
__global__ void sobelGlobal(unsigned char* input, unsigned char* output,
                            int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float gx = 0.0f, gy = 0.0f;

        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                int px = min(max(x + kx, 0), width - 1);
                int py = min(max(y + ky, 0), height - 1);
                int idx = (py * width + px) * channels;

                float gray = toGray(input[idx], input[idx + 1], input[idx + 2]);
                gx += gray * c_Gx[ky + 1][kx + 1];
                gy += gray * c_Gy[ky + 1][kx + 1];
            }
        }

        unsigned char mag = (unsigned char)min(sqrtf(gx * gx + gy * gy), 255.0f);
        int outIdx = (y * width + x) * channels;
        output[outIdx]     = mag;
        output[outIdx + 1] = mag;
        output[outIdx + 2] = mag;
        if (channels == 4) {
            output[outIdx + 3] = input[outIdx + 3];
        }
    }
}

// GPU kernel — shared memory with tiled loading
__global__ void sobelShared(unsigned char* input, unsigned char* output,
                            int width, int height, int channels) {
    __shared__ float tile[BLOCK_WIDTH][BLOCK_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Output pixel coordinates
    int x = blockIdx.x * TILE_WIDTH + tx;
    int y = blockIdx.y * TILE_WIDTH + ty;

    // Load the tile including 1-pixel halo into shared memory
    // Each thread loads one element; extra threads handle the halo
    for (int dy = ty; dy < BLOCK_WIDTH; dy += TILE_WIDTH) {
        for (int dx = tx; dx < BLOCK_WIDTH; dx += TILE_WIDTH) {
            int gx = blockIdx.x * TILE_WIDTH + dx - 1;
            int gy = blockIdx.y * TILE_WIDTH + dy - 1;
            gx = min(max(gx, 0), width - 1);
            gy = min(max(gy, 0), height - 1);
            int idx = (gy * width + gx) * channels;
            tile[dy][dx] = toGray(input[idx], input[idx + 1], input[idx + 2]);
        }
    }
    __syncthreads();

    if (x < width && y < height) {
        float sumGx = 0.0f, sumGy = 0.0f;

        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                float val = tile[ty + 1 + ky][tx + 1 + kx];
                sumGx += val * c_Gx[ky + 1][kx + 1];
                sumGy += val * c_Gy[ky + 1][kx + 1];
            }
        }

        unsigned char mag = (unsigned char)min(sqrtf(sumGx * sumGx + sumGy * sumGy), 255.0f);
        int outIdx = (y * width + x) * channels;
        output[outIdx]     = mag;
        output[outIdx + 1] = mag;
        output[outIdx + 2] = mag;
        if (channels == 4) {
            output[outIdx + 3] = input[outIdx + 3];
        }
    }
}

// CPU implementation
void sobelCPU(unsigned char* input, unsigned char* output,
              int width, int height, int channels) {
    int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float gx = 0.0f, gy = 0.0f;

            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int px = x + kx;
                    int py = y + ky;
                    px = (px < 0) ? 0 : ((px >= width) ? width - 1 : px);
                    py = (py < 0) ? 0 : ((py >= height) ? height - 1 : py);
                    int idx = (py * width + px) * channels;

                    float gray = toGray(input[idx], input[idx + 1], input[idx + 2]);
                    gx += gray * Gx[ky + 1][kx + 1];
                    gy += gray * Gy[ky + 1][kx + 1];
                }
            }

            float mag = sqrtf(gx * gx + gy * gy);
            if (mag > 255.0f) mag = 255.0f;
            unsigned char val = (unsigned char)mag;
            int outIdx = (y * width + x) * channels;
            output[outIdx]     = val;
            output[outIdx + 1] = val;
            output[outIdx + 2] = val;
            if (channels == 4) {
                output[outIdx + 3] = input[outIdx + 3];
            }
        }
    }
}

// Usage: sobel <input_image> <output_image> <mode>
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
        clock_t start = clock();
        sobelCPU(h_input, h_output, width, height, channels);
        clock_t end = clock();
        cpuTimeMs = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;
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
            sobelGlobal<<<gridSize, blockSize>>>(d_input, d_output, width, height, channels);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&globalTimeMs, start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
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
            sobelShared<<<sharedGrid, sharedBlock>>>(d_input, d_output, width, height, channels);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&sharedTimeMs, start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
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
