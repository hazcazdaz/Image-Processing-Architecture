#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define BLOCK_SIZE 16
#define TILE_WIDTH 16
#define MEDIAN_RADIUS 1          // 3x3 window
#define WINDOW_SIZE ((2*MEDIAN_RADIUS+1)*(2*MEDIAN_RADIUS+1))  // 9
#define BLOCK_WIDTH (TILE_WIDTH + 2*MEDIAN_RADIUS)  // 18

// Insertion sort for small arrays (9 elements)
__device__ __host__ inline void insertionSort(unsigned char* arr, int n) {
    for (int i = 1; i < n; i++) {
        unsigned char key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

// GPU kernel — global memory
__global__ void medianGlobal(unsigned char* input, unsigned char* output,
                             int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        for (int c = 0; c < min(channels, 3); c++) {
            unsigned char window[WINDOW_SIZE];
            int count = 0;

            for (int ky = -MEDIAN_RADIUS; ky <= MEDIAN_RADIUS; ky++) {
                for (int kx = -MEDIAN_RADIUS; kx <= MEDIAN_RADIUS; kx++) {
                    int px = min(max(x + kx, 0), width - 1);
                    int py = min(max(y + ky, 0), height - 1);
                    window[count++] = input[(py * width + px) * channels + c];
                }
            }

            insertionSort(window, WINDOW_SIZE);
            output[(y * width + x) * channels + c] = window[WINDOW_SIZE / 2];
        }

        if (channels == 4) {
            output[(y * width + x) * channels + 3] = input[(y * width + x) * channels + 3];
        }
    }
}

// GPU kernel — shared memory with tiled loading
__global__ void medianShared(unsigned char* input, unsigned char* output,
                             int width, int height, int channels) {
    // Shared tile for one channel at a time
    __shared__ unsigned char tile[BLOCK_WIDTH][BLOCK_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * TILE_WIDTH + tx;
    int y = blockIdx.y * TILE_WIDTH + ty;

    for (int c = 0; c < min(channels, 3); c++) {
        // Load tile including halo
        for (int dy = ty; dy < BLOCK_WIDTH; dy += TILE_WIDTH) {
            for (int dx = tx; dx < BLOCK_WIDTH; dx += TILE_WIDTH) {
                int gx = blockIdx.x * TILE_WIDTH + dx - MEDIAN_RADIUS;
                int gy = blockIdx.y * TILE_WIDTH + dy - MEDIAN_RADIUS;
                gx = min(max(gx, 0), width - 1);
                gy = min(max(gy, 0), height - 1);
                tile[dy][dx] = input[(gy * width + gx) * channels + c];
            }
        }
        __syncthreads();

        if (x < width && y < height) {
            unsigned char window[WINDOW_SIZE];
            int count = 0;

            for (int ky = -MEDIAN_RADIUS; ky <= MEDIAN_RADIUS; ky++) {
                for (int kx = -MEDIAN_RADIUS; kx <= MEDIAN_RADIUS; kx++) {
                    window[count++] = tile[ty + MEDIAN_RADIUS + ky][tx + MEDIAN_RADIUS + kx];
                }
            }

            insertionSort(window, WINDOW_SIZE);
            output[(y * width + x) * channels + c] = window[WINDOW_SIZE / 2];
        }
        __syncthreads();
    }

    if (x < width && y < height && channels == 4) {
        output[(y * width + x) * channels + 3] = input[(y * width + x) * channels + 3];
    }
}

// CPU implementation
void medianCPU(unsigned char* input, unsigned char* output,
               int width, int height, int channels) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < 3 && c < channels; c++) {
                unsigned char window[WINDOW_SIZE];
                int count = 0;

                for (int ky = -MEDIAN_RADIUS; ky <= MEDIAN_RADIUS; ky++) {
                    for (int kx = -MEDIAN_RADIUS; kx <= MEDIAN_RADIUS; kx++) {
                        int px = x + kx;
                        int py = y + ky;
                        px = (px < 0) ? 0 : ((px >= width) ? width - 1 : px);
                        py = (py < 0) ? 0 : ((py >= height) ? height - 1 : py);
                        window[count++] = input[(py * width + px) * channels + c];
                    }
                }

                insertionSort(window, WINDOW_SIZE);
                output[(y * width + x) * channels + c] = window[WINDOW_SIZE / 2];
            }

            if (channels == 4) {
                output[(y * width + x) * channels + 3] = input[(y * width + x) * channels + 3];
            }
        }
    }
}

// Usage: median <input_image> <output_image> <mode>
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
        medianCPU(h_input, h_output, width, height, channels);
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
            medianGlobal<<<gridSize, blockSize>>>(d_input, d_output, width, height, channels);
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
            medianShared<<<sharedGrid, sharedBlock>>>(d_input, d_output, width, height, channels);
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
