#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Luminance weights: ITU-R BT.601
#define W_R 0.299f
#define W_G 0.587f
#define W_B 0.114f

#define BLOCK_SIZE 16

// GPU kernel — each thread converts one pixel to grayscale
__global__ void grayscaleGlobal(unsigned char* input, unsigned char* output,
                                 int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * channels;
        unsigned char r = input[idx];
        unsigned char g = input[idx + 1];
        unsigned char b = input[idx + 2];
        unsigned char gray = (unsigned char)(W_R * r + W_G * g + W_B * b);

        output[idx]     = gray;
        output[idx + 1] = gray;
        output[idx + 2] = gray;

        // Preserve alpha channel if present
        if (channels == 4) {
            output[idx + 3] = input[idx + 3];
        }
    }
}

// GPU kernel — shared memory version using tiled loading
__global__ void grayscaleShared(unsigned char* input, unsigned char* output,
                                 int width, int height, int channels) {
    extern __shared__ unsigned char tile[];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int localIdx = (threadIdx.y * blockDim.x + threadIdx.x) * channels;

    // Load pixel into shared memory
    if (x < width && y < height) {
        int globalIdx = (y * width + x) * channels;
        for (int c = 0; c < channels; c++) {
            tile[localIdx + c] = input[globalIdx + c];
        }
    }
    __syncthreads();

    // Convert to grayscale from shared memory
    if (x < width && y < height) {
        unsigned char r = tile[localIdx];
        unsigned char g = tile[localIdx + 1];
        unsigned char b = tile[localIdx + 2];
        unsigned char gray = (unsigned char)(W_R * r + W_G * g + W_B * b);

        int globalIdx = (y * width + x) * channels;
        output[globalIdx]     = gray;
        output[globalIdx + 1] = gray;
        output[globalIdx + 2] = gray;

        if (channels == 4) {
            output[globalIdx + 3] = tile[localIdx + 3];
        }
    }
}

// CPU implementation
void grayscaleCPU(unsigned char* input, unsigned char* output,
                  int width, int height, int channels) {
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * channels;
            unsigned char r = input[idx];
            unsigned char g = input[idx + 1];
            unsigned char b = input[idx + 2];
            unsigned char gray = (unsigned char)(W_R * r + W_G * g + W_B * b);

            output[idx]     = gray;
            output[idx + 1] = gray;
            output[idx + 2] = gray;

            if (channels == 4) {
                output[idx + 3] = input[idx + 3];
            }
        }
    }
}

// Usage: grayscale <input_image> <output_image> ALL
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
        grayscaleCPU(h_input, h_output, width, height, channels);
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
            grayscaleGlobal<<<gridSize, blockSize>>>(d_input, d_output, width, height, channels);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&globalTimeMs, start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) fprintf(stderr, "grayscaleGlobal: %s\n", cudaGetErrorString(err));
        }

        // --- Shared memory ---
        if (runShared) {
            size_t sharedMemSize = BLOCK_SIZE * BLOCK_SIZE * channels * sizeof(unsigned char);

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
            grayscaleShared<<<gridSize, blockSize, sharedMemSize>>>(d_input, d_output, width, height, channels);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&sharedTimeMs, start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) fprintf(stderr, "grayscaleShared: %s\n", cudaGetErrorString(err));
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
