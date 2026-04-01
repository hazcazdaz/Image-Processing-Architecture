#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Max kernel size we support for shared memory path
#define MAX_RADIUS 15
#define MAX_KERNEL_SIZE (2 * MAX_RADIUS + 1)

// Shared memory tiling config (used when kernel radius <= MAX_RADIUS)
#define TILE_WIDTH 22
#define BLOCK_WIDTH_SHARED(r) (TILE_WIDTH + 2 * (r))

// Constant memory for shared memory filter
__constant__ float c_Filter[MAX_KERNEL_SIZE][MAX_KERNEL_SIZE];

// Shared memory kernel — uses dynamic sizing via template-like approach
// For simplicity, we fix the shared tile to the max possible size
__global__ void gaussianBlurShared(unsigned char* in, unsigned char* out,
                                    int Width, int Height, int channels,
                                    int radius, int blockW) {
    extern __shared__ float ds_in[];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty - radius;
    int Col = bx * TILE_WIDTH + tx - radius;

    for (int c = 0; c < channels; ++c) {
        if (Row >= 0 && Row < Height && Col >= 0 && Col < Width) {
            ds_in[ty * blockW + tx] = in[(Row * Width + Col) * channels + c];
        } else {
            ds_in[ty * blockW + tx] = 0.0f;
        }
        __syncthreads();

        if (ty >= radius && ty < blockW - radius &&
            tx >= radius && tx < blockW - radius) {
            float Pvalue = 0.0f;
            for (int i = -radius; i <= radius; ++i) {
                for (int j = -radius; j <= radius; ++j) {
                    Pvalue += ds_in[(ty + i) * blockW + (tx + j)] * c_Filter[i + radius][j + radius];
                }
            }

            int outRow = by * TILE_WIDTH + (ty - radius);
            int outCol = bx * TILE_WIDTH + (tx - radius);

            if (outRow < Height && outCol < Width) {
                out[(outRow * Width + outCol) * channels + c] = (unsigned char)Pvalue;
            }
        }
        __syncthreads();
    }
}

// Global memory kernel
__global__ void gaussianBlurGlobal(unsigned char* input, unsigned char* output,
                                    int channels, int width, int height,
                                    float* kernel, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int radius = kernelSize / 2;
        for (int c = 0; c < channels; c++) {
            float sum = 0.0f;
            for (int ky = -radius; ky <= radius; ky++) {
                for (int kx = -radius; kx <= radius; kx++) {
                    int px = x + kx;
                    int py = y + ky;
                    px = (px < 0) ? 0 : ((px >= width) ? width - 1 : px);
                    py = (py < 0) ? 0 : ((py >= height) ? height - 1 : py);
                    int pixel_idx = (py * width + px) * channels + c;
                    float kernel_val = kernel[(ky + radius) * kernelSize + (kx + radius)];
                    sum += input[pixel_idx] * kernel_val;
                }
            }
            output[(y * width + x) * channels + c] = (unsigned char)sum;
        }
    }
}

// CPU implementation
void gaussianBlurCPU(unsigned char* input, unsigned char* output,
                     int width, int height, int channels,
                     float* kernel, int kernelSize) {
    int radius = kernelSize / 2;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                float sum = 0.0f;
                for (int ky = -radius; ky <= radius; ky++) {
                    for (int kx = -radius; kx <= radius; kx++) {
                        int px = x + kx;
                        int py = y + ky;
                        px = (px < 0) ? 0 : ((px >= width) ? width - 1 : px);
                        py = (py < 0) ? 0 : ((py >= height) ? height - 1 : py);
                        int pixel_idx = (py * width + px) * channels + c;
                        float kernel_val = kernel[(ky + radius) * kernelSize + (kx + radius)];
                        sum += input[pixel_idx] * kernel_val;
                    }
                }
                output[(y * width + x) * channels + c] = (unsigned char)sum;
            }
        }
    }
}

void generateGaussianKernel(float* kernel, int size, float sigma) {
    float sum = 0.0f;
    int center = size / 2;
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            float dx = x - center;
            float dy = y - center;
            kernel[y * size + x] = exp(-(dx * dx + dy * dy) / (2.0f * sigma * sigma));
            sum += kernel[y * size + x];
        }
    }
    for (int i = 0; i < size * size; i++) {
        kernel[i] /= sum;
    }
}

// Usage: gaussian_blur <input_image> <output_image> <kernel_size> <mode>
// mode: CPU, GLOBAL, SHARED, or ALL (runs all three and prints JSON)
int main(int argc, char** argv) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <input_image> <output_image> <kernel_size> <mode>\n", argv[0]);
        fprintf(stderr, "  mode: CPU | GLOBAL | SHARED | ALL\n");
        return 1;
    }

    char* inputPath = argv[1];
    char* outputPath = argv[2];
    int kernelSize = atoi(argv[3]);
    char* mode = argv[4];

    if (kernelSize < 1 || kernelSize > MAX_KERNEL_SIZE || kernelSize % 2 == 0) {
        fprintf(stderr, "Kernel size must be odd and between 1 and %d\n", MAX_KERNEL_SIZE);
        return 1;
    }

    int radius = kernelSize / 2;
    float sigma = kernelSize / 6.0f;
    if (sigma < 1.0f) sigma = 1.0f;

    int width, height, channels;
    unsigned char* h_input = stbi_load(inputPath, &width, &height, &channels, 0);
    if (!h_input) {
        fprintf(stderr, "Failed to load image: %s\n", inputPath);
        return 1;
    }

    int imageBytes = width * height * channels * sizeof(unsigned char);
    int kernelBytes = kernelSize * kernelSize * sizeof(float);

    unsigned char* h_output = (unsigned char*)malloc(imageBytes);
    float* h_kernel = (float*)malloc(kernelBytes);
    generateGaussianKernel(h_kernel, kernelSize, sigma);

    double cpuTimeMs = 0;
    float globalTimeMs = 0;
    float sharedTimeMs = 0;
    float transferTimeMs = 0;

    int runCPU    = (strcmp(mode, "CPU") == 0 || strcmp(mode, "ALL") == 0);
    int runGlobal = (strcmp(mode, "GLOBAL") == 0 || strcmp(mode, "ALL") == 0);
    int runShared = (strcmp(mode, "SHARED") == 0 || strcmp(mode, "ALL") == 0);

    // --- CPU ---
    if (runCPU) {
        clock_t start = clock();
        gaussianBlurCPU(h_input, h_output, width, height, channels, h_kernel, kernelSize);
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

        // --- Global memory ---
        if (runGlobal) {
            float* d_kernel;
            cudaMalloc(&d_kernel, kernelBytes);
            cudaMemcpy(d_kernel, h_kernel, kernelBytes, cudaMemcpyHostToDevice);

            dim3 blockSize(16, 16);
            dim3 gridSize((width + 15) / 16, (height + 15) / 16);

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
            gaussianBlurGlobal<<<gridSize, blockSize>>>(d_input, d_output, channels, width, height, d_kernel, kernelSize);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&globalTimeMs, start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cudaFree(d_kernel);
        }

        // --- Shared memory ---
        // Only use shared memory when block dimensions fit within 1024 thread limit
        int blockW = TILE_WIDTH + 2 * radius;
        int canRunShared = runShared && (blockW * blockW <= 1024);

        if (canRunShared) {
            float h_filter[MAX_KERNEL_SIZE][MAX_KERNEL_SIZE] = {0};
            for (int y = 0; y < kernelSize; y++) {
                for (int x = 0; x < kernelSize; x++) {
                    h_filter[y][x] = h_kernel[y * kernelSize + x];
                }
            }
            cudaMemcpyToSymbol(c_Filter, h_filter, sizeof(h_filter));

            dim3 blockSize(blockW, blockW);
            dim3 gridSize((width + TILE_WIDTH - 1) / TILE_WIDTH,
                          (height + TILE_WIDTH - 1) / TILE_WIDTH);
            size_t sharedMemSize = blockW * blockW * sizeof(float);

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
            gaussianBlurShared<<<gridSize, blockSize, sharedMemSize>>>(d_input, d_output, width, height, channels, radius, blockW);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&sharedTimeMs, start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);

            cudaMemcpy(h_output, d_output, imageBytes, cudaMemcpyDeviceToHost);
        } else if (runGlobal) {
            cudaMemcpy(h_output, d_output, imageBytes, cudaMemcpyDeviceToHost);
        }

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

    printf("{\"width\":%d,\"height\":%d,\"channels\":%d,\"kernelSize\":%d,", width, height, channels, kernelSize);
    printf("\"cpuTime\":%.3f,\"globalTime\":%.3f,\"sharedTime\":%.3f,", cpuTimeMs, globalTimeMs, sharedTimeMs);
    printf("\"transferTime\":%.3f,\"throughput\":%d}\n", transferTimeMs, megapixels_per_sec);

    stbi_image_free(h_input);
    free(h_output);
    free(h_kernel);

    return 0;
}
