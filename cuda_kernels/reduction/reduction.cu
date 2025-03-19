#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" {

__global__ void gpu_reduction_sum_kernel(float* d_in, int size, float* d_out) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    float sum = 0.0f;
    if (idx < size) {
        sum = d_in[idx];
        if (idx + blockDim.x < size)
            sum += d_in[idx + blockDim.x];
    }
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0)
        d_out[blockIdx.x] = sdata[0];
}


void gpu_reduction_sum(float* h_data, int size, float* h_result) {
    float *d_data = nullptr, *d_intermediate = nullptr;

    cudaMalloc((void**)&d_data, size * sizeof(float));
    cudaMemcpy(d_data, h_data, size * sizeof(float), cudaMemcpyHostToDevice);

    int minGridSize, blockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, gpu_reduction_sum_kernel, 0, size);
    int gridSize = (size + (blockSize * 2 - 1)) / (blockSize * 2);
    cudaMalloc((void**)&d_intermediate, gridSize * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    gpu_reduction_sum_kernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_data, size, d_intermediate);

    int s = gridSize;
    while (s > 1) {
        int newBlockSize;
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &newBlockSize, gpu_reduction_sum_kernel, 0, s);
        int newGridSize = (s + (newBlockSize * 2 - 1)) / (newBlockSize * 2);
        gpu_reduction_sum_kernel<<<newGridSize, newBlockSize, newBlockSize * sizeof(float)>>>(d_intermediate, s, d_intermediate);
        s = newGridSize;
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Kernel execution time: %f ms\n", elapsedTime);

    cudaMemcpy(h_result, d_intermediate, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_intermediate);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

}