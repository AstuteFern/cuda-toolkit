#include <cuda_runtime.h>
#include <stdio.h>

__global__ void gpu_autocorrelation(const float* __restrict__ data, float* __restrict__ result, int size, int max_lag) {
    extern __shared__ float shared_sum[];
    int lag = blockIdx.x;
    if (lag >= max_lag) return;

    int tid = threadIdx.x;
    int block_size = blockDim.x;
    float sum = 0.0f;

    for (int i = tid; i < size - lag; i += block_size) {
        sum += data[i] * data[i + lag];
    }

    shared_sum[tid] = sum;
    __syncthreads();

    for (int s = block_size / 2; s > 32; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32) {
        volatile float* vshared = shared_sum;
        vshared[tid] += vshared[tid + 32];
        vshared[tid] += vshared[tid + 16];
        vshared[tid] += vshared[tid + 8];
        vshared[tid] += vshared[tid + 4];
        vshared[tid] += vshared[tid + 2];
        vshared[tid] += vshared[tid + 1];
    }

    if (tid == 0) {
        result[lag] = shared_sum[0];
    }
}

extern "C" void run_autocorrelation(const float* data, float* result, int size, int max_lag) {
    float *d_data, *d_result;
    cudaError_t err;

    err = cudaMalloc(&d_data, size * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_data failed: %s\n", cudaGetErrorString(err));
        return;
    }
    err = cudaMalloc(&d_result, max_lag * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_result failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return;
    }

    err = cudaMemcpy(d_data, data, size * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy to d_data failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        cudaFree(d_result);
        return;
    }

    int optimalBlockSize = 0;
    int minGridSize = 0;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &optimalBlockSize, gpu_autocorrelation, 0, 0);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Running on GPU: %s, Optimal block size: %d\n", prop.name, optimalBlockSize);

    gpu_autocorrelation<<<max_lag, optimalBlockSize, optimalBlockSize * sizeof(float)>>>(d_data, d_result, size, max_lag);

    err = cudaMemcpy(result, d_result, max_lag * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy to host failed: %s\n", cudaGetErrorString(err));
    }

    cudaFree(d_data);
    cudaFree(d_result);
}