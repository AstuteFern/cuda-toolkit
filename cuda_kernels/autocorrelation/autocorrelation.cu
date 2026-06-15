#include <cuda_runtime.h>
#include <stdio.h>

// Export macro so the extern "C" entry point is visible to ctypes.
// On Windows, symbols are not exported from a DLL without __declspec(dllexport).
#ifdef _WIN32
#define CUDA_API __declspec(dllexport)
#else
#define CUDA_API
#endif

// Fixed power-of-two block size >= 64. The warp-unrolled reduction below assumes
// blockDim.x is a power of two and at least 64 (so tid+32 stays in bounds).
#define BLOCK_SIZE 256

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

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                          \
                    cudaGetErrorString(_e), __FILE__, __LINE__);                \
            cudaFree(d_data);                                                    \
            cudaFree(d_result);                                                  \
            return -1;                                                          \
        }                                                                       \
    } while (0)

// Returns 0 on success, non-zero on any CUDA failure so the Python layer can
// fall back to the CPU implementation instead of trusting a bad result.
extern "C" CUDA_API int run_autocorrelation(const float* data, float* result, int size, int max_lag) {
    float *d_data = nullptr, *d_result = nullptr;

    if (size <= 0 || max_lag <= 0) {
        return 0;
    }

    CUDA_CHECK(cudaMalloc((void**)&d_data, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_result, max_lag * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data, data, size * sizeof(float), cudaMemcpyHostToDevice));

    const int blockSize = BLOCK_SIZE;
    gpu_autocorrelation<<<max_lag, blockSize, blockSize * sizeof(float)>>>(d_data, d_result, size, max_lag);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(result, d_result, max_lag * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_data);
    cudaFree(d_result);
    return 0;
}
