#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Export macro so the extern "C" entry point is visible to ctypes.
// On Windows, symbols are not exported from a DLL without __declspec(dllexport).
#ifdef _WIN32
#define CUDA_API __declspec(dllexport)
#else
#define CUDA_API
#endif

// Fixed power-of-two block size. The tree reduction below is only correct for a
// power-of-two block size; an arbitrary size (as returned by the occupancy API)
// silently drops elements.
#define BLOCK_SIZE 256

__global__ void gpu_reduction_sum_kernel(const float* d_in, int size, float* d_out) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    float sum = 0.0f;
    if (idx < (unsigned int)size) {
        sum = d_in[idx];
        if (idx + blockDim.x < (unsigned int)size)
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

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                          \
                    cudaGetErrorString(_e), __FILE__, __LINE__);                \
            cudaFree(d_a);                                                       \
            cudaFree(d_b);                                                       \
            return -1;                                                          \
        }                                                                       \
    } while (0)

// Returns 0 on success, non-zero on any CUDA failure so the Python layer can
// fall back to the CPU implementation instead of trusting a bad result.
extern "C" CUDA_API int run_reduction(const float* h_data, int size, float* h_result) {
    float *d_a = nullptr, *d_b = nullptr;

    if (size <= 0) {
        if (h_result) h_result[0] = 0.0f;
        return 0;
    }

    const int blockSize = BLOCK_SIZE;
    int gridSize = (size + (blockSize * 2 - 1)) / (blockSize * 2);

    CUDA_CHECK(cudaMalloc((void**)&d_a, size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_a, h_data, size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void**)&d_b, gridSize * sizeof(float)));

    // First pass: reduce `size` elements in d_a into `gridSize` partials in d_b.
    gpu_reduction_sum_kernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_a, size, d_b);
    CUDA_CHECK(cudaGetLastError());

    // Iteratively reduce the partials. Ping-pong between two buffers instead of
    // reducing in place — an in-place multi-pass reduction has a read/write race
    // because one block's output slot overlaps another block's input range.
    int s = gridSize;
    float* d_in = d_b;       // currently holds `s` partial sums
    float* d_scratch = d_a;  // large enough (size >= gridSize >= every later s)
    while (s > 1) {
        int gs = (s + (blockSize * 2 - 1)) / (blockSize * 2);
        gpu_reduction_sum_kernel<<<gs, blockSize, blockSize * sizeof(float)>>>(d_in, s, d_scratch);
        CUDA_CHECK(cudaGetLastError());
        float* tmp = d_in; d_in = d_scratch; d_scratch = tmp;
        s = gs;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_result, d_in, sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_a);
    cudaFree(d_b);
    return 0;
}
