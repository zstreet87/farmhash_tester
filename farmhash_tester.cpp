#include <iostream>
#include <cstring>
#include <hip/hip_runtime.h>

#include "farmhash_gpu.h"
#include "farmhash.h"

// We set the buffer size to 20 as it is sufficient to cover the number of
// digits in any integer type.
//constexpr int kSharedMemBufferSizePerThread = 20;

//template <typename T>
//__global__ __launch_bounds__(1024) void ComputeHashes(
//    const T* __restrict__ vals, int vals_size, int64 num_buckets,
//    int64* __restrict__ hashes) {
//
//  extern __shared__ char s[];
//  uint64_t a_hash = ::util_gpu::Fingerprint64(
//    s + threadIdx.x * kSharedMemBufferSizePerThread, size);
//}

__global__ void hipFarmHash(const char* input, size_t length, uint64_t* output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        output[idx] = ::util_gpu::Fingerprint64(input, length);
    }
}

int main() {
    const char* inputString = "Zach is testing farmhash";
    size_t length = std::strlen(inputString);
    uint64_t* gpuHashResult;
    uint64_t cpuHashResult;

    hipMalloc(&gpuHashResult, length * sizeof(uint64_t));

    int blockSize = 256;
    int numBlocks = (length + blockSize - 1) / blockSize;
    hipLaunchKernelGGL(hipFarmHash, dim3(1), dim3(1), 0, 0, inputString, length, gpuHashResult);

    hipDeviceSynchronize();

    cpuHashResult = ::util::Fingerprint64(inputString, length);

    bool success = true;
    for (size_t i = 0; i < length; i++) {
        if (gpuHashResult[i] != cpuHashResult) {
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "Hashing successful. GPU and CPU hashes match.\n";
    } else {
        std::cout << "Hashing failed. GPU and CPU hashes do not match.\n";
        std::cout << gpuHashResult[1] << " " << cpuHashResult << "\n" << std::endl;
    }

    hipFree(gpuHashResult);

    return 0;
}
