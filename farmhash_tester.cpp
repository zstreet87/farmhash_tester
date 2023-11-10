#include <cstring>
#include <hip/hip_runtime.h>
#include <iostream>
#include <random> //getting extern char s

#include "farmhash_gpu.h"
#include "gpu_kernel_helper.h"

// We set the buffer size to 20 as it is sufficient to cover the number of
// digits in any integer type.
constexpr int kSharedMemBufferSizePerThread = 20;

template <typename T>
__device__ __forceinline__ void FillDigits(T val, int num_digits, int *i,
                                           char *buf) {
  int factor = (val < 0 ? -1 : 1);

  int num_digits_a = num_digits;
  do {
    int digit = static_cast<int>((val % 10) * factor);
    buf[(*i) + num_digits - 1] = digit + '0';
    val /= 10;
    num_digits--;
  } while (val != 0);

  (*i) += num_digits_a;
}

template <typename T>
__device__ __forceinline__ int IntegerToString(T val, char *buf) {
  int num_digits = 0;
  T val_a = val;
  do {
    val_a = val_a / 10;
    num_digits++;
  } while (val_a != 0);

  int i = 0;
  if (val < 0) {
    buf[i++] = '-';
  }

  FillDigits(val, num_digits, &i, buf);

  return i;
}

template <typename T>

__global__ __launch_bounds__(1024) void kernel(const T *__restrict__ vals, uint64_t *output) {

  extern __shared__ char s[];

  GPU_1D_KERNEL_LOOP(tid, 1) {
    int size = IntegerToString(vals[tid],
                               s + threadIdx.x * kSharedMemBufferSizePerThread);
    uint64_t a_hash = ::util_gpu::Fingerprint64(
        s + threadIdx.x * kSharedMemBufferSizePerThread, size);
    int64_t a_bucket = static_cast<int64_t>(a_hash % 100);
    output[tid] = a_bucket;
  }
}

int main() {

  const int length = 1;

  using templateType = int;

  templateType inputCPU = 1;
  templateType *inputGPU;

  uint64_t *gpuHashResult;
  uint64_t *gpuHashResultHost =
      new uint64_t[length]; // Host memory for GPU results

  if (hipMalloc(&inputGPU, length * sizeof(templateType)) != hipSuccess) {
    std::cerr << "hipMalloc failed for inputGPU" << std::endl;
    return 1;
  }
  hipMemcpy(inputGPU, &inputCPU, length * sizeof(templateType),
            hipMemcpyHostToDevice);

  if (hipMalloc(&gpuHashResult, length * sizeof(templateType)) != hipSuccess) {
    std::cerr << "hipMalloc failed for gpuHashResult" << std::endl;
    return 1;
  }

  int smem_bytes_per_block = 20480;

  hipLaunchKernelGGL(kernel<templateType>, dim3(24), dim3(1024), smem_bytes_per_block, 0, inputGPU,
                     gpuHashResult);

  if (hipDeviceSynchronize() != hipSuccess) {
    std::cerr << "hipDeviceSynchronize failed" << std::endl;
    return 1;
  }

  if (hipMemcpy(gpuHashResultHost, gpuHashResult, length * sizeof(templateType),
                hipMemcpyDeviceToHost) != hipSuccess) {
    std::cerr << "hipMemcpy failed" << std::endl;
    return 1;
  }

  std::cout << gpuHashResultHost[0] << std::endl;

  hipFree(inputGPU);
  hipFree(gpuHashResult);
  delete[] gpuHashResultHost;

  return 0;
}
