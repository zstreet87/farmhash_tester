#include <cstring>
#include <hip/hip_runtime.h>
#include <iostream>
#include <random> //getting extern char s

#include "farmhash.h"
#include "farmhash_gpu.h"

// We set the buffer size to 20 as it is sufficient to cover the number of
// digits in any integer type.
constexpr int kSharedMemBufferSizePerThread = 20;

template <typename T>
__device__ __forceinline__ void FillDigits(T val, int num_digits, int *i,
                                           char *buf) {
  // eigen_assert(num_digits <= kSharedMemBufferSizePerThread - (*i));

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
__global__ void kernel(const T *__restrict__ vals, uint64_t *output) {

  // output[0] = 10;
  // int idx = blockIdx.x * blockDim.x + threadIdx.x;
  extern __shared__ char s[];

  int size =
      IntegerToString(vals[0], s + threadIdx.x * kSharedMemBufferSizePerThread);
  output[0] = ::util_gpu::Fingerprint64(
      s + threadIdx.x * kSharedMemBufferSizePerThread, size);
}

int main() {

  const int length = 1;

  int8_t inputCPU = 6;
  int8_t *inputGPU;

  uint64_t *gpuHashResult;
  uint64_t *gpuHashResultHost =
      new uint64_t[length]; // Host memory for GPU results

  if (hipMalloc(&inputGPU, length * sizeof(uint64_t)) != hipSuccess) {
    std::cerr << "hipMalloc failed" << std::endl;
    return 1;
  }
  hipMemcpy(inputGPU, &inputCPU, length * sizeof(int8_t),
            hipMemcpyHostToDevice);

  if (hipMalloc(&gpuHashResult, length * sizeof(uint64_t)) != hipSuccess) {
    std::cerr << "hipMalloc failed" << std::endl;
    return 1;
  }

  hipLaunchKernelGGL(kernel<int8_t>, dim3(1), dim3(1), 0, 0, inputGPU,
                     gpuHashResult);

  if (hipDeviceSynchronize() != hipSuccess) {
    std::cerr << "hipDeviceSynchronize failed" << std::endl;
    return 1;
  }

  if (hipMemcpy(gpuHashResultHost, gpuHashResult, length * sizeof(uint64_t),
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
