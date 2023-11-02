#include <cstring>
#include <hip/hip_runtime.h>
#include <iostream>

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
__global__ void hipFarmHash(const T *__restrict__ vals, size_t length,
                            uint64_t *output) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  char s[] = " - ";
  if (idx < length) {
    int size = IntegerToString(vals[idx],
                               s + threadIdx.x * kSharedMemBufferSizePerThread);
    uint64_t a_hash = ::util_gpu::Fingerprint64(
        s + threadIdx.x * kSharedMemBufferSizePerThread, size);
  }
}

int main() {
  const char *inputString = "MY_STRING";
  size_t length = std::strlen(inputString);
  uint64_t *gpuHashResult;
  uint64_t cpuHashResult;

  const int64_t input = 139753374614784;

  hipMalloc(&gpuHashResult, length * sizeof(uint64_t));

  hipLaunchKernelGGL(hipFarmHash<int64_t>, dim3(1), dim3(1), 0, 0, &input, 1,
                     gpuHashResult);

  // Just single gpu so no sync needed
  // hipDeviceSynchronize();

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
    std::cout << gpuHashResult[0] << " " << cpuHashResult << "\n" << std::endl;
  }

  // this would make my dev node hang?...
  // hipFree(gpuHashResult);

  return 0;
}
