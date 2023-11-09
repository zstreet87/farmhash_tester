/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef GPU_DEVICE_FUNCTIONS_H_
#define GPU_DEVICE_FUNCTIONS_H_

#include <algorithm>
#include <complex>

#include <hip/hip_complex.h>

using gpuStream_t = hipStream_t;
using gpuEvent_t = hipEvent_t;
using cudaError = int;
using cudaError_t = int;
#define cudaSuccess 0
#define cudaGetLastError hipGetLastError
#define gpuEventRecord hipEventRecord
#define gpuEventDestroy hipEventDestroy
#define gpuEventSynchronize hipEventSynchronize
#define gpuEventCreate hipEventCreate
#define gpuEventCreateWithFlags hipEventCreateWithFlags
#define gpuEventDisableTiming hipEventDisableTiming
#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuFree hipFree
static std::string cudaGetErrorString(int err) { return std::to_string(err); }

#define TF_RETURN_IF_CUDA_ERROR(result)                   \
  do {                                                    \
    cudaError_t error(result);                            \
    if (!SE_PREDICT_TRUE(error == cudaSuccess)) {         \
      return errors::Internal("Cuda call failed with ",   \
                              cudaGetErrorString(error)); \
    }                                                     \
  } while (0)

#define TF_OP_REQUIRES_CUDA_SUCCESS(context, result)                   \
  do {                                                                 \
    cudaError_t error(result);                                         \
    if (!SE_PREDICT_TRUE(error == cudaSuccess)) {                      \
      context->SetStatus(errors::Internal("Cuda call failed with",     \
                                          cudaGetErrorString(error))); \
      return;                                                          \
    }                                                                  \
  } while (0)

namespace tensorflow {
// According to HIP developer guide at
// https://github.com/ROCm-Developer-Tools/HIP/blob/master/docs/markdown/hip_kernel_language.md#assert
// assert is not supported by HIP. While we are waiting for assert support in
// hip kernels, the assert call should be macroed to NOP so that it does not
// block us from creating a debug build
#if TENSORFLOW_USE_ROCM
#undef assert
#define assert(x) \
  {}
#endif

namespace detail {

// Helper for range-based for loop using 'delta' increments.
// Usage: see GpuGridRange?() functions below.
template <typename T>
class GpuGridRange {
  struct Iterator {
    __device__ Iterator(T index, T delta) : index_(index), delta_(delta) {}
    __device__ T operator*() const { return index_; }
    __device__ Iterator& operator++() {
      index_ += delta_;
      return *this;
    }
    __device__ bool operator!=(const Iterator& other) const {
      bool greater = index_ > other.index_;
      bool less = index_ < other.index_;
      // Anything past an end iterator (delta_ == 0) is equal.
      // In range-based for loops, this optimizes to 'return less'.
      if (!other.delta_) {
        return less;
      }
      if (!delta_) {
        return greater;
      }
      return less || greater;
    }

   private:
    T index_;
    const T delta_;
  };

 public:
  __device__ GpuGridRange(T begin, T delta, T end)
      : begin_(begin), delta_(delta), end_(end) {}

  __device__ Iterator begin() const { return Iterator{begin_, delta_}; }
  __device__ Iterator end() const { return Iterator{end_, 0}; }

 private:
  T begin_;
  T delta_;
  T end_;
};
} // namespace detail

// Helper to visit indices in the range 0 <= i < count, using the x-coordinate
// of the global thread index. That is, each index i is visited by all threads
// with the same x-coordinate.
// Usage: for(int i : GpuGridRangeX(count)) { visit(i); }
template <typename T>
__device__ detail::GpuGridRange<T> GpuGridRangeX(T count) {
  return detail::GpuGridRange<T>(
      /*begin=*/blockIdx.x * blockDim.x + threadIdx.x,
      /*delta=*/gridDim.x * blockDim.x, /*end=*/count);
}

} // namespace tensorflow

#endif //GPU_DEVICE_FUNCTIONS_H_
#
