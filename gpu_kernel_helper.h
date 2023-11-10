#ifndef GPU_KERNEL_HELPER_H_
#define GPU_KERNEL_HELPER_H_

#include "gpu_device_functions.h"

// Deprecated, use 'for(int i : GpuGridRangeX(n))' instead.
#define GPU_1D_KERNEL_LOOP(i, n) \
  for (int i : ::tensorflow::GpuGridRangeX<int>(n))

// Launches a GPU kernel through cudaLaunchKernel in CUDA environment, or
// hipLaunchKernel in ROCm environment with the given arguments.
//
// The kernel parameters 'Ts' must be constructible from the arguments 'Args'.
template <typename... Ts, typename... Args>
void GpuLaunchKernel(void (*function)(Ts...), dim3 grid_dim, dim3 block_dim,
                       size_t shared_memory_size_bytes, hipStream_t stream,
                       Args... arguments) {
  if (grid_dim.x * grid_dim.y * grid_dim.z > 0 &&
      block_dim.x * block_dim.y * block_dim.z > 0) {
    hipLaunchKernelGGL(function, grid_dim, block_dim, shared_memory_size_bytes,
                       stream, std::forward<Args>(arguments)...);
  }
}

#endif  // GPU_KERNEL_HELPER_H_
