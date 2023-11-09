#ifndef GPU_KERNEL_HELPER_H_
#define GPU_KERNEL_HELPER_H_

#include "gpu_device_functions.h"

// Deprecated, use 'for(int i : GpuGridRangeX(n))' instead.
#define GPU_1D_KERNEL_LOOP(i, n) \
  for (int i : ::tensorflow::GpuGridRangeX<int>(n))

#endif  // GPU_KERNEL_HELPER_H_
