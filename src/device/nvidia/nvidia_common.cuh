#pragma once

#include "llaisys.h"

#include <cublas_v2.h>
#include <cstddef>

namespace llaisys::device::nvidia {

// Light-weight descriptor for a chunk of device memory.
struct DeviceBufferView {
    std::byte *ptr{nullptr};
    size_t bytes{0};
    int device_id{0};
};

// Generic launch description used by op interfaces.
struct KernelLaunchConfig {
    unsigned int grid_x{1};
    unsigned int grid_y{1};
    unsigned int grid_z{1};
    unsigned int block_x{1};
    unsigned int block_y{1};
    unsigned int block_z{1};
    size_t shared_mem_bytes{0};
    llaisysStream_t stream{nullptr};
};

// Per-op execution context for future CUDA integration.
// cublas的handle也放在这里
struct OpExecutionContext {
    int device_id{0};
    llaisysStream_t stream{nullptr};
    cublasHandle_t cublas_handle{nullptr};
};

} // namespace llaisys::device::nvidia
