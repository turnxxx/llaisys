#include "rearrange_nvidia.cuh"

#include "../../../core/llaisys_core.hpp"
#include "../../../utils.hpp"

#include <cuda_runtime.h>
#include <cstddef>

namespace llaisys::ops::nvidia {

    namespace {
    constexpr int kMaxRearrangeNDim = 16;
    
    struct RearrangeMeta {
        int ndim;
        size_t shape[kMaxRearrangeNDim];
        ptrdiff_t src_strides[kMaxRearrangeNDim];
        ptrdiff_t dst_strides[kMaxRearrangeNDim];
    };
    
    // 使用模板 T 来指定真实的数据宽度，避免逐字节拷贝
    template <typename T>
    __global__ void rearrange_kernel_optimized(T *dst,
                                               const T *src,
                                               size_t numel,
                                               RearrangeMeta meta) {
        // 引入 Grid-Stride Loop，解决超大 numel 导致的 Grid 越界问题
        const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    
        for (size_t i = tid; i < numel; i += stride) {
            size_t rem = i;
            ptrdiff_t src_offset = 0;
            ptrdiff_t dst_offset = 0;
            
            // 核心坐标计算
            for (int dim = meta.ndim - 1; dim >= 0; --dim) {
                const size_t dim_size = meta.shape[dim];
                const size_t coord = rem % dim_size;
                rem /= dim_size;
                src_offset += static_cast<ptrdiff_t>(coord) * meta.src_strides[dim];
                dst_offset += static_cast<ptrdiff_t>(coord) * meta.dst_strides[dim];
            }
    
            // 现在是一次性读取和写入一个完整的 T (例如 uint32_t)，极大地提升了访存效率
            dst[dst_offset] = src[src_offset];
        }
    }
    } // namespace
    
    void rearrange(tensor_t out, tensor_t in) {
        ASSERT(out != nullptr && in != nullptr, "rearrange: null tensor");
        ASSERT(out->dtype() == in->dtype(), "rearrange: dtype mismatch");
        ASSERT(out->shape() == in->shape(), "rearrange: shape mismatch");
        ASSERT(out->deviceType() == LLAISYS_DEVICE_NVIDIA && in->deviceType() == LLAISYS_DEVICE_NVIDIA,
               "rearrange: only NVIDIA tensors are supported");
        ASSERT(in->ndim() <= static_cast<size_t>(kMaxRearrangeNDim),
               "rearrange: ndim exceeds CUDA implementation limit");
    
        const size_t numel = in->numel();
        if (numel == 0) {
            return;
        }
    
        RearrangeMeta meta{};
        meta.ndim = static_cast<int>(in->ndim());
        const auto &shape = in->shape();
        const auto &src_strides = in->strides();
        const auto &dst_strides = out->strides();
        
        for (int i = 0; i < meta.ndim; ++i) {
            meta.shape[i] = shape[static_cast<size_t>(i)];
            meta.src_strides[i] = src_strides[static_cast<size_t>(i)];
            meta.dst_strides[i] = dst_strides[static_cast<size_t>(i)];
        }
    
        constexpr int block_size = 256;
        // 限制最大 grid_size，配合 kernel 里的 grid-stride loop 使用
        const int num_blocks = std::min(static_cast<int>((numel + block_size - 1) / block_size), 65535);
        
        llaisys::core::context().setDevice(LLAISYS_DEVICE_NVIDIA, out->deviceId());
        auto &runtime = llaisys::core::context().runtime();
        auto stream = reinterpret_cast<cudaStream_t>(runtime.stream());
    
        // 根据元素的字节大小分发到不同的 Typed Kernel
        const size_t elem_size = in->elementSize();
        if (elem_size == 1) {
            rearrange_kernel_optimized<uint8_t><<<num_blocks, block_size, 0, stream>>>(
                reinterpret_cast<uint8_t*>(out->data()), reinterpret_cast<const uint8_t*>(in->data()), numel, meta);
        } else if (elem_size == 2) {
            rearrange_kernel_optimized<uint16_t><<<num_blocks, block_size, 0, stream>>>(
                reinterpret_cast<uint16_t*>(out->data()), reinterpret_cast<const uint16_t*>(in->data()), numel, meta);
        } else if (elem_size == 4) {
            rearrange_kernel_optimized<uint32_t><<<num_blocks, block_size, 0, stream>>>(
                reinterpret_cast<uint32_t*>(out->data()), reinterpret_cast<const uint32_t*>(in->data()), numel, meta);
        } else if (elem_size == 8) {
            rearrange_kernel_optimized<uint64_t><<<num_blocks, block_size, 0, stream>>>(
                reinterpret_cast<uint64_t*>(out->data()), reinterpret_cast<const uint64_t*>(in->data()), numel, meta);
        } else {
            ASSERT(false, "rearrange: unsupported element size");
        }
    
        runtime.api()->stream_synchronize(runtime.stream());
    }
    } // namespace llaisys::ops::nvidia