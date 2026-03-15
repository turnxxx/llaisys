#include "embedding_nvidia.cuh"

#include "../../../core/llaisys_core.hpp"
#include "../../../utils.hpp"

#include <cuda_runtime.h>

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void embedding_kernel(T *out,
                                 const int64_t *index,
                                 const T *weight,
                                 ptrdiff_t out_s0,
                                 ptrdiff_t out_s1,
                                 ptrdiff_t weight_s0,
                                 ptrdiff_t weight_s1,
                                 size_t n,
                                 size_t d) {
    const size_t row = static_cast<size_t>(blockIdx.x);
    if (row >= n) {
        return;
    }

    const size_t col = static_cast<size_t>(blockIdx.y) * blockDim.x + threadIdx.x;
    if (col >= d) {
        return;
    }

    __shared__ int64_t weight_row_shared;
    if (threadIdx.x == 0) {
        weight_row_shared = index[row];
    }
    __syncthreads();

    const int64_t weight_row = weight_row_shared;
    if (weight_row < 0) {
        return;
    }

    ptrdiff_t out_offset = static_cast<ptrdiff_t>(row) * out_s0 + static_cast<ptrdiff_t>(col) * out_s1;
    ptrdiff_t weight_offset = static_cast<ptrdiff_t>(weight_row) * weight_s0 + static_cast<ptrdiff_t>(col) * weight_s1;
    out[out_offset] = weight[weight_offset];
}

template <typename T>
void embedding_impl(tensor_t out, tensor_t index, tensor_t weight) {
    const size_t n = index->numel();
    const size_t d = weight->shape()[1];
    if (n == 0 || d == 0) {
        return;
    }

    auto *out_ptr = reinterpret_cast<T *>(out->data());
    auto *index_ptr = reinterpret_cast<const int64_t *>(index->data());
    auto *weight_ptr = reinterpret_cast<const T *>(weight->data());

    constexpr int block_size = 256;
    const unsigned int grid_x = static_cast<unsigned int>(n);
    const unsigned int grid_y = static_cast<unsigned int>((d + block_size - 1) / block_size);

    ptrdiff_t out_s0 = out->strides()[0];
    ptrdiff_t out_s1 = out->strides()[1];
    ptrdiff_t weight_s0 = weight->strides()[0];
    ptrdiff_t weight_s1 = weight->strides()[1];

    auto &runtime = llaisys::core::context().runtime();
    auto stream = reinterpret_cast<cudaStream_t>(runtime.stream());
    embedding_kernel<T><<<dim3(grid_x, grid_y), block_size, 0, stream>>>(
        out_ptr,
        index_ptr,
        weight_ptr,
        out_s0,
        out_s1,
        weight_s0,
        weight_s1,
        n,
        d);
}

void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    ASSERT(out->ndim() == 2 && index->ndim() == 1 && weight->ndim() == 2,
           "Embedding: out, index and weight must be 2D and 1D tensors");
    ASSERT(out->shape()[0] == index->shape()[0] && out->shape()[1] == weight->shape()[1],
           "Embedding: out shape must be [index_numel, weight_dim]");
    ASSERT(index->dtype() == LLAISYS_DTYPE_I64,
           "Embedding: Data Type of index must be LLAISYS_DTYPE_I64");
    ASSERT(out->deviceId() == index->deviceId() && out->deviceId() == weight->deviceId(),
           "Embedding: device_id of out, index and weight must be the same");
    llaisys::core::context().setDevice(LLAISYS_DEVICE_NVIDIA, out->deviceId());

    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
        embedding_impl<float>(out, index, weight);
        break;
    case LLAISYS_DTYPE_BF16:
        embedding_impl<llaisys::bf16_t>(out, index, weight);
        break;
    case LLAISYS_DTYPE_F16:
        embedding_impl<llaisys::fp16_t>(out, index, weight);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }

    auto &runtime = llaisys::core::context().runtime();
    runtime.api()->stream_synchronize(runtime.stream());
}
} // namespace llaisys::ops::nvidia
