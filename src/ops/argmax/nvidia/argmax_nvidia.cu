#include "argmax_nvidia.cuh"

#include "../../../core/llaisys_core.hpp"
#include "../../../utils.hpp"
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
namespace llaisys::ops::nvidia {

template <typename T>
__device__ float to_float_val(T val);

template <>
__device__ float to_float_val<float>(float val) { return val; }

template <>
__device__ float to_float_val<__half>(__half val) { return __half2float(val); }

template <>
__device__ float to_float_val<__nv_bfloat16>(__nv_bfloat16 val) { return __bfloat162float(val); }

// 单 block 归约：每个线程先用 stride 循环找到自己负责的局部最大值，
// 然后在 shared memory 中做树形归约得到全局 argmax
template <typename T>
__global__ void argmax_kernel(size_t *max_idx, T *max_val, const T *vals, size_t numel) {
    extern __shared__ char smem[];
    float *s_val = reinterpret_cast<float *>(smem);
    size_t *s_idx = reinterpret_cast<size_t *>(smem + blockDim.x * sizeof(float));

    unsigned int tid = threadIdx.x;

    float local_max = to_float_val(vals[0]);
    size_t local_idx = 0;
    for (size_t i = tid; i < numel; i += blockDim.x) {
        float v = to_float_val(vals[i]);
        if (v > local_max) {
            local_max = v;
            local_idx = i;
        }
    }

    s_val[tid] = local_max;
    s_idx[tid] = local_idx;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_val[tid + s] > s_val[tid]) {
                s_val[tid] = s_val[tid + s];
                s_idx[tid] = s_idx[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *max_val = vals[s_idx[0]];
        *max_idx = s_idx[0];
    }
}

template <typename T>
void argmax_thrust(tensor_t max_idx, tensor_t max_val, tensor_t vals){
    ASSERT(vals->numel() > 0, "Argmax: vals must contain at least one element");
    auto &runtime = llaisys::core::context().runtime();
    auto stream = reinterpret_cast<cudaStream_t>(runtime.stream());

    auto *vals_ptr = reinterpret_cast<const T *>(vals->data());
    auto *max_val_ptr = reinterpret_cast<T *>(max_val->data());
    auto *max_idx_ptr = reinterpret_cast<size_t *>(max_idx->data());

    auto begin = thrust::device_pointer_cast(vals_ptr);
    auto end = begin + vals->numel();

    auto max_it = thrust::max_element(thrust::cuda::par.on(stream), begin, end);
    size_t idx = static_cast<size_t>(max_it - begin);

    runtime.api()->memcpy_async(
        max_val_ptr,
        thrust::raw_pointer_cast(max_it),
        sizeof(T),
        LLAISYS_MEMCPY_D2D,
        runtime.stream());
    // idx is a host stack variable; use sync copy to avoid async lifetime hazards.
    runtime.api()->memcpy_sync(
        max_idx_ptr,
        &idx,
        sizeof(size_t),
        LLAISYS_MEMCPY_H2D);
}
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());
    ASSERT(max_idx->ndim() == 1 && max_val->ndim() == 1 && vals->ndim() == 1,
           "Argmax: max_idx, max_val and vals must be 1D tensors");
    ASSERT(max_idx->shape()[0] == 1 && max_val->shape()[0] == 1,
           "Argmax: max_idx and max_val must be 1D tensors with a single element");
    ASSERT(max_idx->dtype() == LLAISYS_DTYPE_I64,
           "Argmax: Data Type of max_idx must be LLAISYS_DTYPE_I64");
    ASSERT(max_idx->deviceId() == max_val->deviceId() && max_idx->deviceId() == vals->deviceId(),
           "Argmax: device_id of max_idx, max_val and vals must be the same");
    llaisys::core::context().setDevice(LLAISYS_DEVICE_NVIDIA, max_idx->deviceId());

    switch (vals->dtype()) {
    case LLAISYS_DTYPE_F32:
        argmax_thrust<float>(max_idx, max_val, vals);
        break;
    case LLAISYS_DTYPE_BF16:
        argmax_thrust<__nv_bfloat16>(max_idx, max_val, vals);
        break;
    case LLAISYS_DTYPE_F16:
        argmax_thrust<__half>(max_idx, max_val, vals);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(vals->dtype());
    }

    auto &runtime = llaisys::core::context().runtime();
    auto stream = runtime.stream();
    runtime.api()->stream_synchronize(stream);
}
} // namespace llaisys::ops::nvidia
