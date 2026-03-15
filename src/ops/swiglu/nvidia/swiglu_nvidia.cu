#include "swiglu_nvidia.cuh"

#include "../../../core/llaisys_core.hpp"
#include "../../../utils.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace llaisys::ops::nvidia {

namespace {
inline void check_cuda(cudaError_t err, const char *msg) {
    ASSERT(err == cudaSuccess, msg);
}

__device__ __forceinline__ float sigmoidf_fast(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void swiglu_kernel_f32(float *out,
                                  const float *gate,
                                  const float *up,
                                  size_t numel) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= numel) {
        return;
    }
    const float g = gate[idx];
    out[idx] = up[idx] * g * sigmoidf_fast(g);
}

__global__ void swiglu_kernel_f16(llaisys::fp16_t *out,
                                  const llaisys::fp16_t *gate,
                                  const llaisys::fp16_t *up,
                                  size_t numel) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= numel) {
        return;
    }
    const __half g_h = reinterpret_cast<const __half *>(gate)[idx];
    const __half u_h = reinterpret_cast<const __half *>(up)[idx];
    const float g = __half2float(g_h);
    const float u = __half2float(u_h);
    const float y = u * g * sigmoidf_fast(g);
    reinterpret_cast<__half *>(out)[idx] = __float2half_rn(y);
}

__global__ void swiglu_kernel_bf16(llaisys::bf16_t *out,
                                   const llaisys::bf16_t *gate,
                                   const llaisys::bf16_t *up,
                                   size_t numel) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= numel) {
        return;
    }
    const __nv_bfloat16 g_b = reinterpret_cast<const __nv_bfloat16 *>(gate)[idx];
    const __nv_bfloat16 u_b = reinterpret_cast<const __nv_bfloat16 *>(up)[idx];
    const float g = __bfloat162float(g_b);
    const float u = __bfloat162float(u_b);
    const float y = u * g * sigmoidf_fast(g);
    reinterpret_cast<__nv_bfloat16 *>(out)[idx] = __float2bfloat16_rn(y);
}
} // namespace

void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    const size_t numel = out->numel();
    if (numel == 0) {
        return;
    }

    constexpr int block_size = 256;
    const int num_blocks = static_cast<int>((numel + block_size - 1) / block_size);

    llaisys::core::context().setDevice(LLAISYS_DEVICE_NVIDIA, out->deviceId());
    auto &runtime = llaisys::core::context().runtime();
    auto stream = runtime.stream();
    auto cu_stream = reinterpret_cast<cudaStream_t>(stream);

    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
        swiglu_kernel_f32<<<num_blocks, block_size, 0, cu_stream>>>(
            reinterpret_cast<float *>(out->data()),
            reinterpret_cast<const float *>(gate->data()),
            reinterpret_cast<const float *>(up->data()),
            numel);
        check_cuda(cudaGetLastError(), "SwiGLU(NVIDIA): f32 kernel launch failed");
        break;
    case LLAISYS_DTYPE_F16:
        swiglu_kernel_f16<<<num_blocks, block_size, 0, cu_stream>>>(
            reinterpret_cast<llaisys::fp16_t *>(out->data()),
            reinterpret_cast<const llaisys::fp16_t *>(gate->data()),
            reinterpret_cast<const llaisys::fp16_t *>(up->data()),
            numel);
        check_cuda(cudaGetLastError(), "SwiGLU(NVIDIA): f16 kernel launch failed");
        break;
    case LLAISYS_DTYPE_BF16:
        swiglu_kernel_bf16<<<num_blocks, block_size, 0, cu_stream>>>(
            reinterpret_cast<llaisys::bf16_t *>(out->data()),
            reinterpret_cast<const llaisys::bf16_t *>(gate->data()),
            reinterpret_cast<const llaisys::bf16_t *>(up->data()),
            numel);
        check_cuda(cudaGetLastError(), "SwiGLU(NVIDIA): bf16 kernel launch failed");
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }

    auto api = runtime.api();
    api->stream_synchronize(stream);
}
} // namespace llaisys::ops::nvidia
