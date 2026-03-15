#include "rms_norm_nvidia.cuh"

#include "../../../core/llaisys_core.hpp"
#include "../../../utils.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace llaisys::ops::nvidia {

namespace {
__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float blockReduceSum(float val) {
    __shared__ float shared[32];
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;

    val = warpReduceSum(val);
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    const int warp_count = (blockDim.x + 31) / 32;
    val = (threadIdx.x < warp_count) ? shared[lane] : 0.0f;
    if (wid == 0) {
        val = warpReduceSum(val);
    }
    return val;
}

__global__ void rmsnorm_kernel_f32(float *out,
                                   const float *input,
                                   const float *weight,
                                   float eps,
                                   int d) {
    const int row = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);

    const float *in_row = input + static_cast<size_t>(row) * d;
    float *out_row = out + static_cast<size_t>(row) * d;

    const bool can_vec4 =
        ((reinterpret_cast<uintptr_t>(in_row) & (alignof(float4) - 1)) == 0) &&
        ((reinterpret_cast<uintptr_t>(out_row) & (alignof(float4) - 1)) == 0) &&
        ((reinterpret_cast<uintptr_t>(weight) & (alignof(float4) - 1)) == 0);

    float sum_sq = 0.0f;
    if (can_vec4) {
        const int vec_elems = d / 4;
        const float4 *in_v4 = reinterpret_cast<const float4 *>(in_row);
        for (int i = tid; i < vec_elems; i += blockDim.x) {
            const float4 v = in_v4[i];
            sum_sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
        for (int i = vec_elems * 4 + tid; i < d; i += blockDim.x) {
            const float v = in_row[i];
            sum_sq += v * v;
        }
    } else {
        for (int i = tid; i < d; i += blockDim.x) {
            const float v = in_row[i];
            sum_sq += v * v;
        }
    }

    sum_sq = blockReduceSum(sum_sq);

    __shared__ float inv_rms;
    if (tid == 0) {
        inv_rms = rsqrtf(sum_sq / static_cast<float>(d) + eps);
    }
    __syncthreads();

    if (can_vec4) {
        const int vec_elems = d / 4;
        const float4 *in_v4 = reinterpret_cast<const float4 *>(in_row);
        const float4 *w_v4 = reinterpret_cast<const float4 *>(weight);
        float4 *out_v4 = reinterpret_cast<float4 *>(out_row);
        for (int i = tid; i < vec_elems; i += blockDim.x) {
            const float4 in_v = in_v4[i];
            const float4 w_v = w_v4[i];
            float4 o;
            o.x = in_v.x * inv_rms * w_v.x;
            o.y = in_v.y * inv_rms * w_v.y;
            o.z = in_v.z * inv_rms * w_v.z;
            o.w = in_v.w * inv_rms * w_v.w;
            out_v4[i] = o;
        }
        for (int i = vec_elems * 4 + tid; i < d; i += blockDim.x) {
            out_row[i] = in_row[i] * inv_rms * weight[i];
        }
    } else {
        for (int i = tid; i < d; i += blockDim.x) {
            out_row[i] = in_row[i] * inv_rms * weight[i];
        }
    }
}

__global__ void rmsnorm_kernel_f16(llaisys::fp16_t *out,
                                   const llaisys::fp16_t *input,
                                   const llaisys::fp16_t *weight,
                                   float eps,
                                   int d) {
    const int row = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);

    const __half *in_row = reinterpret_cast<const __half *>(input + static_cast<size_t>(row) * d);
    __half *out_row = reinterpret_cast<__half *>(out + static_cast<size_t>(row) * d);
    const __half *w_row = reinterpret_cast<const __half *>(weight);

    const bool can_vec2 =
        ((reinterpret_cast<uintptr_t>(in_row) & (alignof(__half2) - 1)) == 0) &&
        ((reinterpret_cast<uintptr_t>(out_row) & (alignof(__half2) - 1)) == 0) &&
        ((reinterpret_cast<uintptr_t>(w_row) & (alignof(__half2) - 1)) == 0);

    float sum_sq = 0.0f;
    if (can_vec2) {
        const int vec_elems = d / 2;
        const __half2 *in_v2 = reinterpret_cast<const __half2 *>(in_row);
        for (int i = tid; i < vec_elems; i += blockDim.x) {
            const float2 v = __half22float2(in_v2[i]);
            sum_sq += v.x * v.x + v.y * v.y;
        }
        for (int i = vec_elems * 2 + tid; i < d; i += blockDim.x) {
            const float v = __half2float(in_row[i]);
            sum_sq += v * v;
        }
    } else {
        for (int i = tid; i < d; i += blockDim.x) {
            const float v = __half2float(in_row[i]);
            sum_sq += v * v;
        }
    }

    sum_sq = blockReduceSum(sum_sq);

    __shared__ float inv_rms;
    if (tid == 0) {
        inv_rms = rsqrtf(sum_sq / static_cast<float>(d) + eps);
    }
    __syncthreads();

    if (can_vec2) {
        const int vec_elems = d / 2;
        const __half2 *in_v2 = reinterpret_cast<const __half2 *>(in_row);
        const __half2 *w_v2 = reinterpret_cast<const __half2 *>(w_row);
        __half2 *out_v2 = reinterpret_cast<__half2 *>(out_row);
        for (int i = tid; i < vec_elems; i += blockDim.x) {
            const float2 in_f = __half22float2(in_v2[i]);
            const float2 w_f = __half22float2(w_v2[i]);
            const float o0 = in_f.x * inv_rms * w_f.x;
            const float o1 = in_f.y * inv_rms * w_f.y;
            out_v2[i] = __floats2half2_rn(o0, o1);
        }
        for (int i = vec_elems * 2 + tid; i < d; i += blockDim.x) {
            const float in_f = __half2float(in_row[i]);
            const float w_f = __half2float(w_row[i]);
            out_row[i] = __float2half_rn(in_f * inv_rms * w_f);
        }
    } else {
        for (int i = tid; i < d; i += blockDim.x) {
            const float in_f = __half2float(in_row[i]);
            const float w_f = __half2float(w_row[i]);
            out_row[i] = __float2half_rn(in_f * inv_rms * w_f);
        }
    }
}

__global__ void rmsnorm_kernel_bf16(llaisys::bf16_t *out,
                                    const llaisys::bf16_t *input,
                                    const llaisys::bf16_t *weight,
                                    float eps,
                                    int d) {
    const int row = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);

    const __nv_bfloat16 *in_row = reinterpret_cast<const __nv_bfloat16 *>(input + static_cast<size_t>(row) * d);
    __nv_bfloat16 *out_row = reinterpret_cast<__nv_bfloat16 *>(out + static_cast<size_t>(row) * d);
    const __nv_bfloat16 *w_row = reinterpret_cast<const __nv_bfloat16 *>(weight);

    const bool can_vec2 =
        ((reinterpret_cast<uintptr_t>(in_row) & (alignof(__nv_bfloat162) - 1)) == 0) &&
        ((reinterpret_cast<uintptr_t>(out_row) & (alignof(__nv_bfloat162) - 1)) == 0) &&
        ((reinterpret_cast<uintptr_t>(w_row) & (alignof(__nv_bfloat162) - 1)) == 0);

    float sum_sq = 0.0f;
    if (can_vec2) {
        const int vec_elems = d / 2;
        const __nv_bfloat162 *in_v2 = reinterpret_cast<const __nv_bfloat162 *>(in_row);
        for (int i = tid; i < vec_elems; i += blockDim.x) {
            const float2 v = __bfloat1622float2(in_v2[i]);
            sum_sq += v.x * v.x + v.y * v.y;
        }
        for (int i = vec_elems * 2 + tid; i < d; i += blockDim.x) {
            const float v = __bfloat162float(in_row[i]);
            sum_sq += v * v;
        }
    } else {
        for (int i = tid; i < d; i += blockDim.x) {
            const float v = __bfloat162float(in_row[i]);
            sum_sq += v * v;
        }
    }

    sum_sq = blockReduceSum(sum_sq);

    __shared__ float inv_rms;
    if (tid == 0) {
        inv_rms = rsqrtf(sum_sq / static_cast<float>(d) + eps);
    }
    __syncthreads();

    if (can_vec2) {
        const int vec_elems = d / 2;
        const __nv_bfloat162 *in_v2 = reinterpret_cast<const __nv_bfloat162 *>(in_row);
        const __nv_bfloat162 *w_v2 = reinterpret_cast<const __nv_bfloat162 *>(w_row);
        __nv_bfloat162 *out_v2 = reinterpret_cast<__nv_bfloat162 *>(out_row);
        for (int i = tid; i < vec_elems; i += blockDim.x) {
            const float2 in_f = __bfloat1622float2(in_v2[i]);
            const float2 w_f = __bfloat1622float2(w_v2[i]);
            const float o0 = in_f.x * inv_rms * w_f.x;
            const float o1 = in_f.y * inv_rms * w_f.y;
            out_v2[i] = __floats2bfloat162_rn(o0, o1);
        }
        for (int i = vec_elems * 2 + tid; i < d; i += blockDim.x) {
            const float in_f = __bfloat162float(in_row[i]);
            const float w_f = __bfloat162float(w_row[i]);
            out_row[i] = __float2bfloat16(in_f * inv_rms * w_f);
        }
    } else {
        for (int i = tid; i < d; i += blockDim.x) {
            const float in_f = __bfloat162float(in_row[i]);
            const float w_f = __bfloat162float(w_row[i]);
            out_row[i] = __float2bfloat16(in_f * inv_rms * w_f);
        }
    }
}
} // namespace

void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    const int n = static_cast<int>(in->shape()[0]);
    const int d = static_cast<int>(in->shape()[1]);
    if (n == 0 || d == 0) {
        return;
    }

    llaisys::core::context().setDevice(LLAISYS_DEVICE_NVIDIA, out->deviceId());
    auto &runtime = llaisys::core::context().runtime();
    auto stream = reinterpret_cast<cudaStream_t>(runtime.stream());
    constexpr int block_size = 256;

    switch (in->dtype()) {
    case LLAISYS_DTYPE_F32:
        rmsnorm_kernel_f32<<<n, block_size, 0, stream>>>(
            reinterpret_cast<float *>(out->data()),
            reinterpret_cast<const float *>(in->data()),
            reinterpret_cast<const float *>(weight->data()),
            eps,
            d);
        break;
    case LLAISYS_DTYPE_F16:
        rmsnorm_kernel_f16<<<n, block_size, 0, stream>>>(
            reinterpret_cast<llaisys::fp16_t *>(out->data()),
            reinterpret_cast<const llaisys::fp16_t *>(in->data()),
            reinterpret_cast<const llaisys::fp16_t *>(weight->data()),
            eps,
            d);
        break;
    case LLAISYS_DTYPE_BF16:
        rmsnorm_kernel_bf16<<<n, block_size, 0, stream>>>(
            reinterpret_cast<llaisys::bf16_t *>(out->data()),
            reinterpret_cast<const llaisys::bf16_t *>(in->data()),
            reinterpret_cast<const llaisys::bf16_t *>(weight->data()),
            eps,
            d);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(in->dtype());
    }

    runtime.api()->stream_synchronize(runtime.stream());
}
} // namespace llaisys::ops::nvidia
