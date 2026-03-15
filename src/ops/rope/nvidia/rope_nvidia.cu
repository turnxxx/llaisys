#include "rope_nvidia.cuh"

#include "../../../core/llaisys_core.hpp"
#include "../../../utils.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>

namespace llaisys::ops::nvidia {

namespace {
__global__ void rope_kernel_f32(float *out,
                                const float *in,
                                const int64_t *pos_ids,
                                float log_theta,
                                int nhead,
                                int half_d,
                                int d) {
    const int seq = static_cast<int>(blockIdx.x);
    const int head = static_cast<int>(blockIdx.y);
    const int base = seq * (nhead * d) + head * d;

    const double p = static_cast<double>(pos_ids[seq]);
    const double d_log_theta = static_cast<double>(log_theta);

    for (int k = static_cast<int>(threadIdx.x); k < half_d; k += blockDim.x) {
        const double inv_freq = exp((-2.0 * static_cast<double>(k) / static_cast<double>(d)) * d_log_theta);
        const double phi = p * inv_freq;

        float s, c;
        sincosf(static_cast<float>(phi), &s, &c);

        const int idx_a = base + k;
        const int idx_b = base + k + half_d;
        const float a = in[idx_a];
        const float b = in[idx_b];
        out[idx_a] = a * c - b * s;
        out[idx_b] = b * c + a * s;
    }
}

__global__ void rope_kernel_f16(__half *out,
                                const __half *in,
                                const int64_t *pos_ids,
                                float log_theta,
                                int nhead,
                                int half_d,
                                int d) {
    const int seq = static_cast<int>(blockIdx.x);
    const int head = static_cast<int>(blockIdx.y);
    const int base = seq * (nhead * d) + head * d;

    const double p = static_cast<double>(pos_ids[seq]);
    const double d_log_theta = static_cast<double>(log_theta);

    for (int k = static_cast<int>(threadIdx.x); k < half_d; k += blockDim.x) {
        const double inv_freq = exp((-2.0 * static_cast<double>(k) / static_cast<double>(d)) * d_log_theta);
        const double phi = p * inv_freq;

        float s, c;
        sincosf(static_cast<float>(phi), &s, &c);

        const int idx_a = base + k;
        const int idx_b = base + k + half_d;
        const float a = __half2float(in[idx_a]);
        const float b = __half2float(in[idx_b]);
        out[idx_a] = __float2half_rn(a * c - b * s);
        out[idx_b] = __float2half_rn(b * c + a * s);
    }
}

__global__ void rope_kernel_bf16(__nv_bfloat16 *out,
                                 const __nv_bfloat16 *in,
                                 const int64_t *pos_ids,
                                 float log_theta,
                                 int nhead,
                                 int half_d,
                                 int d) {
    const int seq = static_cast<int>(blockIdx.x);
    const int head = static_cast<int>(blockIdx.y);
    const int base = seq * (nhead * d) + head * d;

    const double p = static_cast<double>(pos_ids[seq]);
    const double d_log_theta = static_cast<double>(log_theta);

    for (int k = static_cast<int>(threadIdx.x); k < half_d; k += blockDim.x) {
        const double inv_freq = exp((-2.0 * static_cast<double>(k) / static_cast<double>(d)) * d_log_theta);
        const double phi = p * inv_freq;

        float s, c;
        sincosf(static_cast<float>(phi), &s, &c);

        const int idx_a = base + k;
        const int idx_b = base + k + half_d;
        const float a = __bfloat162float(in[idx_a]);
        const float b = __bfloat162float(in[idx_b]);
        out[idx_a] = __float2bfloat16_rn(a * c - b * s);
        out[idx_b] = __float2bfloat16_rn(b * c + a * s);
    }
}
} // namespace

void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    ASSERT(out != nullptr && in != nullptr && pos_ids != nullptr, "Rope: null tensor");
    ASSERT(in->ndim() == 3, "Rope: input must be 3D [seqlen, nhead, d]");
    ASSERT(out->isContiguous() && in->isContiguous(),
           "Rope(NVIDIA): currently requires contiguous in/out");
    ASSERT(theta > 0.0f, "Rope: theta must be positive");

    const int seqlen = static_cast<int>(in->shape()[0]);
    const int nhead = static_cast<int>(in->shape()[1]);
    const int d = static_cast<int>(in->shape()[2]);
    ASSERT((d % 2) == 0, "Rope: last dimension must be even");

    const int half_d = d / 2;
    if (seqlen == 0 || nhead == 0 || half_d == 0) {
        return;
    }

    const float log_theta = std::log(theta);

    constexpr int block_size = 256;
    dim3 grid(static_cast<unsigned int>(seqlen),
              static_cast<unsigned int>(nhead),
              1u);

    llaisys::core::context().setDevice(LLAISYS_DEVICE_NVIDIA, out->deviceId());
    auto &runtime = llaisys::core::context().runtime();
    auto stream = runtime.stream();
    auto cu_stream = reinterpret_cast<cudaStream_t>(stream);
    auto *pos_ptr = reinterpret_cast<const int64_t *>(pos_ids->data());

    switch (in->dtype()) {
    case LLAISYS_DTYPE_F32:
        rope_kernel_f32<<<grid, block_size, 0, cu_stream>>>(
            reinterpret_cast<float *>(out->data()),
            reinterpret_cast<const float *>(in->data()),
            pos_ptr, log_theta, nhead, half_d, d);
        break;
    case LLAISYS_DTYPE_F16:
        rope_kernel_f16<<<grid, block_size, 0, cu_stream>>>(
            reinterpret_cast<__half *>(out->data()),
            reinterpret_cast<const __half *>(in->data()),
            pos_ptr, log_theta, nhead, half_d, d);
        break;
    case LLAISYS_DTYPE_BF16:
        rope_kernel_bf16<<<grid, block_size, 0, cu_stream>>>(
            reinterpret_cast<__nv_bfloat16 *>(out->data()),
            reinterpret_cast<const __nv_bfloat16 *>(in->data()),
            pos_ptr, log_theta, nhead, half_d, d);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(in->dtype());
    }

    auto api = runtime.api();
    api->stream_synchronize(stream);
}
} // namespace llaisys::ops::nvidia