#include "self_attention_nvidia.cuh"

#include "../../../core/llaisys_core.hpp"
#include "../../../utils.hpp"
#include "../../flah_infer_wrapper.cuh"

namespace llaisys::ops::nvidia {

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    const auto& q_shape = q->shape();
    const auto& k_shape = k->shape();

    int seq_q        = static_cast<int>(q_shape[0]);
    int num_qo_heads = static_cast<int>(q_shape[1]);
    int seq_kv        = static_cast<int>(k_shape[0]);
    int num_kv_heads  = static_cast<int>(k_shape[1]);

    llaisys::core::context().setDevice(LLAISYS_DEVICE_NVIDIA, q->deviceId());
    auto& runtime = llaisys::core::context().runtime();
    auto stream = reinterpret_cast<cudaStream_t>(runtime.stream());

    switch (q->dtype()) {
    case LLAISYS_DTYPE_F16:
        launch_flashinfer_decode_fp16_h128(
            q->data(), k->data(), v->data(), attn_val->data(),
            seq_q, seq_kv, num_qo_heads, num_kv_heads,
            /*is_causal=*/true, scale,
            /*workspace=*/nullptr, /*workspace_size=*/0,
            stream);
        break;
    case LLAISYS_DTYPE_BF16:
        launch_flashinfer_decode_bf16_h128(
            q->data(), k->data(), v->data(), attn_val->data(),
            seq_q, seq_kv, num_qo_heads, num_kv_heads,
            /*is_causal=*/true, scale,
            /*workspace=*/nullptr, /*workspace_size=*/0,
            stream);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(q->dtype());
    }

    runtime.api()->stream_synchronize(runtime.stream());
}

} // namespace llaisys::ops::nvidia
