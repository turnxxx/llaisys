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

void self_attention_paged(tensor_t attn_val,
                          tensor_t q,
                          tensor_t paged_kv_data,
                          tensor_t kv_indptr,
                          tensor_t kv_indices,
                          tensor_t kv_last_page_len,
                          int page_size,
                          float scale) {
    const auto& q_shape = q->shape();
    const auto& kv_shape = paged_kv_data->shape();
    ASSERT(q_shape.size() == 3, "self_attention_paged: q should be [batch,qo_heads,head_dim]");
    ASSERT(kv_shape.size() >= 5,
           "self_attention_paged: paged_kv_data should be [num_pages,2,kv_heads,page,head_dim]");
    ASSERT(static_cast<int>(q_shape[2]) == 128, "self_attention_paged: only head_dim=128 is supported");
    ASSERT(page_size > 0, "self_attention_paged: page_size must be > 0");
    ASSERT(static_cast<int>(kv_shape[1]) == 2,
           "self_attention_paged: paged_kv_data second dim must be 2 (K/V)");
    ASSERT(static_cast<int>(kv_shape[3]) == page_size,
           "self_attention_paged: page_size mismatch with paged_kv_data shape");
    ASSERT(static_cast<int>(kv_shape[4]) == 128,
           "self_attention_paged: paged_kv_data head_dim must be 128");

    int batch_size = static_cast<int>(q_shape[0]);
    int num_qo_heads = static_cast<int>(q_shape[1]);
    int num_kv_heads = static_cast<int>(kv_shape[2]);

    ASSERT(static_cast<int>(kv_indptr->shape()[0]) == batch_size + 1,
           "self_attention_paged: kv_indptr shape must be [batch+1]");
    ASSERT(static_cast<int>(kv_last_page_len->shape()[0]) == batch_size,
           "self_attention_paged: kv_last_page_len shape must be [batch]");

    llaisys::core::context().setDevice(LLAISYS_DEVICE_NVIDIA, q->deviceId());
    auto& runtime = llaisys::core::context().runtime();
    auto stream = reinterpret_cast<cudaStream_t>(runtime.stream());

    switch (q->dtype()) {
    case LLAISYS_DTYPE_F16:
        launch_flashinfer_decode_fp16_h128_pagedattn(
            q->data(),
            paged_kv_data->data(),
            reinterpret_cast<int32_t*>(kv_indptr->data()),
            reinterpret_cast<int32_t*>(kv_indices->data()),
            reinterpret_cast<int32_t*>(kv_last_page_len->data()),
            attn_val->data(),
            batch_size,
            num_qo_heads,
            num_kv_heads,
            page_size,
            scale,
            /*workspace=*/nullptr,
            /*workspace_size=*/0,
            stream);
        break;
    case LLAISYS_DTYPE_BF16:
        launch_flashinfer_decode_bf16_h128_pagedattn(
            q->data(),
            paged_kv_data->data(),
            reinterpret_cast<int32_t*>(kv_indptr->data()),
            reinterpret_cast<int32_t*>(kv_indices->data()),
            reinterpret_cast<int32_t*>(kv_last_page_len->data()),
            attn_val->data(),
            batch_size,
            num_qo_heads,
            num_kv_heads,
            page_size,
            scale,
            /*workspace=*/nullptr,
            /*workspace_size=*/0,
            stream);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(q->dtype());
    }

    runtime.api()->stream_synchronize(runtime.stream());
}

} // namespace llaisys::ops::nvidia
