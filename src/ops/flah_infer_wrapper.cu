// FlashInfer 非 Paged Attention wrapper
// 本文件编译为独立的 .o，隔离 FlashInfer 重量级模板实例化，
// 其他编译单元只需链接此 .o 即可调用 attention 功能。

#include "flah_infer_wrapper.cuh"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#include <flashinfer/attention_impl.cuh>
#include <flashinfer/attention/decode.cuh>
#include <flashinfer/page.cuh>

namespace {

inline void check_cuda_error(cudaError_t err, const char* func, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "[FlashInfer Wrapper] CUDA error at %s:%d in %s: %s\n",
                file, line, func, cudaGetErrorString(err));
    }
}

#define FI_CUDA_CHECK(err) check_cuda_error((err), __func__, __FILE__, __LINE__)

// seq_q == 1: SingleDecode (逐 token 推理, 每次只处理一个 query token)
// seq_q >  1: SinglePrefill (支持 causal mask, 适用于 prefill 阶段)
template <typename DType>
void launch_flashinfer_attention_h128_impl(
    void* q, void* k, void* v, void* out,
    int seq_q, int seq_kv,
    int num_qo_heads, int num_kv_heads,
    bool is_causal, float sm_scale,
    void* workspace, size_t workspace_size,
    cudaStream_t stream)
{
    using namespace flashinfer;
    constexpr uint32_t HEAD_DIM = 128;

    DType* q_ptr = static_cast<DType*>(q);
    DType* k_ptr = static_cast<DType*>(k);
    DType* v_ptr = static_cast<DType*>(v);
    DType* o_ptr = static_cast<DType*>(out);
    DType* tmp_ptr = (workspace && workspace_size > 0)
                     ? static_cast<DType*>(workspace)
                     : nullptr;

    using Variant = DefaultAttention<
        /*use_custom_mask=*/false,
        /*use_sliding_window=*/false,
        /*use_logits_soft_cap=*/false,
        /*use_alibi=*/false>;

    // FlashInfer decode 内核仅支持 group_size ∈ {1,2,3,4,8}，
    // 不匹配时回退到 prefill 路径（seq_q=1 的 prefill 功能等价于 decode）
    bool use_decode = (seq_q == 1);
    if (use_decode) {
        uint32_t group = static_cast<uint32_t>(num_qo_heads) / static_cast<uint32_t>(num_kv_heads);
        if (group != 1 && group != 2 && group != 3 && group != 4 && group != 8) {
            use_decode = false;
        }
    }

    if (use_decode) {
        // ---- Decode 路径 ----
        SingleDecodeParams<DType, DType, DType> params(
            q_ptr, k_ptr, v_ptr, o_ptr,
            /*maybe_alibi_slopes=*/nullptr,
            static_cast<uint32_t>(seq_kv),
            static_cast<uint32_t>(num_qo_heads),
            static_cast<uint32_t>(num_kv_heads),
            QKVLayout::kNHD,
            HEAD_DIM,
            /*window_left=*/-1,
            /*logits_soft_cap=*/0.0f,
            sm_scale,
            /*rope_scale=*/1.0f,
            /*rope_theta=*/1e4f
        );

        FI_CUDA_CHECK(
            (SingleDecodeWithKVCacheDispatched<
                HEAD_DIM,
                PosEncodingMode::kNone,
                Variant>(params, tmp_ptr, stream))
        );
    } else {
        // ---- Prefill 路径 (也处理不支持 decode group_size 的 seq_q=1 情况) ----
        const uint32_t q_stride_n = static_cast<uint32_t>(num_qo_heads) * HEAD_DIM;
        const uint32_t q_stride_h = HEAD_DIM;
        const uint32_t kv_stride_n = static_cast<uint32_t>(num_kv_heads) * HEAD_DIM;
        const uint32_t kv_stride_h = HEAD_DIM;

        SinglePrefillParams<DType, DType, DType> params(
            q_ptr, k_ptr, v_ptr,
            /*maybe_custom_mask=*/nullptr,
            o_ptr,
            /*lse=*/nullptr,
            /*maybe_alibi_slopes=*/nullptr,
            static_cast<uint32_t>(num_qo_heads),
            static_cast<uint32_t>(num_kv_heads),
            static_cast<uint32_t>(seq_q),
            static_cast<uint32_t>(seq_kv),
            q_stride_n, q_stride_h,
            kv_stride_n, kv_stride_h,
            HEAD_DIM,
            /*window_left=*/-1,
            /*logits_soft_cap=*/0.0f,
            sm_scale,
            /*rope_scale=*/1.0f,
            /*rope_theta=*/1e4f
        );

        if (is_causal) {
            FI_CUDA_CHECK(
                (SinglePrefillWithKVCacheDispatched<
                    HEAD_DIM, HEAD_DIM,
                    PosEncodingMode::kNone,
                    /*USE_FP16_QK_REDUCTION=*/false,
                    MaskMode::kCausal,
                    Variant>(params, tmp_ptr, stream))
            );
        } else {
            FI_CUDA_CHECK(
                (SinglePrefillWithKVCacheDispatched<
                    HEAD_DIM, HEAD_DIM,
                    PosEncodingMode::kNone,
                    /*USE_FP16_QK_REDUCTION=*/false,
                    MaskMode::kNone,
                    Variant>(params, tmp_ptr, stream))
            );
        }
    }
}

template <typename DType>
void launch_flashinfer_paged_attention_h128_impl(
    void* q,
    void* paged_kv_data,
    int32_t* kv_indptr,
    int32_t* kv_indices,
    int32_t* kv_last_page_len,
    void* out,
    int batch_size,
    int num_qo_heads,
    int num_kv_heads,
    int page_size,
    float sm_scale,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream) {
    using namespace flashinfer;
    constexpr uint32_t HEAD_DIM = 128;

    DType* q_ptr = static_cast<DType*>(q);
    DType* o_ptr = static_cast<DType*>(out);
    DType* kv_ptr = static_cast<DType*>(paged_kv_data);
    (void)workspace;
    (void)workspace_size;

    // 假设 paged_kv_data 的单页布局为 [2, num_kv_heads, page_size, head_dim]，
    // 且 page 维度连续拼接: [num_pages, 2, num_kv_heads, page_size, head_dim]。
    const int64_t elems_per_k_page =
        static_cast<int64_t>(num_kv_heads) * static_cast<int64_t>(page_size) * HEAD_DIM;
    const int64_t kv_strides[4] = {
        2 * elems_per_k_page,                    // stride_page
        static_cast<int64_t>(page_size) * HEAD_DIM, // stride_h (HND)
        HEAD_DIM,                                // stride_n (HND)
        1};
    DType* k_data = kv_ptr;
    DType* v_data = kv_ptr + elems_per_k_page;

    paged_kv_t<DType, int32_t> paged_kv(
        static_cast<uint32_t>(num_kv_heads),
        static_cast<uint32_t>(page_size),
        HEAD_DIM,
        static_cast<uint32_t>(batch_size),
        QKVLayout::kHND,
        k_data,
        v_data,
        kv_strides,
        kv_indices,
        kv_indptr,
        kv_last_page_len,
        /*rope_pos_offset=*/nullptr);

    using Variant = DefaultAttention<
        /*use_custom_mask=*/false,
        /*use_sliding_window=*/false,
        /*use_logits_soft_cap=*/false,
        /*use_alibi=*/false>;
    BatchDecodeParams<DType, DType, DType, int32_t> params(
        q_ptr,
        /*q_rope_offset=*/nullptr,
        paged_kv,
        o_ptr,
        /*lse=*/nullptr,
        /*maybe_alibi_slopes=*/nullptr,
        static_cast<uint32_t>(num_qo_heads),
        static_cast<int32_t>(num_qo_heads * HEAD_DIM),
        static_cast<int32_t>(HEAD_DIM),
        /*window_left=*/-1,
        /*logits_soft_cap=*/0.0f,
        sm_scale,
        /*rope_scale=*/1.0f,
        /*rope_theta=*/1e4f);
    params.padded_batch_size = static_cast<uint32_t>(batch_size);

    FI_CUDA_CHECK((BatchDecodeWithPagedKVCacheDispatched<
                   HEAD_DIM,
                   PosEncodingMode::kNone,
                   Variant>(params,
                            /*tmp_v=*/nullptr,
                            /*tmp_s=*/nullptr,
                            /*enable_pdl=*/false,
                            stream)));
}

} // anonymous namespace

namespace llaisys::ops::nvidia {

void launch_flashinfer_decode_fp16_h128_pagedattn(
    void* q,
    void* paged_kv_data,
    int32_t* kv_indptr,
    int32_t* kv_indices,
    int32_t* kv_last_page_len,
    void* out,
    int batch_size,
    int num_qo_heads,
    int num_kv_heads,
    int page_size,
    float sm_scale,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream) {
    launch_flashinfer_paged_attention_h128_impl<__half>(
        q,
        paged_kv_data,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        out,
        batch_size,
        num_qo_heads,
        num_kv_heads,
        page_size,
        sm_scale,
        workspace,
        workspace_size,
        stream);
}

void launch_flashinfer_decode_fp32_h128_pagedattn(
    void* q,
    void* paged_kv_data,
    int32_t* kv_indptr,
    int32_t* kv_indices,
    int32_t* kv_last_page_len,
    void* out,
    int batch_size,
    int num_qo_heads,
    int num_kv_heads,
    int page_size,
    float sm_scale,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream) {
    (void)q;
    (void)paged_kv_data;
    (void)kv_indptr;
    (void)kv_indices;
    (void)kv_last_page_len;
    (void)out;
    (void)batch_size;
    (void)num_qo_heads;
    (void)num_kv_heads;
    (void)page_size;
    (void)sm_scale;
    (void)workspace;
    (void)workspace_size;
    (void)stream;
    fprintf(stderr,
            "[FlashInfer Wrapper] FP32 paged decode is unsupported by FlashInfer kernels.\n");
}

void launch_flashinfer_decode_bf16_h128_pagedattn(
    void* q,
    void* paged_kv_data,
    int32_t* kv_indptr,
    int32_t* kv_indices,
    int32_t* kv_last_page_len,
    void* out,
    int batch_size,
    int num_qo_heads,
    int num_kv_heads,
    int page_size,
    float sm_scale,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream) {
    launch_flashinfer_paged_attention_h128_impl<__nv_bfloat16>(
        q,
        paged_kv_data,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        out,
        batch_size,
        num_qo_heads,
        num_kv_heads,
        page_size,
        sm_scale,
        workspace,
        workspace_size,
        stream);
}

// ======================== FP16 ========================
void launch_flashinfer_decode_fp16_h128(
    void* q, void* k, void* v, void* out,
    int seq_q, int seq_kv,
    int num_qo_heads, int num_kv_heads,
    bool is_causal, float sm_scale,
    void* workspace, size_t workspace_size,
    cudaStream_t stream)
{
    launch_flashinfer_attention_h128_impl<__half>(
        q, k, v, out, seq_q, seq_kv,
        num_qo_heads, num_kv_heads,
        is_causal, sm_scale,
        workspace, workspace_size, stream);
}

// ======================== BF16 ========================
void launch_flashinfer_decode_bf16_h128(
    void* q, void* k, void* v, void* out,
    int seq_q, int seq_kv,
    int num_qo_heads, int num_kv_heads,
    bool is_causal, float sm_scale,
    void* workspace, size_t workspace_size,
    cudaStream_t stream)
{
    launch_flashinfer_attention_h128_impl<__nv_bfloat16>(
        q, k, v, out, seq_q, seq_kv,
        num_qo_heads, num_kv_heads,
        is_causal, sm_scale,
        workspace, workspace_size, stream);
}

} // namespace llaisys::ops::nvidia
