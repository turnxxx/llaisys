#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

namespace llaisys::ops::nvidia {

// ======================== Paged Attention 版本 ========================

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
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream);

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
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream);

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
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream);

// ======================== 非 Paged Attention 版本 ========================
// Q: [seq_q, num_qo_heads, head_dim]   NHD layout
// K: [seq_kv, num_kv_heads, head_dim]   NHD layout
// V: [seq_kv, num_kv_heads, head_dim]   NHD layout
// O: [seq_q, num_qo_heads, head_dim]
//
// seq_q == 1 时走 FlashInfer SingleDecode 路径 (针对逐 token 推理优化)
// seq_q >  1 时走 FlashInfer SinglePrefill 路径 (针对 prefill 阶段优化)
//
// 注意: FlashInfer 内核依赖 Tensor Core MMA 和 cp_async 硬件指令，
//       仅支持 2 字节数据类型 (FP16 / BF16)，不支持 FP32。

void launch_flashinfer_decode_fp16_h128(
    void* q,
    void* k,
    void* v,
    void* out,
    int seq_q,
    int seq_kv,
    int num_qo_heads,
    int num_kv_heads,
    bool is_causal,
    float sm_scale,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream);

void launch_flashinfer_decode_bf16_h128(
    void* q,
    void* k,
    void* v,
    void* out,
    int seq_q,
    int seq_kv,
    int num_qo_heads,
    int num_kv_heads,
    bool is_causal,
    float sm_scale,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream);

}
