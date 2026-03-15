#pragma once

#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::nvidia {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale);
void self_attention_paged(tensor_t attn_val,
                          tensor_t q,
                          tensor_t paged_kv_data,
                          tensor_t kv_indptr,
                          tensor_t kv_indices,
                          tensor_t kv_last_page_len,
                          int page_size,
                          float scale);
}
