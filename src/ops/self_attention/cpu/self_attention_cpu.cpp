#include "self_attention_cpu.hpp"
#include "../../../tensor/tensor.hpp"
#include "../../../utils.hpp"
#include <cmath>
#include <cstring>
#include <vector>
template <typename T>
void self_attention_(T *attn_val_data, T *q_data, T *k_data, T *v_data,
                     float scale,
                     const std::vector<size_t> &q_shape,
                     const std::vector<size_t> &k_shape,
                     const std::vector<size_t> &v_shape,
                     const std::vector<ptrdiff_t> &q_strides,
                     const std::vector<ptrdiff_t> &k_strides,
                     const std::vector<ptrdiff_t> &v_strides,
                     const std::vector<ptrdiff_t> &attn_val_strides) {
    size_t seqlen = q_shape[0];
    size_t nhead = q_shape[1];
    size_t d = q_shape[2];
    size_t total_len = k_shape[0];
    size_t nkvhead = k_shape[1];
    size_t dv = v_shape[2];
    size_t group = nhead / nkvhead;
    size_t shift = total_len >= seqlen ? (total_len - seqlen) : 0;

    std::vector<float> scores(total_len);
    std::vector<float> probs(total_len);

    for (size_t h = 0; h < nhead; h++) {
        size_t kv_h = h / group;
        for (size_t i = 0; i < seqlen; i++) {
            // compute scores
            float max_score = -1e30f;
            for (size_t t = 0; t < total_len; t++) {
                if (t > i + shift) {
                    scores[t] = -1e30f;
                    continue;
                }
                float acc = 0.0f;
                for (size_t l = 0; l < d; l++) {
                    size_t q_idx = i * static_cast<size_t>(q_strides[0])
                                 + h * static_cast<size_t>(q_strides[1])
                                 + l * static_cast<size_t>(q_strides[2]);
                    size_t k_idx = t * static_cast<size_t>(k_strides[0])
                                 + kv_h * static_cast<size_t>(k_strides[1])
                                 + l * static_cast<size_t>(k_strides[2]);
                    if constexpr (std::is_same_v<T, llaisys::bf16_t>
                                  || std::is_same_v<T, llaisys::fp16_t>) {
                        acc += llaisys::utils::cast<float>(q_data[q_idx])
                             * llaisys::utils::cast<float>(k_data[k_idx]);
                    } else {
                        acc += static_cast<float>(q_data[q_idx])
                             * static_cast<float>(k_data[k_idx]);
                    }
                }
                acc *= scale;
                scores[t] = acc;
                if (acc > max_score) {
                    max_score = acc;
                }
            }

            // softmax
            float sum = 0.0f;
            for (size_t t = 0; t < total_len; t++) {
                float v = std::exp(scores[t] - max_score);
                probs[t] = v;
                sum += v;
            }
            float inv_sum = sum > 0.0f ? (1.0f / sum) : 0.0f;
            for (size_t t = 0; t < total_len; t++) {
                probs[t] *= inv_sum;
            }

            // weighted sum with V
            for (size_t dv_idx = 0; dv_idx < dv; dv_idx++) {
                float acc = 0.0f;
                for (size_t t = 0; t < total_len; t++) {
                    size_t v_idx = t * static_cast<size_t>(v_strides[0])
                                 + kv_h * static_cast<size_t>(v_strides[1])
                                 + dv_idx * static_cast<size_t>(v_strides[2]);
                    if constexpr (std::is_same_v<T, llaisys::bf16_t>
                                  || std::is_same_v<T, llaisys::fp16_t>) {
                        acc += probs[t] * llaisys::utils::cast<float>(v_data[v_idx]);
                    } else {
                        acc += probs[t] * static_cast<float>(v_data[v_idx]);
                    }
                }
                size_t out_idx = i * static_cast<size_t>(attn_val_strides[0])
                               + h * static_cast<size_t>(attn_val_strides[1])
                               + dv_idx * static_cast<size_t>(attn_val_strides[2]);
                if constexpr (std::is_same_v<T, llaisys::bf16_t>
                              || std::is_same_v<T, llaisys::fp16_t>) {
                    attn_val_data[out_idx] = llaisys::utils::cast<T>(acc);
                } else {
                    attn_val_data[out_idx] = static_cast<T>(acc);
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    switch (q->dtype()) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(attn_val->data()),
                               reinterpret_cast<float *>(q->data()),
                               reinterpret_cast<float *>(k->data()),
                               reinterpret_cast<float *>(v->data()),
                               scale,
                               q->shape(), k->shape(), v->shape(),
                               q->strides(), k->strides(), v->strides(),
                               attn_val->strides());
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t *>(attn_val->data()),
                               reinterpret_cast<llaisys::bf16_t *>(q->data()),
                               reinterpret_cast<llaisys::bf16_t *>(k->data()),
                               reinterpret_cast<llaisys::bf16_t *>(v->data()),
                               scale,
                               q->shape(), k->shape(), v->shape(),
                               q->strides(), k->strides(), v->strides(),
                               attn_val->strides());
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<llaisys::fp16_t *>(attn_val->data()),
                               reinterpret_cast<llaisys::fp16_t *>(q->data()),
                               reinterpret_cast<llaisys::fp16_t *>(k->data()),
                               reinterpret_cast<llaisys::fp16_t *>(v->data()),
                               scale,
                               q->shape(), k->shape(), v->shape(),
                               q->strides(), k->strides(), v->strides(),
                               attn_val->strides());
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(q->dtype());
    }
}
} // namespace llaisys::ops::cpu