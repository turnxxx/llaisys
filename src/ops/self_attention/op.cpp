#include "op.hpp"
#include "./cpu/self_attention_cpu.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "SelfAttention: inputs must be contiguous");
    ASSERT(attn_val->shape().size() == 3 && q->shape().size() == 3
               && k->shape().size() == 3 && v->shape().size() == 3,
           "SelfAttention: invalid shape size");
    return llaisys::ops::cpu::self_attention(attn_val, q, k, v, scale);
}
} // namespace llaisys::ops
