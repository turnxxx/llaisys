#include "op.hpp"
#include "./cpu/matmul_cpu.hpp"

namespace llaisys::ops {
void transpose_matmul(tensor_t c, tensor_t a, tensor_t b, float scale) {
    CHECK_SAME_DEVICE(c, a, b);
    CHECK_SAME_DTYPE(c->dtype(), a->dtype(), b->dtype());
    ASSERT(a->isContiguous() && b->isContiguous() && c->isContiguous(),
           "Matmul: input and output must be contiguous");
    ASSERT(a->shape().size() == 2 && b->shape().size() == 2 && c->shape().size() == 2,
           "Matmul: Invalid shape size");
    ASSERT(a->shape()[0] == c->shape()[0]
               && a->shape()[1] == b->shape()[1]
               && b->shape()[0] == c->shape()[1],
           "Matmul: Invalid shape number");
    return llaisys::ops::cpu::transpose_matmul(c, a, b, scale);
}
} // namespace llaisys::ops
