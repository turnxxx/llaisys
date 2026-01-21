#include "op.hpp"
#include "./cpu/swiglu_cpu.hpp"

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out, gate, up);
    CHECK_SAME_DTYPE(out->dtype(), gate->dtype(), up->dtype());
    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(),
           "SwiGLU: inputs must be contiguous");
    ASSERT(out->shape().size() == 2 && gate->shape().size() == 2 && up->shape().size() == 2,
           "SwiGLU: invalid shape size");
    ASSERT(out->shape() == gate->shape() && out->shape() == up->shape(),
           "SwiGLU: shape mismatch");
    return llaisys::ops::cpu::swiglu(out, gate, up);
}
} // namespace llaisys::ops
