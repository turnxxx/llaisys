
#include "op.hpp"
#include "./cpu/linear_cpu.hpp"
namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight, bias);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype(), bias->dtype());
    ASSERT(weight->isContiguous()
               && in->isContiguous()
               && bias->isContiguous(),
           "Linear:in, weight and bias must be contiguous");

    ASSERT(weight->shape().size() == 2
               && in->shape().size() == 2
               && out->shape().size() == 2,
           "Linear: Invalid shape size");
    ASSERT(in->shape()[0] == out->shape()[0] && in->shape()[1] == weight->shape()[1] && weight->shape()[0] == out->shape()[1],
           "Invalid shape number");
    return llaisys::ops::cpu::linear(out, in, weight, bias);
}
} // namespace llaisys::ops
