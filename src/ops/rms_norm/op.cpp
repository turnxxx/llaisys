#include "op.hpp"
#include "./cpu/rms_norm_cpu.hpp"
namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), "Rms_norm: Tensor must be contiguous");
    ASSERT(out->shape().size() == 2 && weight->shape().size() == 1 && in->shape().size() == 2, "Rms_norm: Invalid tensor shape");
    ASSERT(eps > 0.0f, "Rms_norm: eps must larger than 0");
    ASSERT(out->shape() == in->shape() && weight->shape()[0] == in->shape()[1], "Rms_norm:Mismatch shape size");
    if (in->deviceType() == LLAISYS_DEVICE_CPU) {
        return llaisys::ops::cpu::rms_norm(out, in, weight, eps);
    }
}
} // namespace llaisys::ops
