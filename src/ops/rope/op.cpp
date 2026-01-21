#include "op.hpp"
#include "./cpu/rope_cpu.hpp"
namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "Rope: data type of pos_ids must be int64");
    ASSERT(pos_ids->shape()[0] == in->shape()[0], "Rope:Shape mismatch");
    if (in->deviceType() == LLAISYS_DEVICE_CPU) {
        return llaisys::ops::cpu::rope(out, in, pos_ids, theta);
    }
}
} // namespace llaisys::ops
