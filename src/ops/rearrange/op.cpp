#include "op.hpp"
#include "./cpu/rearrange_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "./nvidia/rearrange_nvidia.cuh"
#endif

namespace llaisys::ops {
void rearrange(tensor_t out, tensor_t in) {
    CHECK_SAME_DEVICE(out, in);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    if (in->deviceType() == LLAISYS_DEVICE_CPU) {
        return llaisys::ops::cpu::rearrange(out, in);
    }
#ifdef ENABLE_NVIDIA_API
    if (in->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        return nvidia::rearrange(out, in);
    }
#endif
    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
