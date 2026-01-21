#include "op.hpp"

#include "cpu/argmax_cpu.hpp"

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());
    ASSERT(max_idx->ndim() == 1 && max_val->ndim() == 1 && vals->ndim() == 1,
           "Argmax: max_idx, max_val and vals must be 1D tensors");
    ASSERT(max_idx->shape()[0] == 1 && max_val->shape()[0] == 1,
           "Argmax: max_idx and max_val must be 1D tensors with a single element");
    // std::cout << "dtype of max_idx is" << max_idx->dtype() << std::endl;
    ASSERT(max_idx->dtype() == LLAISYS_DTYPE_I64,
           "Argmax:Data Type of max_idx must be LLAISYS_DTYPE_I64");

    if (vals->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::argmax(max_idx->data(),
                           max_val->data(),
                           vals->data(),
                           vals->numel(),
                           static_cast<size_t>(vals->strides()[0]),
                           vals->dtype());
    }
}
} // namespace llaisys::ops
