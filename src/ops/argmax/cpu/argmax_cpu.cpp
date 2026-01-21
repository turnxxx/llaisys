#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
// 模板化argmax
template <typename T>
void argmax_(size_t *max_idx, T *max_val, const T *vals, size_t numel, size_t stride) {
    ASSERT(numel > 0 && stride > 0, "Argmax: numel and stride must be greater than 0");
    *max_val = vals[0];
    *max_idx = 0;
    for (size_t i = stride; i < numel; i += stride) {
        if constexpr (std::is_same_v<T, llaisys::bf16_t>
                      || std::is_same_v<T, llaisys::fp16_t>) {
            if (llaisys::utils::cast<float>(vals[i]) > llaisys::utils::cast<float>(*max_val)) {
                *max_val = vals[i];
                *max_idx = i;
            }
        } else {
            if (vals[i] > *max_val) {
                *max_val = vals[i];
                *max_idx = i;
            }
        }
    }
}
namespace llaisys::ops::cpu {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals,
            size_t numel, size_t stride, llaisysDataType_t type) {
    switch (type) {

    case LLAISYS_DTYPE_F32:
        return argmax_(reinterpret_cast<size_t *>(max_idx),
                       reinterpret_cast<float *>(max_val),
                       reinterpret_cast<const float *>(vals),
                       numel, stride);
    case LLAISYS_DTYPE_BF16:
        return argmax_(reinterpret_cast<size_t *>(max_idx),
                       reinterpret_cast<llaisys::bf16_t *>(max_val),
                       reinterpret_cast<const llaisys::bf16_t *>(vals),
                       numel, stride);
    case LLAISYS_DTYPE_F16:
        return argmax_(reinterpret_cast<size_t *>(max_idx),
                       reinterpret_cast<llaisys::fp16_t *>(max_val),
                       reinterpret_cast<const llaisys::fp16_t *>(vals),
                       numel, stride);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cpu