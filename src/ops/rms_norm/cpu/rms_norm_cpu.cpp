#include "rms_norm_cpu.hpp"
#include "../../../tensor/tensor.hpp"
#include "../../../utils.hpp"

#include <cmath>
#include <cstring>

template <typename T>
void rms_norm_(T *out_data, T *in_data, T *weight_data,
               float eps, const std::vector<size_t> &shape) {
    size_t m = shape[0]; // 行维度
    size_t n = shape[1]; // 列维度
    for (size_t i = 0; i < m; i++) {
        // 先计算分母
        float norm = 0.0f;
        for (size_t j = 0; j < n; j++) {
            if constexpr (std::is_same_v<T, llaisys::bf16_t>
                          || std::is_same_v<T, llaisys::fp16_t>) {
                norm += llaisys::utils::cast<float>(in_data[i * n + j])
                      * llaisys::utils::cast<float>(in_data[i * n + j]);
            } else {
                norm += in_data[i * n + j] * in_data[i * n + j];
            }
        }
        norm = std::sqrt(norm / n + eps);
        // 计算分子
        for (size_t j = 0; j < n; j++) {
            if constexpr (std::is_same_v<T, llaisys::bf16_t>
                          || std::is_same_v<T, llaisys::fp16_t>) {
                out_data[i * n + j] = llaisys::utils::cast<T>(
                    llaisys::utils::cast<float>(weight_data[j])
                    * llaisys::utils::cast<float>(in_data[i * n + j]) / norm);
            } else {
                out_data[i * n + j] = weight_data[j] * in_data[i * n + j] / norm;
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    switch (in->dtype()) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out->data()),
                         reinterpret_cast<float *>(in->data()),
                         reinterpret_cast<float *>(weight->data()),
                         eps, in->shape());
        break;
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out->data()),
                         reinterpret_cast<llaisys::bf16_t *>(in->data()),
                         reinterpret_cast<llaisys::bf16_t *>(weight->data()),
                         eps, in->shape());
        break;
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out->data()),
                         reinterpret_cast<llaisys::fp16_t *>(in->data()),
                         reinterpret_cast<llaisys::fp16_t *>(weight->data()),
                         eps, in->shape());
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(in->dtype());
    }
}

} // namespace llaisys::ops::cpu