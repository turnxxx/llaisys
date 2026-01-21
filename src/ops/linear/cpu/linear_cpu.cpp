#include "linear_cpu.hpp"
#include "../../../tensor/tensor.hpp"
#include "../../../utils.hpp"

#include <cmath>
#include <cstring>
// 无偏置情形, 2D情形
template <typename T>
void non_bias_linear_(T *out_data, const T *in_data, const T *weight_data,
                      const std::vector<size_t> &shape) {
    size_t m = shape[0];
    size_t k = shape[1];
    size_t n = shape[2];
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            size_t out_idx = i * n + j;
            if constexpr (std::is_same_v<T, llaisys::bf16_t>
                          || std::is_same_v<T, llaisys::fp16_t>) {
                float acc = 0.0f;
                for (size_t l = 0; l < k; l++) {
                    size_t in_idx = i * k + l;
                    size_t weight_idx = j * k + l;
                    acc += llaisys::utils::cast<float>(in_data[in_idx])
                         * llaisys::utils::cast<float>(weight_data[weight_idx]);
                }
                out_data[out_idx] = llaisys::utils::cast<T>(acc);
            } else {
                T acc = T(0);
                for (size_t l = 0; l < k; l++) {
                    size_t in_idx = i * k + l;
                    size_t weight_idx = j * k + l;
                    acc += in_data[in_idx] * weight_data[weight_idx];
                }
                out_data[out_idx] = acc;
            }
        }
    }
}
// 有偏置情形
template <typename T>
void bias_linear_(T *out_data, const T *in_data, const T *weight_data, const T *bias_data,
                  const std::vector<size_t> &shape) {
    size_t m = shape[0];
    size_t k = shape[1];
    size_t n = shape[2];
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            if constexpr (std::is_same_v<T, llaisys::bf16_t>
                          || std::is_same_v<T, llaisys::fp16_t>) {
                float acc = 0.0f;
                for (size_t l = 0; l < k; l++) {
                    size_t in_idx = i * k + l;
                    size_t weight_idx = j * k + l;
                    acc += llaisys::utils::cast<float>(in_data[in_idx])
                         * llaisys::utils::cast<float>(weight_data[weight_idx]);
                }
                acc += llaisys::utils::cast<float>(bias_data[j]);
                out_data[i * n + j] = llaisys::utils::cast<T>(acc);
            } else {
                T acc = T(0);
                for (size_t l = 0; l < k; l++) {
                    size_t in_idx = i * k + l;
                    size_t weight_idx = j * k + l;
                    acc += in_data[in_idx] * weight_data[weight_idx];
                }
                acc += bias_data[j];
                out_data[i * n + j] = acc;
            }
        }
    }
}
// 对外接口
namespace llaisys::ops::cpu {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    // 计算新的shape
    std::vector<size_t> shape = {in->shape()[0], in->shape()[1], weight->shape()[0]};
    // 选择是否提供偏置
    bool has_bias = bias != nullptr && bias->numel() == out->shape()[1];
    if (!has_bias) {
        // 无偏置
        switch (out->dtype()) {
        case LLAISYS_DTYPE_F32:
            /* code */
            return non_bias_linear_(reinterpret_cast<float *>(out->data()),
                                    reinterpret_cast<float *>(in->data()),
                                    reinterpret_cast<float *>(weight->data()),
                                    shape);
            break;
        case LLAISYS_DTYPE_BF16:
            return non_bias_linear_(reinterpret_cast<llaisys::bf16_t *>(out->data()),
                                    reinterpret_cast<llaisys::bf16_t *>(in->data()),
                                    reinterpret_cast<llaisys::bf16_t *>(weight->data()),
                                    shape);
            break;
        case LLAISYS_DTYPE_F16:
            return non_bias_linear_(reinterpret_cast<llaisys::fp16_t *>(out->data()),
                                    reinterpret_cast<llaisys::fp16_t *>(in->data()),
                                    reinterpret_cast<llaisys::fp16_t *>(weight->data()),
                                    shape);
            break;
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(weight->dtype());
        }
    } else {
        switch (out->dtype()) {
        case LLAISYS_DTYPE_F32:
            /* code */
            return bias_linear_(reinterpret_cast<float *>(out->data()),
                                reinterpret_cast<float *>(in->data()),
                                reinterpret_cast<float *>(weight->data()),
                                reinterpret_cast<float *>(bias->data()),
                                shape);
            break;
        case LLAISYS_DTYPE_BF16:
            return bias_linear_(reinterpret_cast<llaisys::bf16_t *>(out->data()),
                                reinterpret_cast<llaisys::bf16_t *>(in->data()),
                                reinterpret_cast<llaisys::bf16_t *>(weight->data()),
                                reinterpret_cast<llaisys::bf16_t *>(bias->data()),
                                shape);
            break;
        case LLAISYS_DTYPE_F16:
            return bias_linear_(reinterpret_cast<llaisys::fp16_t *>(out->data()),
                                reinterpret_cast<llaisys::fp16_t *>(in->data()),
                                reinterpret_cast<llaisys::fp16_t *>(weight->data()),
                                reinterpret_cast<llaisys::fp16_t *>(bias->data()),
                                shape);
            break;
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(weight->dtype());
        }
    }
}
} // namespace llaisys::ops::cpu