
#include "rope_cpu.hpp"

#include "../../../tensor/tensor.hpp"
#include "../../../utils.hpp"

#include <cmath>
#include <cstring>

template <typename T>

void rope_(T *out_data, T *in_data, int64_t *pos_ids,
           float theta, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides) {
    size_t seqlen = shape[0];
    size_t nhead = shape[1];
    size_t d = shape[2];
    for (size_t i = 0; i < seqlen; i++) {
        for (size_t j = 0; j < nhead; j++) {
            size_t start_offset = i * static_cast<size_t>(strides[0]) + j * static_cast<size_t>(strides[1]);
            for (size_t k = 0; k < d / 2; k++) {
                size_t a_idx = start_offset + k;
                size_t b_idx = start_offset + k + d / 2;
                if constexpr (std::is_same_v<T, llaisys::bf16_t>
                              || std::is_same_v<T, llaisys::fp16_t>) {
                    float angle = pos_ids[i]
                                / (std::pow(theta, 2.0f * k / d));
                    float c = std::cos(angle);
                    float s = std::sin(angle);
                    out_data[a_idx] = llaisys::utils::cast<T>(llaisys::utils::cast<float>(in_data[a_idx]) * c
                                                              - llaisys::utils::cast<float>(in_data[b_idx]) * s);
                    out_data[b_idx] = llaisys::utils::cast<T>(llaisys::utils::cast<float>(in_data[b_idx]) * c
                                                              + llaisys::utils::cast<float>(in_data[a_idx]) * s);
                } else {
                    float angle = pos_ids[i]
                                / (std::pow(theta, 2.0f * k / d));
                    float c = std::cos(angle);
                    float s = std::sin(angle);
                    out_data[a_idx] = in_data[a_idx] * c
                                    - in_data[b_idx] * s;
                    out_data[b_idx] = in_data[b_idx] * c
                                    + in_data[a_idx] * s;
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    switch (in->dtype()) {
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t *>(out->data()),
                     reinterpret_cast<llaisys::bf16_t *>(in->data()),
                     reinterpret_cast<int64_t *>(pos_ids->data()),
                     theta,
                     in->shape(),
                     in->strides());
        break;
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t *>(out->data()),
                     reinterpret_cast<llaisys::fp16_t *>(in->data()),
                     reinterpret_cast<int64_t *>(pos_ids->data()),
                     theta,
                     in->shape(),
                     in->strides());
        break;
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out->data()),
                     reinterpret_cast<float *>(in->data()),
                     reinterpret_cast<int64_t *>(pos_ids->data()),
                     theta,
                     in->shape(),
                     in->strides());
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(in->dtype());
    }
}
} // namespace llaisys::ops::cpu