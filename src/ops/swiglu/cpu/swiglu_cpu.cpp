#include "swiglu_cpu.hpp"
#include "../../../tensor/tensor.hpp"
#include "../../../utils.hpp"
#include <cmath>
#include <cstring>
#include <vector>
template <typename T>
void swiglu_(T *out_data, T *gate_data, T *up_data,
             const std::vector<size_t> &shape) {
    size_t seqlen = shape[0];
    size_t intermediate_size = shape[1];
    for (size_t i = 0; i < seqlen; i++) {
        for (size_t j = 0; j < intermediate_size; j++) {
            size_t idx = i * intermediate_size + j;
            if constexpr (std::is_same_v<T, llaisys::bf16_t>
                          || std::is_same_v<T, llaisys::fp16_t>) {
                float acc = 0.0f;
                acc = llaisys::utils::cast<float>(up_data[idx])
                    * llaisys::utils::cast<float>(gate_data[idx])
                    / (1 + std::exp(-llaisys::utils::cast<float>(gate_data[idx])));
                out_data[idx] = llaisys::utils::cast<T>(acc);
            } else {
                out_data[idx] = up_data[idx] * gate_data[idx] / (1 + std::exp(-gate_data[idx]));
            }
        }
    }
}

namespace llaisys::ops::cpu {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    const auto &shape = out->shape();
    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
        return swiglu_(reinterpret_cast<float *>(out->data()),
                       reinterpret_cast<float *>(gate->data()),
                       reinterpret_cast<float *>(up->data()),
                       shape);
    case LLAISYS_DTYPE_BF16:
        return swiglu_(reinterpret_cast<llaisys::bf16_t *>(out->data()),
                       reinterpret_cast<llaisys::bf16_t *>(gate->data()),
                       reinterpret_cast<llaisys::bf16_t *>(up->data()),
                       shape);
    case LLAISYS_DTYPE_F16:
        return swiglu_(reinterpret_cast<llaisys::fp16_t *>(out->data()),
                       reinterpret_cast<llaisys::fp16_t *>(gate->data()),
                       reinterpret_cast<llaisys::fp16_t *>(up->data()),
                       shape);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
}
} // namespace llaisys::ops::cpu