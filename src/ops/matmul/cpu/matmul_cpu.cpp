#include "../../../tensor/tensor.hpp"
#include "../../../utils.hpp"

#include "matmul_cpu.hpp"
#include <cmath>
#include <cstring>
#include <type_traits>
// 任意步长的矩阵乘法实现 C=A*B^T
template <typename T>
void transpose_matmul_(T *C, T *A, T *B,
                       const std::vector<size_t> &C_shape,
                       const std::vector<size_t> &A_shape,
                       const std::vector<size_t> &B_shape,
                       const std::vector<size_t> &C_strides,
                       const std::vector<size_t> &A_strides,
                       const std::vector<size_t> &B_strides,
                       float scale) {
    size_t m = A_shape[0];
    size_t k = A_shape[1];
    size_t n = B_shape[0];
    size_t c_s0 = C_strides[0];
    size_t c_s1 = C_strides[1];
    size_t a_s0 = A_strides[0];
    size_t a_s1 = A_strides[1];
    size_t b_s0 = B_strides[0];
    size_t b_s1 = B_strides[1];

    /*  if (C_shape[0] != m || C_shape[1] != n || B_shape[1] != k) {
         return;
     } */

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            size_t c_idx = i * c_s0 + j * c_s1;
            if constexpr (std::is_same_v<T, llaisys::bf16_t>
                          || std::is_same_v<T, llaisys::fp16_t>) {
                float acc = 0.0f;
                for (size_t l = 0; l < k; ++l) {
                    size_t a_idx = i * a_s0 + l * a_s1;
                    size_t b_idx = j * b_s0 + l * b_s1;
                    acc += llaisys::utils::cast<float>(A[a_idx])
                         * llaisys::utils::cast<float>(B[b_idx]);
                }
                C[c_idx] = llaisys::utils::cast<T>(acc * scale);
            } else {
                T acc = T(0);
                for (size_t l = 0; l < k; ++l) {
                    size_t a_idx = i * a_s0 + l * a_s1;
                    size_t b_idx = j * b_s0 + l * b_s1;
                    acc += A[a_idx] * B[b_idx];
                }
                C[c_idx] = acc * scale;
            }
        }
    }
}

namespace llaisys::ops::cpu {
void transpose_matmul(tensor_t c, tensor_t a, tensor_t b, float scale) {
    const auto &c_shape = c->shape();
    const auto &a_shape = a->shape();
    const auto &b_shape = b->shape();
    std::vector<size_t> c_strides(c->strides().begin(), c->strides().end());
    std::vector<size_t> a_strides(a->strides().begin(), a->strides().end());
    std::vector<size_t> b_strides(b->strides().begin(), b->strides().end());

    switch (c->dtype()) {
    case LLAISYS_DTYPE_F32:
        return transpose_matmul_(reinterpret_cast<float *>(c->data()),
                                 reinterpret_cast<float *>(a->data()),
                                 reinterpret_cast<float *>(b->data()),
                                 c_shape, a_shape, b_shape,
                                 c_strides, a_strides, b_strides,
                                 scale);
    case LLAISYS_DTYPE_BF16:
        return transpose_matmul_(reinterpret_cast<llaisys::bf16_t *>(c->data()),
                                 reinterpret_cast<llaisys::bf16_t *>(a->data()),
                                 reinterpret_cast<llaisys::bf16_t *>(b->data()),
                                 c_shape, a_shape, b_shape,
                                 c_strides, a_strides, b_strides,
                                 scale);
    case LLAISYS_DTYPE_F16:
        return transpose_matmul_(reinterpret_cast<llaisys::fp16_t *>(c->data()),
                                 reinterpret_cast<llaisys::fp16_t *>(a->data()),
                                 reinterpret_cast<llaisys::fp16_t *>(b->data()),
                                 c_shape, a_shape, b_shape,
                                 c_strides, a_strides, b_strides,
                                 scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(c->dtype());
    }
}
} // namespace llaisys::ops::cpu
