#include "embedding_cpu.hpp"
#include "../../../tensor/tensor.hpp"
#include "../../../utils.hpp"

#include <cmath>
#include <cstring>
/**
 * @brief CPU 版本的 连续embedding：按 index 选择 weight 的行并写入 out。
 *
 * @param out_data    输出缓冲区，连续存放 [index_numel, col_numel]。
 * @param index_data  索引数组（int64），长度为 index_numel。
 * @param weight_data 权重矩阵缓冲区，按行存放 [vocab, col_numel]。
 * @param index_numel index 的元素个数（输出行数）。
 * @param col_numel   每行元素个数（embedding 维度）。
 * @note  这个函数要求 weight的列 stride=1
 * 假设已经确定好了out_data的大小
 */
template <typename T>
void contiguous_embedding_(T *out_data, int64_t *index_data, T *weight_data,
                           size_t index_numel, size_t col_numel, size_t row_stride) {

    for (size_t i = 0; i < index_numel; i++) {
        int64_t row_idx = index_data[i]; // weight的行号
        ASSERT(row_idx >= 0, "Embedding: index must be non-negative");
        // weight_data第row_idx行起始地址为
        size_t weight_start_idx = static_cast<size_t>(row_idx) * row_stride;
        std::memcpy(out_data + i * col_numel,
                    weight_data + weight_start_idx,
                    sizeof(T) * col_numel);
    }
}
// 列不连续时的embedding
template <typename T>
void uncontiguous_embedding_(T *out_data, int64_t *index_data, T *weight_data,
                             size_t index_numel, size_t col_numel,
                             size_t row_stride, size_t col_stride) {

    // 遍历行，再遍历列
    for (size_t i = 0; i < index_numel; i++) {
        int64_t row_idx = index_data[i];
        ASSERT(row_idx >= 0, "Embedding: index must be non-negative");
        size_t weight_start_idx = static_cast<size_t>(row_idx) * row_stride;
        for (size_t j = 0; j < col_numel; j++) {
            // 遍历列
            size_t offset = col_stride * j;
            out_data[i * col_numel + j] = weight_data[weight_start_idx + offset];
        }
    }
}
// 根据shape的stride类型来分别使用两种实现
namespace llaisys::ops::cpu {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    // 列连续
    if (weight->strides()[1] == 1) {
        size_t index_numel = static_cast<size_t>(index->numel());
        size_t col_numel = static_cast<size_t>(weight->shape()[1]);
        size_t row_stride = static_cast<size_t>(weight->strides()[0]);
        switch (weight->dtype()) {
        case LLAISYS_DTYPE_F32:
            return contiguous_embedding_(reinterpret_cast<float *>(out->data()),
                                         reinterpret_cast<int64_t *>(index->data()),
                                         reinterpret_cast<float *>(weight->data()),
                                         index_numel, col_numel, row_stride);
            break;
        case LLAISYS_DTYPE_BF16:
            return contiguous_embedding_(reinterpret_cast<llaisys::bf16_t *>(out->data()),
                                         reinterpret_cast<int64_t *>(index->data()),
                                         reinterpret_cast<llaisys::bf16_t *>(weight->data()),
                                         index_numel, col_numel, row_stride);
            break;
        case LLAISYS_DTYPE_F16:
            return contiguous_embedding_(reinterpret_cast<llaisys::fp16_t *>(out->data()),
                                         reinterpret_cast<int64_t *>(index->data()),
                                         reinterpret_cast<llaisys::fp16_t *>(weight->data()),
                                         index_numel, col_numel, row_stride);
            break;
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(weight->dtype());
        }
    } else {
        // 列不连续的情况
        size_t index_numel = static_cast<size_t>(index->numel());
        size_t col_numel = static_cast<size_t>(weight->shape()[1]);
        size_t row_stride = static_cast<size_t>(weight->strides()[0]);
        size_t col_stride = static_cast<size_t>(weight->strides()[1]);
        switch (weight->dtype()) {
        case LLAISYS_DTYPE_F32:
            return uncontiguous_embedding_(reinterpret_cast<float *>(out->data()),
                                           reinterpret_cast<int64_t *>(index->data()),
                                           reinterpret_cast<float *>(weight->data()),
                                           index_numel, col_numel, row_stride, col_stride);
            break;
        case LLAISYS_DTYPE_BF16:
            return uncontiguous_embedding_(reinterpret_cast<llaisys::bf16_t *>(out->data()),
                                           reinterpret_cast<int64_t *>(index->data()),
                                           reinterpret_cast<llaisys::bf16_t *>(weight->data()),
                                           index_numel, col_numel, row_stride, col_stride);
            break;
        case LLAISYS_DTYPE_F16:
            return uncontiguous_embedding_(reinterpret_cast<llaisys::fp16_t *>(out->data()),
                                           reinterpret_cast<int64_t *>(index->data()),
                                           reinterpret_cast<llaisys::fp16_t *>(weight->data()),
                                           index_numel, col_numel, row_stride, col_stride);
            break;
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(weight->dtype());
        }
    }
}
} // namespace llaisys::ops::cpu