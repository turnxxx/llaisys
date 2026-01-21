#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"
namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    ASSERT(weight->shape().size() == 2, "Embedding: shape of weight must be 2-D");
    ASSERT(index->shape().size() == 1, "Embedding: shape of weight must be 1-D");
    ASSERT(out->shape().size() == 2, "Embedding: shape of out must be 2-D");
    CHECK_SAME_DEVICE(out, index, weight);
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    CHECK_SAME_DTYPE(index->dtype(), LLAISYS_DTYPE_I64);
    // 检查out的数据区大小是否合理
    {
        size_t embedding_numel = weight->shape()[1] * index->numel();
        ASSERT(embedding_numel <= out->numel(), "Embedding: numel of out must larger than numel of embedding");
    }
    // 调用cpu实现
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return llaisys::ops::cpu::embedding(out, index, weight);
    }
}
} // namespace llaisys::ops
