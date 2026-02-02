#include "llaisys/weights_buffer.h"

#include "../model/model_utils.hpp"
#include "llaisys_tensor.hpp"

__C {
    struct LlaisysWeightBuffer {
        llaisys::model::Weight_buffer buffer;
    };

    llaisysWeightBuffer_t weightBufferCreate() {
        return new LlaisysWeightBuffer{};
    }

    void weightBufferDestroy(llaisysWeightBuffer_t buffer) {
        delete buffer;
    }

    void weightBufferClear(llaisysWeightBuffer_t buffer) {
        if (!buffer) {
            return;
        }
        buffer->buffer.clear();
    }

    size_t weightBufferSize(llaisysWeightBuffer_t buffer) {
        return buffer ? buffer->buffer.size() : 0;
    }

    uint8_t weightBufferHas(llaisysWeightBuffer_t buffer, const char *name) {
        if (!buffer || !name) {
            return 0;
        }
        return static_cast<uint8_t>(buffer->buffer.has(name));
    }

    void weightBufferAdd(
        llaisysWeightBuffer_t buffer,
        const char *name,
        llaisysTensor_t weight) {
        if (!buffer || !name || !weight) {
            return;
        }
        const auto &weight_tensor = weight->tensor;
        buffer->buffer.add_tensor(name, weight_tensor);
    }
}
