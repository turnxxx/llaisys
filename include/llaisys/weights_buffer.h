#ifndef LLAISYS_WEIGHTS_BUFFER_H
#define LLAISYS_WEIGHTS_BUFFER_H

#include "../llaisys.h"
#include "tensor.h"

__C {
    // 前向定义 llaisysWeightBuffer_t的指针
    typedef struct LlaisysWeightBuffer *llaisysWeightBuffer_t;

    __export llaisysWeightBuffer_t weightBufferCreate();
    __export void weightBufferDestroy(llaisysWeightBuffer_t buffer);
    __export void weightBufferClear(llaisysWeightBuffer_t buffer);
    __export size_t weightBufferSize(llaisysWeightBuffer_t buffer);
    __export uint8_t weightBufferHas(llaisysWeightBuffer_t buffer, const char *name);
    __export void weightBufferAdd(
        llaisysWeightBuffer_t buffer,
        const char *name,
        llaisysTensor_t weight);
}

#endif // LLAISYS_WEIGHTS_BUFFER_H
