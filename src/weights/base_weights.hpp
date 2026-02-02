/*
权重基类定义：用于在推理框架中统一表示权重张量与其元信息。
*/
#pragma once

#include "llaisys.h"

#include "../tensor/tensor.hpp"
#include <string>
#include <utility>

namespace llaisys {

class Weights {
private:
    tensor_t _tensor;
    std::string _name;

public:
    Weights(std::string name, tensor_t tensor)
        : _tensor(std::move(tensor)), _name(std::move(name)) {}
    const tensor_t &weights() const { return _tensor; }
    const std::string &name() const { return _name; }
    llaisysDataType_t dtype();
    // llaisysDeviceType_t device_type();
};
using Weights_t = std::shared_ptr<llaisys::Weights>;
} // namespace llaisys