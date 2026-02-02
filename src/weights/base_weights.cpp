#include "base_weights.hpp"

llaisysDataType_t llaisys::Weights::dtype() {
    ASSERT(this->_tensor != nullptr, "Weights: tensor is null");
    return this->_tensor->dtype();
}

/* llaisysDeviceType_t llaisys::Weights::deviceType() {
    ASSERT(this->_tensor != nullptr, "Weights: tensor is null");
    return this->_tensor->deviceType();
} */