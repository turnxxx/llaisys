#include "model_base.hpp"

namespace llaisys::model {

void ModelBase::setDeviceSpec(const DeviceSpec &device) {
    this->_device = device;
}
void ModelBase::setParallelSpec(const ParallelSpec &parallel) {
    this->_parallel = parallel;
}

} // namespace llaisys::model