#include "nvidia_resource.cuh"

namespace llaisys::device::nvidia {

Resource::Resource(int device_id)
    : llaisys::device::DeviceResource(LLAISYS_DEVICE_NVIDIA, device_id) {
    _op_ctx.device_id = device_id;
    _op_ctx.stream = nullptr;
}

Resource::~Resource() = default;

const OpExecutionContext &Resource::opContext() const {
    return _op_ctx;
}

} // namespace llaisys::device::nvidia
