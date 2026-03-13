#pragma once

#include "../device_resource.hpp"
#include "nvidia_common.cuh"

namespace llaisys::device::nvidia {
class Resource : public llaisys::device::DeviceResource {
private:
    OpExecutionContext _op_ctx;

public:
    Resource(int device_id);
    ~Resource();

    const OpExecutionContext &opContext() const;
    void setStream(llaisysStream_t stream);
};

Resource &getResource(int device_id);
} // namespace llaisys::device::nvidia
