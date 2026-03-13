#pragma once

#include "../../../tensor/tensor.hpp"
#include "../../../device/nvidia/nvidia_resource.cuh"
namespace llaisys::ops::nvidia {
void add(tensor_t c, tensor_t a, tensor_t b);
}
