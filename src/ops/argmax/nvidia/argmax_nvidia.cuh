#pragma once

#include "../../../tensor/tensor.hpp"
#include "../../../device/nvidia/nvidia_resource.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
namespace llaisys::ops::nvidia {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals);
}
