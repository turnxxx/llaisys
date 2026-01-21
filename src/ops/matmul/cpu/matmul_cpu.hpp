#pragma once

#include "../../../tensor/tensor.hpp"
// compute c=a*transpose(b)
namespace llaisys::ops::cpu {
void transpose_matmul(tensor_t c, tensor_t a, tensor_t b, float scale);
}
