#pragma once

#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::nvidia {
void transpose_matmul(tensor_t c, tensor_t a, tensor_t b, float scale);
}
