#pragma once
#include "../../../tensor/tensor.hpp"
#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::cpu {
void embedding(tensor_t out, tensor_t index, tensor_t weight);
}