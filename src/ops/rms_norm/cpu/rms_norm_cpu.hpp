#include "../../../tensor/tensor.hpp"
#include "../../../utils.hpp"

#include <cmath>
#include <cstring>
namespace llaisys::ops::cpu {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps);
}