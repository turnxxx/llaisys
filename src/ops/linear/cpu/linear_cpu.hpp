
#include "../../../tensor/tensor.hpp"
#include "../../../utils.hpp"

#include <cmath>
#include <cstring>
namespace llaisys::ops::cpu {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias);
}
