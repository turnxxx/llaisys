#pragma once

#include "../base_weights.hpp"

namespace llaisys::Qwen2 {
struct mlp_weights {
    Weights_t gate;
    Weights_t up;
    Weights_t down;
};

} // namespace llaisys::Qwen2
