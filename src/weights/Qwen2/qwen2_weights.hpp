#pragma once

#include "../base_weights.hpp"
#include "layer_weights.hpp"

#include <vector>

namespace llaisys::Qwen2 {
struct qwen2_weights {
    Weights_t embed_tokens;
    Weights_t final_norm;
    Weights_t lm_head;
    std::vector<layer_weights> layers;
};

} // namespace llaisys::Qwen2
