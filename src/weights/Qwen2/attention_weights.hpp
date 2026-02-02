#pragma once

#include "../base_weights.hpp"

namespace llaisys::Qwen2 {
struct attention_weights {
    Weights_t q;
    Weights_t k;
    Weights_t v;
    Weights_t o;
    Weights_t bias_q;
    Weights_t bias_k;
    Weights_t bias_v;
};

} // namespace llaisys::Qwen2