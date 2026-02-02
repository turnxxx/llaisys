#pragma once

#include "attention_weights.hpp"
#include "layer_norm_weights.hpp"
#include "mlp_weights.hpp"

namespace llaisys::Qwen2 {

struct layer_weights {
    layer_norm_weights input_layernorm;
    attention_weights attention;
    layer_norm_weights post_attention_layernorm;
    mlp_weights mlp;
};

} // namespace llaisys::Qwen2
