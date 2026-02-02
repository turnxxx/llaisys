#include "../../KVcache/KVcacheBase.hpp"
#include "../../model/model_utils.hpp"
#include "../../ops/ops.hpp"
#include "../../tensor/tensor.hpp"
#include "../../weights/Qwen2/qwen2_weights.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace llaisys::Qwen2 {
tensor_t qwen2_decoder(
    tensor_t &hidden_states,
    const llaisys::Qwen2::layer_weights &weights,
    llaisys::KVcache::KVcache_t kv_cache,
    const llaisys::model::meta_data &meta_data,
    size_t token_pos,
    size_t layer);
}