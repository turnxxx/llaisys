#include "naive_session.hpp"
#include "../../KVcache/NaiveCache/NaiveCache.hpp"

namespace llaisys::model {
void naive_session::init(const llaisys::model::meta_data &meta_data,
                         std::vector<int64_t> &tokens) {
    // 初始化naive cache
    size_t head_dim = meta_data.hidden_size / meta_data.num_attention_heads;
    llaisys::KVcache::CacheMeta cache_meta{
        meta_data.num_hidden_layers,
        meta_data.hidden_size,
        meta_data.max_position_embeddings,
        meta_data.num_attention_heads,
        head_dim,
        meta_data.num_key_value_heads,
        1};
    kv_cache_ = llaisys::KVcache::NaiveCache::create(cache_meta, LLAISYS_DEVICE_CPU,
                                                     meta_data.torch_type, nullptr);
    tokens_ = tokens;
    seq_len_ = tokens.size();
    token_pos_ = 0;
}
void naive_session::append(int64_t next_token) {
    tokens_.push_back(next_token);
    seq_len_ = tokens_.size();
    token_pos_ = seq_len_;
}
const std::vector<int64_t> &naive_session::tokens() const {
    return tokens_;
}

size_t naive_session::seq_len() const {
    return seq_len_;
}

size_t naive_session::token_pos() const {
    return token_pos_;
}

KVcache_t naive_session::kv_cache() const {
    return kv_cache_;
}
} // namespace llaisys::model