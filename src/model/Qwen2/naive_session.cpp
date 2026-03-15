#include "naive_session.hpp"
#include "../../KVcache/NaiveCache/NaiveCache.hpp"
#include "../../KVcache/pagedCache/PagedCache.hpp"
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>

namespace llaisys::model {
namespace {
bool parse_env_bool(const char *env, bool fallback) {
    if (!env) {
        return fallback;
    }
    std::string s(env);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (s == "1" || s == "true" || s == "on" || s == "yes") {
        return true;
    }
    if (s == "0" || s == "false" || s == "off" || s == "no") {
        return false;
    }
    return fallback;
}

bool should_use_paged_attention(const llaisys::model::meta_data &meta_data,
                                llaisysDeviceType_t device_type) {
    if (device_type != LLAISYS_DEVICE_NVIDIA) {
        return false;
    }
    bool enabled = meta_data.use_paged_attention;
    enabled = parse_env_bool(std::getenv("LLAISYS_USE_PAGED_ATTENTION"), enabled);
    return enabled;
}
} // namespace

void naive_session::init(const llaisys::model::meta_data &meta_data,
                         std::vector<int64_t> &tokens,
                         llaisysDeviceType_t device_type,
                         int device_id) {
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
    const bool enable_paged = should_use_paged_attention(meta_data, device_type);
    if (enable_paged) {
        kv_cache_ = llaisys::KVcache::PagedCache::create(
            cache_meta, device_type, device_id, meta_data.torch_type, nullptr);
        LOG_INFO("naive_session::init: use PagedCache (paged attention enabled)");
    } else {
        kv_cache_ = llaisys::KVcache::NaiveCache::create(
            cache_meta, device_type, device_id, meta_data.torch_type, nullptr);
        LOG_INFO("naive_session::init: use NaiveCache (paged attention disabled)");
    }
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