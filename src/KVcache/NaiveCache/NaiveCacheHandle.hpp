#pragma once

#include "../CacheHandle.hpp"
#include "NaiveCache.hpp"

namespace llaisys::KVcache {

class NaiveCacheHandle : public CacheHandle {
public:
    explicit NaiveCacheHandle(KVcache_t cache) : cache_(std::move(cache)) {}
    ~NaiveCacheHandle() override = default;

    void reset() override { cache_->reset(); }
    size_t seq_len() const override { return cache_->seq_len(); }

    void append(size_t layer, llaisys::tensor_t& k, llaisys::tensor_t& v,
                size_t token_idx = 0) override {
        cache_->append(layer, k, v, token_idx);
    }

    void get(llaisys::tensor_t& k, llaisys::tensor_t& v, size_t layer) override {
        cache_->get(k, v, layer);
    }

private:
    KVcache_t cache_;
};

} // namespace llaisys::KVcache
