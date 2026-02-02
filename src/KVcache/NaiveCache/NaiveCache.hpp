#pragma once
#include "../../utils.hpp"
#include "../KVcacheBase.hpp"
namespace llaisys::KVcache {
// 只能用一次的最简单KVcache，固定一块内存，用完了就没了
class NaiveCache : public KVcacheBase {
private:
    std::vector<size_t> k_cur_len_;
    std::vector<size_t> v_cur_len_;
    llaisys::tensor_t k_cache_;
    llaisys::tensor_t v_cache_;

public:
    ~NaiveCache() override;
    void reset() override;
    static KVcache_t create(llaisys::KVcache::CacheMeta meta_,
                            llaisysDeviceType_t device_, llaisysDataType_t dtype_,
                            llaisys::KVcache::IKVAllocator *allocator_ = nullptr) {
        auto cache = std::make_shared<NaiveCache>();
        cache->init(meta_, device_, dtype_, allocator_);
        LOG_INFO("NaiveCache::create:complete");
#ifdef LLAISYS_ENABLE_LOG
        if (cache->k_cache_) {
            LOG_INFO("NaiveCache::k_cache_ " << cache->k_cache_->info());
        }
        if (cache->v_cache_) {
            LOG_INFO("NaiveCache::v_cache_ " << cache->v_cache_->info());
        }
#endif

        return cache;
    }

    void init(llaisys::KVcache::CacheMeta meta_,
              llaisysDeviceType_t device_, llaisysDataType_t dtype_,
              llaisys::KVcache::IKVAllocator *allocator_) override;
    void reserve(size_t max_seq) override;
    bool ensure(size_t seq_len) override;
    void append(size_t layer, llaisys::tensor_t &k,
                llaisys::tensor_t &v,
                size_t token_idx = 0) override; // K/V_cache[layer]:[seq_len,nkvhead,d]->[seq_len+1,nkvhead,d]
    void get(llaisys::tensor_t &k, llaisys::tensor_t &v,
             size_t layer) override; // 得到K/V_cache[layer]
};

} // namespace llaisys::KVcache
