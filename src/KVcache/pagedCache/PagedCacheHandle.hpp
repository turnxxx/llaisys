#pragma once

#include "../CacheHandle.hpp"
#include "PagedCache.hpp"

namespace llaisys::KVcache {

class PagedCacheHandle : public CacheHandle {
public:
    explicit PagedCacheHandle(std::shared_ptr<PagedCache> paged_cache);
    ~PagedCacheHandle() override;

    void reset() override;
    size_t seq_len() const override;

    void append(size_t layer, llaisys::tensor_t& k, llaisys::tensor_t& v,
                size_t token_idx = 0) override;
    void get(llaisys::tensor_t& k, llaisys::tensor_t& v, size_t layer) override;

    bool is_paged() const override { return true; }
    llaisys::tensor_t paged_kv_data(size_t layer) const override;
    llaisys::tensor_t kv_indptr() const override;
    llaisys::tensor_t kv_indices() const override;
    llaisys::tensor_t kv_last_page_len() const override;
    int block_size() const override;

private:
    void refresh_metadata();

    std::shared_ptr<PagedCache> paged_cache_;
    int request_id_ = -1;
    size_t context_len_ = 0;
    llaisys::tensor_t kv_indptr_;
    llaisys::tensor_t kv_indices_;
    llaisys::tensor_t kv_last_page_len_;
};

} // namespace llaisys::KVcache
