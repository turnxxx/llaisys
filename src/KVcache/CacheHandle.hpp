#pragma once

#include "src/tensor/tensor.hpp"
#include <cstddef>
#include <memory>

namespace llaisys::KVcache {

class CacheHandle;
using CacheHandle_t = std::shared_ptr<CacheHandle>;

class CacheHandle {
public:
    virtual ~CacheHandle() = default;
    virtual void reset() = 0;
    virtual size_t seq_len() const = 0;
    virtual void append(size_t layer, llaisys::tensor_t& k, llaisys::tensor_t& v,
                        size_t token_idx = 0) = 0;
    virtual void get(llaisys::tensor_t& k, llaisys::tensor_t& v, size_t layer) = 0;
    virtual bool is_paged() const { return false; }
    virtual llaisys::tensor_t paged_kv_data(size_t layer) const { (void)layer; return nullptr; }
    virtual llaisys::tensor_t kv_indptr() const { return nullptr; }
    virtual llaisys::tensor_t kv_indices() const { return nullptr; }
    virtual llaisys::tensor_t kv_last_page_len() const { return nullptr; }
    virtual int block_size() const { return 0; }
};

} // namespace llaisys::KVcache
