#pragma once

#include "../../utils.hpp"
#include "../KVcacheBase.hpp"
#include "pagedCache_manager.hpp"
#include "../../tensor/tensor.hpp"

#include <memory>
#include <vector>

namespace llaisys::KVcache {

// 基于 KVCacheManager 的包装类：
// - 对外兼容 KVcacheBase 通用接口
// - 额外暴露 pagedAttention 所需的 request/block-table/slot-mapping 接口
class PagedCache : public KVcacheBase {
public:
    ~PagedCache() override;

    static KVcache_t create(
        llaisys::KVcache::CacheMeta meta_,
        llaisysDeviceType_t device_,
        int device_id,
        llaisysDataType_t dtype_,
        llaisys::KVcache::IKVAllocator* allocator_ = nullptr);

    // ---- KVcacheBase 通用接口 ----
    void reset() override;
    void init(
        llaisys::KVcache::CacheMeta meta_,
        llaisysDeviceType_t device_,
        int device_id,
        llaisysDataType_t dtype_,
        llaisys::KVcache::IKVAllocator* allocator_) override;
    void reserve(size_t max_seq) override;
    bool ensure(size_t seq_len) override;
    void append(size_t layer, llaisys::tensor_t& k, llaisys::tensor_t& v, size_t token_idx = 0) override;
    void get(llaisys::tensor_t& k, llaisys::tensor_t& v, size_t layer) override;

    // ---- pagedAttention 专用接口 ----
    int add_request();
    void remove_request(int request_id);
    bool has_request(int request_id) const;

    ::AllocResult allocate_tokens(int request_id, int num_tokens);
    bool can_allocate(int request_id, int num_tokens) const;

    std::vector<int64_t> get_slot_mapping(int request_id, const std::vector<int>& positions) const;
    int64_t get_slot_for_token(int request_id, int position) const;
    std::vector<int32_t> get_block_table_row(int request_id) const;
    int get_context_len(int request_id) const;

    const int32_t* block_table_data() const;
    const ::BlockTable& block_table() const;
    tensor_t paged_kv_data(size_t layer) const;
    tensor_t kv_indptr() const;
    tensor_t kv_indices() const;
    tensor_t kv_last_page_len() const;

    int num_free_blocks() const;
    int num_total_blocks() const;
    float usage() const;
    int num_active_requests() const;
    int block_size() const;

    // 默认请求（单对话路径）便捷接口
    int default_request_id() const { return default_request_id_; }
    void reset_default_request();
    ::AllocResult allocate_tokens(int num_tokens);
    std::vector<int64_t> get_slot_mapping(const std::vector<int>& positions) const;
    int64_t get_slot_for_token(int position) const;
    std::vector<int32_t> get_block_table_row() const;
    int get_context_len() const;

    void write_tokens_to_pages(size_t layer,
                               const std::vector<int64_t>& slots,
                               llaisys::tensor_t& k,
                               llaisys::tensor_t& v);

    llaisysDeviceType_t cache_device() const { return device_; }
    int cache_device_id() const { return device_id_; }
    size_t computed_num_blocks() const { return computed_num_blocks_; }
    const KVCacheConfig& cache_config() const { return config_; }

private:
    static int dtype_size_bytes(llaisysDataType_t dtype);
    void rebuild_manager(size_t max_seq);
    void update_used_bytes();
    void refresh_page_metadata();

private:
    KVCacheConfig config_{};
    std::unique_ptr<::KVCacheManager> manager_;
    int default_request_id_ = -1;
    size_t computed_num_blocks_ = 0;
    std::vector<size_t> layer_seq_lens_;
    std::vector<tensor_t> paged_kv_layers_;
    tensor_t kv_indptr_;
    tensor_t kv_indices_;
    tensor_t kv_last_page_len_;
};

} // namespace llaisys::KVcache

