#include "PagedCacheHandle.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include <stdexcept>

namespace llaisys::KVcache {

PagedCacheHandle::PagedCacheHandle(std::shared_ptr<PagedCache> paged_cache)
    : paged_cache_(std::move(paged_cache)) {
    ASSERT(paged_cache_ != nullptr, "PagedCacheHandle: paged_cache is null");
    request_id_ = paged_cache_->add_request();
    context_len_ = 0;

    auto device = paged_cache_->cache_device();
    auto device_id = paged_cache_->cache_device_id();
    auto num_blocks = paged_cache_->computed_num_blocks();

    kv_indptr_ = llaisys::Tensor::create({2}, LLAISYS_DTYPE_I32, device, device_id);
    kv_last_page_len_ = llaisys::Tensor::create({1}, LLAISYS_DTYPE_I32, device, device_id);
    kv_indices_ = llaisys::Tensor::create({num_blocks}, LLAISYS_DTYPE_I32, device, device_id);
    refresh_metadata();
}

PagedCacheHandle::~PagedCacheHandle() {
    if (paged_cache_ && request_id_ >= 0) {
        paged_cache_->remove_request(request_id_);
    }
}

void PagedCacheHandle::reset() {
    if (paged_cache_ && request_id_ >= 0 && paged_cache_->has_request(request_id_)) {
        paged_cache_->remove_request(request_id_);
    }
    request_id_ = paged_cache_->add_request();
    context_len_ = 0;
    refresh_metadata();
}

size_t PagedCacheHandle::seq_len() const {
    return context_len_;
}

void PagedCacheHandle::append(size_t layer, llaisys::tensor_t& k, llaisys::tensor_t& v,
                              size_t token_idx) {
    ASSERT(paged_cache_ != nullptr, "PagedCacheHandle::append: paged_cache is null");
    ASSERT(request_id_ >= 0, "PagedCacheHandle::append: invalid request_id");
    ASSERT(k != nullptr && v != nullptr, "PagedCacheHandle::append: k/v is null");

    const int seq = static_cast<int>(k->shape()[0]);
    if (seq <= 0) return;

    const int cur_context = paged_cache_->get_context_len(request_id_);
    const int target_context = static_cast<int>(token_idx) + seq;
    if (layer == 0 && target_context > cur_context) {
        const int need = target_context - cur_context;
        auto result = paged_cache_->allocate_tokens(request_id_, need);
        ASSERT(result.success, "PagedCacheHandle::append: allocate_tokens failed");
    }

    std::vector<int> positions(static_cast<size_t>(seq));
    for (int i = 0; i < seq; ++i) {
        positions[static_cast<size_t>(i)] = static_cast<int>(token_idx) + i;
    }
    auto slots = paged_cache_->get_slot_mapping(request_id_, positions);
    paged_cache_->write_tokens_to_pages(layer, slots, k, v);

    if (layer == 0) {
        context_len_ = static_cast<size_t>(paged_cache_->get_context_len(request_id_));
        refresh_metadata();
    }
}

void PagedCacheHandle::get(llaisys::tensor_t& k, llaisys::tensor_t& v, size_t layer) {
    (void)k; (void)v; (void)layer;
    throw std::runtime_error("PagedCacheHandle::get is not supported; use pagedAttention interfaces");
}

llaisys::tensor_t PagedCacheHandle::paged_kv_data(size_t layer) const {
    return paged_cache_->paged_kv_data(layer);
}

llaisys::tensor_t PagedCacheHandle::kv_indptr() const {
    return kv_indptr_;
}

llaisys::tensor_t PagedCacheHandle::kv_indices() const {
    return kv_indices_;
}

llaisys::tensor_t PagedCacheHandle::kv_last_page_len() const {
    return kv_last_page_len_;
}

int PagedCacheHandle::block_size() const {
    return paged_cache_->block_size();
}

void PagedCacheHandle::refresh_metadata() {
    ASSERT(paged_cache_ != nullptr, "PagedCacheHandle::refresh_metadata: paged_cache is null");

    const auto row = paged_cache_->get_block_table_row(request_id_);
    const int context = paged_cache_->get_context_len(request_id_);
    const int num_pages = static_cast<int>(row.size());
    const int bs = paged_cache_->cache_config().block_size;
    const int last_page_len = (context == 0) ? 0 : ((context - 1) % bs) + 1;

    auto num_blocks = paged_cache_->computed_num_blocks();
    std::vector<int32_t> indptr_host{0, num_pages};
    std::vector<int32_t> last_host{last_page_len};
    std::vector<int32_t> indices_host(num_blocks, -1);
    for (int i = 0; i < num_pages; ++i) {
        indices_host[static_cast<size_t>(i)] = row[static_cast<size_t>(i)];
    }

    auto device = paged_cache_->cache_device();
    auto device_id = paged_cache_->cache_device_id();
    llaisys::core::context().setDevice(device, device_id);
    auto& runtime = llaisys::core::context().runtime();
    auto api = runtime.api();
    auto stream = runtime.stream();
    const llaisysMemcpyKind_t kind =
        (device == LLAISYS_DEVICE_CPU) ? LLAISYS_MEMCPY_H2H : LLAISYS_MEMCPY_H2D;

    api->memcpy_async(kv_indptr_->data(), indptr_host.data(),
                      indptr_host.size() * sizeof(int32_t), kind, stream);
    api->memcpy_async(kv_last_page_len_->data(), last_host.data(),
                      last_host.size() * sizeof(int32_t), kind, stream);
    api->memcpy_async(kv_indices_->data(), indices_host.data(),
                      indices_host.size() * sizeof(int32_t), kind, stream);
    api->stream_synchronize(stream);
}

} // namespace llaisys::KVcache
