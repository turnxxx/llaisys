#include "PagedCache.hpp"

#include "../../core/llaisys_core.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace llaisys::KVcache {

KVcache_t PagedCache::create(
    llaisys::KVcache::CacheMeta meta_,
    llaisysDeviceType_t device_,
    int device_id,
    llaisysDataType_t dtype_,
    llaisys::KVcache::IKVAllocator* allocator_) {
    auto cache = std::make_shared<PagedCache>();
    cache->init(meta_, device_, device_id, dtype_, allocator_);
    LOG_INFO("PagedCache::create: complete");
    return cache;
}

PagedCache::~PagedCache() {
    manager_.reset();
    layer_seq_lens_.clear();
    paged_kv_layers_.clear();
    kv_indptr_.reset();
    kv_indices_.reset();
    kv_last_page_len_.reset();
    allocator_ = nullptr;
    ptr_ = nullptr;
    total_bytes_ = 0;
    used_bytes_ = 0;
}

int PagedCache::dtype_size_bytes(llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_F16:
    case LLAISYS_DTYPE_BF16:
        return 2;
    case LLAISYS_DTYPE_F32:
        return 4;
    case LLAISYS_DTYPE_F8:
        return 1;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
        return 0;
    }
}

void PagedCache::rebuild_manager(size_t max_seq) {
    meta_.max_seq = max_seq;
    config_.num_layers = static_cast<int>(meta_.nlayer);
    config_.num_kv_heads = static_cast<int>(meta_.n_kv_heads);
    config_.head_size = static_cast<int>(meta_.head_dim);
    config_.max_num_reqs = static_cast<int>(meta_.batch);
    config_.max_model_len = static_cast<int>(meta_.max_seq);
    config_.dtype_size = dtype_size_bytes(dtype_);

    const int blocks_per_req = config_.max_num_blocks_per_req();
    computed_num_blocks_ = static_cast<size_t>(config_.max_num_reqs) *
                           static_cast<size_t>(blocks_per_req);

    manager_ = std::make_unique<::KVCacheManager>(config_, static_cast<int>(computed_num_blocks_));
    default_request_id_ = manager_->add_request();
    layer_seq_lens_.assign(meta_.nlayer, 0);

    paged_kv_layers_.clear();
    paged_kv_layers_.reserve(meta_.nlayer);
    const std::vector<size_t> page_shape{
        computed_num_blocks_,
        static_cast<size_t>(2),
        meta_.n_kv_heads,
        static_cast<size_t>(config_.block_size),
        meta_.head_dim};
    for (size_t i = 0; i < meta_.nlayer; ++i) {
        paged_kv_layers_.push_back(
            llaisys::Tensor::create(page_shape, dtype_, device_, device_id_));
    }

    kv_indptr_ = llaisys::Tensor::create({2}, LLAISYS_DTYPE_I32, device_, device_id_);
    kv_last_page_len_ = llaisys::Tensor::create({1}, LLAISYS_DTYPE_I32, device_, device_id_);
    kv_indices_ = llaisys::Tensor::create({computed_num_blocks_}, LLAISYS_DTYPE_I32, device_, device_id_);

    const int64_t one_page_kv_bytes = config_.page_size_bytes_all_layers();
    total_bytes_ =
        static_cast<size_t>(computed_num_blocks_) * static_cast<size_t>(one_page_kv_bytes / 2);
    used_bytes_ = 0;
    refresh_page_metadata();
}

void PagedCache::update_used_bytes() {
    if (!manager_ || default_request_id_ < 0 || !manager_->has_request(default_request_id_)) {
        used_bytes_ = 0;
        return;
    }
    const size_t context_len = static_cast<size_t>(manager_->get_context_len(default_request_id_));
    used_bytes_ = context_len * static_cast<size_t>(config_.dtype_size);
}

void PagedCache::refresh_page_metadata() {
    ASSERT(manager_ != nullptr, "PagedCache::refresh_page_metadata: manager is null");
    const std::vector<int32_t> row = manager_->get_block_table_row(default_request_id_);
    const int context_len = manager_->get_context_len(default_request_id_);
    const int num_pages = static_cast<int>(row.size());
    const int last_page_len =
        (context_len == 0) ? 0 : ((context_len - 1) % config_.block_size) + 1;

    std::vector<int32_t> indptr_host{0, num_pages};
    std::vector<int32_t> last_host{last_page_len};
    std::vector<int32_t> indices_host(computed_num_blocks_, -1);
    for (int i = 0; i < num_pages; ++i) {
        indices_host[static_cast<size_t>(i)] = row[static_cast<size_t>(i)];
    }

    llaisys::core::context().setDevice(device_, device_id_);
    auto &runtime = llaisys::core::context().runtime();
    auto api = runtime.api();
    auto stream = runtime.stream();
    const llaisysMemcpyKind_t kind =
        (device_ == LLAISYS_DEVICE_CPU) ? LLAISYS_MEMCPY_H2H : LLAISYS_MEMCPY_H2D;

    api->memcpy_async(kv_indptr_->data(), indptr_host.data(),
                      indptr_host.size() * sizeof(int32_t), kind, stream);
    api->memcpy_async(kv_last_page_len_->data(), last_host.data(),
                      last_host.size() * sizeof(int32_t), kind, stream);
    api->memcpy_async(kv_indices_->data(), indices_host.data(),
                      indices_host.size() * sizeof(int32_t), kind, stream);
    api->stream_synchronize(stream);
}

void PagedCache::write_tokens_to_pages(
    size_t layer,
    const std::vector<int64_t>& slots,
    llaisys::tensor_t& k,
    llaisys::tensor_t& v) {
    ASSERT(layer < paged_kv_layers_.size(), "PagedCache::write_tokens_to_pages: layer out of range");
    ASSERT(k->isContiguous() && v->isContiguous(), "PagedCache::write_tokens_to_pages: k/v must be contiguous");

    const int seq = static_cast<int>(slots.size());
    const int num_kv_heads = static_cast<int>(meta_.n_kv_heads);
    const int head_dim = static_cast<int>(meta_.head_dim);
    const int page_size = config_.block_size;
    const size_t elem_bytes = static_cast<size_t>(config_.dtype_size);

    CHECK_SAME_DEVICE(paged_kv_layers_[layer], k, v);
    llaisys::core::context().setDevice(device_, device_id_);
    auto &runtime = llaisys::core::context().runtime();
    auto api = runtime.api();
    auto stream = runtime.stream();
    const llaisysMemcpyKind_t memcpy_kind =
        (device_ == LLAISYS_DEVICE_CPU) ? LLAISYS_MEMCPY_H2H : LLAISYS_MEMCPY_D2D;

    std::byte* dst_base = paged_kv_layers_[layer]->data();
    const std::byte* k_src = k->data();
    const std::byte* v_src = v->data();

    for (int t = 0; t < seq; ++t) {
        const int64_t slot = slots[static_cast<size_t>(t)];
        ASSERT(slot >= 0, "PagedCache::write_tokens_to_pages: invalid slot");
        const int page_idx = static_cast<int>(slot / page_size);
        const int offset = static_cast<int>(slot % page_size);
        ASSERT(page_idx >= 0 && static_cast<size_t>(page_idx) < computed_num_blocks_,
               "PagedCache::write_tokens_to_pages: page index out of range");

        for (int h = 0; h < num_kv_heads; ++h) {
            const int64_t k_dst_elem =
                ((((static_cast<int64_t>(page_idx) * 2 + 0) * num_kv_heads + h) * page_size + offset) *
                 head_dim);
            const int64_t v_dst_elem =
                ((((static_cast<int64_t>(page_idx) * 2 + 1) * num_kv_heads + h) * page_size + offset) *
                 head_dim);
            const int64_t src_elem =
                ((static_cast<int64_t>(t) * num_kv_heads + h) * head_dim);

            api->memcpy_async(
                dst_base + static_cast<size_t>(k_dst_elem) * elem_bytes,
                k_src + static_cast<size_t>(src_elem) * elem_bytes,
                static_cast<size_t>(head_dim) * elem_bytes,
                memcpy_kind,
                stream);
            api->memcpy_async(
                dst_base + static_cast<size_t>(v_dst_elem) * elem_bytes,
                v_src + static_cast<size_t>(src_elem) * elem_bytes,
                static_cast<size_t>(head_dim) * elem_bytes,
                memcpy_kind,
                stream);
        }
    }
    api->stream_synchronize(stream);
}

void PagedCache::init(
    llaisys::KVcache::CacheMeta meta_,
    llaisysDeviceType_t device_,
    int device_id,
    llaisysDataType_t dtype_,
    llaisys::KVcache::IKVAllocator* allocator_) {
    ASSERT(meta_.nlayer > 0, "PagedCache::init: nlayer must be > 0");
    ASSERT(meta_.max_seq > 0, "PagedCache::init: max_seq must be > 0");
    ASSERT(meta_.n_kv_heads > 0, "PagedCache::init: n_kv_heads must be > 0");
    ASSERT(meta_.head_dim > 0, "PagedCache::init: head_dim must be > 0");
    ASSERT(meta_.batch > 0, "PagedCache::init: batch must be > 0");

    this->meta_ = meta_;
    this->device_ = device_;
    this->device_id_ = device_id;
    this->dtype_ = dtype_;
    this->allocator_ = allocator_;
    config_.block_size = 16;
    rebuild_manager(meta_.max_seq);
}

void PagedCache::reset() {
    ASSERT(manager_ != nullptr, "PagedCache::reset: manager is null");
    rebuild_manager(meta_.max_seq);
}

void PagedCache::reserve(size_t max_seq) {
    if (max_seq <= meta_.max_seq) {
        return;
    }
    ASSERT(manager_ != nullptr, "PagedCache::reserve: manager is null");
    rebuild_manager(max_seq);
}

bool PagedCache::ensure(size_t seq_len) {
    if (!manager_ || default_request_id_ < 0 || !manager_->has_request(default_request_id_)) {
        return false;
    }
    if (seq_len > meta_.max_seq) {
        return false;
    }
    const int cur = manager_->get_context_len(default_request_id_);
    if (seq_len <= static_cast<size_t>(cur)) {
        return true;
    }
    return manager_->can_allocate(default_request_id_, static_cast<int>(seq_len - cur));
}

void PagedCache::append(size_t layer, llaisys::tensor_t& k, llaisys::tensor_t& v, size_t token_idx) {
    ASSERT(manager_ != nullptr, "PagedCache::append: manager is null");
    ASSERT(default_request_id_ >= 0, "PagedCache::append: default request is invalid");
    ASSERT(layer < meta_.nlayer, "PagedCache::append: layer out of range");
    ASSERT(k != nullptr && v != nullptr, "PagedCache::append: k/v tensor is null");
    ASSERT(k->shape().size() == 3 && v->shape().size() == 3,
           "PagedCache::append: k/v must be [seq,n_kv,head_dim]");
    ASSERT(k->shape()[0] == v->shape()[0], "PagedCache::append: k/v seq mismatch");
    ASSERT(k->shape()[1] == meta_.n_kv_heads && v->shape()[1] == meta_.n_kv_heads,
           "PagedCache::append: n_kv_heads mismatch");
    ASSERT(k->shape()[2] == meta_.head_dim && v->shape()[2] == meta_.head_dim,
           "PagedCache::append: head_dim mismatch");

    const int seq = static_cast<int>(k->shape()[0]);
    if (seq <= 0) {
        return;
    }

    const int cur_context_len = manager_->get_context_len(default_request_id_);
    const int target_context_len = static_cast<int>(token_idx) + seq;
    if (layer == 0 && target_context_len > cur_context_len) {
        const int need = target_context_len - cur_context_len;
        const ::AllocResult r = manager_->allocate_tokens(default_request_id_, need);
        ASSERT(r.success, "PagedCache::append: allocate_tokens failed");
    }

    std::vector<int> positions(static_cast<size_t>(seq));
    for (int i = 0; i < seq; ++i) {
        positions[static_cast<size_t>(i)] = static_cast<int>(token_idx) + i;
    }
    const std::vector<int64_t> slots = manager_->get_slot_mapping(default_request_id_, positions);
    write_tokens_to_pages(layer, slots, k, v);
    layer_seq_lens_[layer] += static_cast<size_t>(seq);

    if (layer == 0) {
        refresh_page_metadata();
        update_used_bytes();
    }
}

void PagedCache::get(llaisys::tensor_t& k, llaisys::tensor_t& v, size_t layer) {
    (void)k;
    (void)v;
    (void)layer;
    throw std::runtime_error("PagedCache::get is not supported; use pagedAttention interfaces");
}

int PagedCache::add_request() {
    ASSERT(manager_ != nullptr, "PagedCache::add_request: manager is null");
    return manager_->add_request();
}

void PagedCache::remove_request(int request_id) {
    ASSERT(manager_ != nullptr, "PagedCache::remove_request: manager is null");
    if (request_id == default_request_id_) {
        manager_->remove_request(request_id);
        default_request_id_ = manager_->add_request();
        refresh_page_metadata();
        update_used_bytes();
        return;
    }
    manager_->remove_request(request_id);
}

bool PagedCache::has_request(int request_id) const {
    ASSERT(manager_ != nullptr, "PagedCache::has_request: manager is null");
    return manager_->has_request(request_id);
}

::AllocResult PagedCache::allocate_tokens(int request_id, int num_tokens) {
    ASSERT(manager_ != nullptr, "PagedCache::allocate_tokens(request): manager is null");
    const ::AllocResult r = manager_->allocate_tokens(request_id, num_tokens);
    if (request_id == default_request_id_) {
        refresh_page_metadata();
        update_used_bytes();
    }
    return r;
}

bool PagedCache::can_allocate(int request_id, int num_tokens) const {
    ASSERT(manager_ != nullptr, "PagedCache::can_allocate: manager is null");
    return manager_->can_allocate(request_id, num_tokens);
}

std::vector<int64_t> PagedCache::get_slot_mapping(
    int request_id,
    const std::vector<int>& positions) const {
    ASSERT(manager_ != nullptr, "PagedCache::get_slot_mapping(request): manager is null");
    return manager_->get_slot_mapping(request_id, positions);
}

int64_t PagedCache::get_slot_for_token(int request_id, int position) const {
    ASSERT(manager_ != nullptr, "PagedCache::get_slot_for_token(request): manager is null");
    return manager_->get_slot_for_token(request_id, position);
}

std::vector<int32_t> PagedCache::get_block_table_row(int request_id) const {
    ASSERT(manager_ != nullptr, "PagedCache::get_block_table_row(request): manager is null");
    return manager_->get_block_table_row(request_id);
}

int PagedCache::get_context_len(int request_id) const {
    ASSERT(manager_ != nullptr, "PagedCache::get_context_len(request): manager is null");
    return manager_->get_context_len(request_id);
}

const int32_t* PagedCache::block_table_data() const {
    ASSERT(manager_ != nullptr, "PagedCache::block_table_data: manager is null");
    return manager_->block_table_data();
}

const ::BlockTable& PagedCache::block_table() const {
    ASSERT(manager_ != nullptr, "PagedCache::block_table: manager is null");
    return manager_->block_table();
}

tensor_t PagedCache::paged_kv_data(size_t layer) const {
    ASSERT(layer < paged_kv_layers_.size(), "PagedCache::paged_kv_data: layer out of range");
    return paged_kv_layers_[layer];
}

tensor_t PagedCache::kv_indptr() const {
    return kv_indptr_;
}

tensor_t PagedCache::kv_indices() const {
    return kv_indices_;
}

tensor_t PagedCache::kv_last_page_len() const {
    return kv_last_page_len_;
}

int PagedCache::num_free_blocks() const {
    ASSERT(manager_ != nullptr, "PagedCache::num_free_blocks: manager is null");
    return manager_->num_free_blocks();
}

int PagedCache::num_total_blocks() const {
    ASSERT(manager_ != nullptr, "PagedCache::num_total_blocks: manager is null");
    return manager_->num_total_blocks();
}

float PagedCache::usage() const {
    ASSERT(manager_ != nullptr, "PagedCache::usage: manager is null");
    return manager_->usage();
}

int PagedCache::num_active_requests() const {
    ASSERT(manager_ != nullptr, "PagedCache::num_active_requests: manager is null");
    return manager_->num_active_requests();
}

int PagedCache::block_size() const {
    ASSERT(manager_ != nullptr, "PagedCache::block_size: manager is null");
    return manager_->block_size();
}

void PagedCache::reset_default_request() {
    ASSERT(manager_ != nullptr, "PagedCache::reset_default_request: manager is null");
    if (default_request_id_ >= 0 && manager_->has_request(default_request_id_)) {
        manager_->remove_request(default_request_id_);
    }
    default_request_id_ = manager_->add_request();
    std::fill(layer_seq_lens_.begin(), layer_seq_lens_.end(), 0);
    refresh_page_metadata();
    update_used_bytes();
}

::AllocResult PagedCache::allocate_tokens(int num_tokens) {
    ASSERT(default_request_id_ >= 0, "PagedCache::allocate_tokens(default): default request invalid");
    return allocate_tokens(default_request_id_, num_tokens);
}

std::vector<int64_t> PagedCache::get_slot_mapping(const std::vector<int>& positions) const {
    ASSERT(default_request_id_ >= 0, "PagedCache::get_slot_mapping(default): default request invalid");
    return get_slot_mapping(default_request_id_, positions);
}

int64_t PagedCache::get_slot_for_token(int position) const {
    ASSERT(default_request_id_ >= 0, "PagedCache::get_slot_for_token(default): default request invalid");
    return get_slot_for_token(default_request_id_, position);
}

std::vector<int32_t> PagedCache::get_block_table_row() const {
    ASSERT(default_request_id_ >= 0, "PagedCache::get_block_table_row(default): default request invalid");
    return get_block_table_row(default_request_id_);
}

int PagedCache::get_context_len() const {
    ASSERT(default_request_id_ >= 0, "PagedCache::get_context_len(default): default request invalid");
    return get_context_len(default_request_id_);
}

} // namespace llaisys::KVcache

