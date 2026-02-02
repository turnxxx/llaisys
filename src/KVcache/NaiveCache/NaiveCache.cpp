#include "NaiveCache.hpp"
#include <cstring>

namespace llaisys::KVcache {
void NaiveCache::init(llaisys::KVcache::CacheMeta meta_,
                      llaisysDeviceType_t device_, llaisysDataType_t dtype_,
                      llaisys::KVcache::IKVAllocator *allocator_) {
    this->meta_ = meta_;
    this->device_ = device_;
    this->dtype_ = dtype_;
    this->allocator_ = allocator_;
    // 分配k_cache_和v_cache_
    std::vector<size_t> cache_shape_{meta_.nlayer, meta_.max_seq,
                                     meta_.n_kv_heads, meta_.head_dim};
    /* std::vector<ptrdiff_t> cache_strides_(cache_shape_.size(), 1);
    for (size_t i = cache_shape_.size(); i-- > 0;) {
        if (i + 1 < cache_shape_.size()) {
            cache_strides_[i] = static_cast<ptrdiff_t>(cache_strides_[i + 1] * static_cast<ptrdiff_t>(cache_shape_[i + 1]));
        }
    }
    llaisys::TensorMeta tensor_meta_{dtype_, cache_shape_, std::move(cache_strides_)}; */
    this->k_cache_ = llaisys::Tensor::create(cache_shape_, dtype_, device_);
    this->v_cache_ = llaisys::Tensor::create(cache_shape_, dtype_, device_);
    this->k_cur_len_ = std::vector<size_t>(meta_.nlayer, 0);
    this->v_cur_len_ = std::vector<size_t>(meta_.nlayer, 0);
    // cache statistics (track seq_len in bytes)
    total_bytes_ = meta_.max_seq * k_cache_->elementSize();
    used_bytes_ = 0;
}

void NaiveCache::reset() {
    for (size_t i = 0; i < meta_.nlayer; ++i) {
        k_cur_len_[i] = 0;
        v_cur_len_[i] = 0;
    }
    used_bytes_ = 0;
}

void NaiveCache::reserve(size_t max_seq) {
    if (max_seq <= meta_.max_seq) {
        return;
    }
    meta_.max_seq = max_seq;
    if (k_cache_) {
        k_cache_.reset();
    }
    if (v_cache_) {
        v_cache_.reset();
    }
    init(meta_, device_, dtype_, allocator_);
}
// 当前上下文长度是否够用
bool NaiveCache::ensure(size_t seq_len) {
    return seq_len <= this->meta_.max_seq;
};
// 更新kv_cache
void NaiveCache::append(size_t layer, llaisys::tensor_t &k,
                        llaisys::tensor_t &v,
                        size_t token_idx) {
    // 当前层的存储的长度
    size_t k_len = k_cur_len_[layer];
    size_t v_len = v_cur_len_[layer];
    ASSERT(k_len == v_len, "NaiveCache::append: k_len and v_len mismatch");
    ASSERT(ensure(k_len + k->shape()[0]), "NaiveCache::append: Cache memory is not enough");
    //  在tensor里append数据
    size_t n_kv = meta_.n_kv_heads;
    size_t d = meta_.head_dim;
    size_t max_seq = meta_.max_seq;
    ASSERT(k->shape().size() == 3 && v->shape().size() == 3,
           "NaiveCache::append: k/v must be [seq,n_kv,head_dim]");
    ASSERT(k->shape()[1] == n_kv && k->shape()[2] == d,
           "NaiveCache::append: k shape mismatch");
    ASSERT(v->shape()[1] == n_kv && v->shape()[2] == d,
           "NaiveCache::append: v shape mismatch");

    size_t seq = k->shape()[0];
    size_t elem_size = k_cache_->elementSize();
    size_t row_elems = n_kv * d;

    std::byte *k_dst = k_cache_->data();
    std::byte *v_dst = v_cache_->data();
    const std::byte *k_src = k->data();
    const std::byte *v_src = v->data();

    for (size_t t = 0; t < seq; ++t) {
        size_t k_base = ((layer * max_seq + (k_len + t)) * row_elems);
        size_t v_base = ((layer * max_seq + (v_len + t)) * row_elems);
        std::memcpy(k_dst + k_base * elem_size, k_src + t * row_elems * elem_size,
                    row_elems * elem_size);
        std::memcpy(v_dst + v_base * elem_size, v_src + t * row_elems * elem_size,
                    row_elems * elem_size);
    }

    k_cur_len_[layer] += seq;
    v_cur_len_[layer] += seq;
    // update cache statistics (track max seq_len across layers)
    size_t cur_len = k_cur_len_[layer];
    size_t cur_bytes = cur_len * k_cache_->elementSize();
    if (cur_bytes > used_bytes_) {
        used_bytes_ = cur_bytes;
    }
}
// 返回layer层的k和v,使用slice即可
void NaiveCache::get(llaisys::tensor_t &k, llaisys::tensor_t &v,
                     size_t layer) {
    ASSERT(layer < meta_.nlayer, "NaiveCache::get: layer out of range");
    size_t k_len = k_cur_len_[layer];
    size_t v_len = v_cur_len_[layer];
    ASSERT(k_len == v_len, "NaiveCache::get: k_len and v_len mismatch");

    auto k_view = k_cache_->slice(0, layer, layer + 1);
    auto v_view = v_cache_->slice(0, layer, layer + 1);
    k_view = k_view->slice(1, 0, k_len);
    v_view = v_view->slice(1, 0, v_len);

    std::vector<size_t> out_shape{k_len, meta_.n_kv_heads, meta_.head_dim};
    k = k_view->reshape(out_shape);
    v = v_view->reshape(out_shape);
}
NaiveCache::~NaiveCache() {
    k_cache_.reset();
    v_cache_.reset();
    k_cur_len_.clear();
    v_cur_len_.clear();
    allocator_ = nullptr;
    ptr_ = nullptr;
    total_bytes_ = 0;
    used_bytes_ = 0;
}
} // namespace llaisys::KVcache