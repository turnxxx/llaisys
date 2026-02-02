#pragma once
#include "llaisys.h"
#include "src/tensor/tensor.hpp"
#include <vector>
// 抽象的KV-cache类，支持多后端，分布式，不同的内存管理模式

namespace llaisys::KVcache {
class KVcacheBase;
using KVcache_t = std::shared_ptr<KVcacheBase>;
struct CacheMeta {
    size_t nlayer;
    size_t hidden_size;
    size_t max_seq = 128;
    size_t nhead;
    size_t head_dim = 128;
    size_t n_kv_heads = 2;
    size_t batch = 1;
};
// KVcache的抽象内存分配器
struct IKVAllocator {
    virtual ~IKVAllocator() = default;
    virtual llaisys::tensor_t alloc(const std::vector<size_t> &shape,
                                    llaisysDataType_t dtype,
                                    llaisysDeviceType_t device,
                                    int device_id)
        = 0;
    virtual void free(const llaisys::tensor_t &tensor) = 0;
};

class KVcacheBase {

protected:
    llaisysDeviceType_t device_;
    llaisysDataType_t dtype_;
    llaisys::KVcache::CacheMeta meta_;
    size_t total_bytes_;      // KVcache管理的内存块大小(K的部分)
    void *ptr_;               // 设备内存指针
    size_t offset_;           // 管理的内存偏移量
    IKVAllocator *allocator_; // 内存分配器
    size_t used_bytes_;       // 当前的缓存消耗了多少内存

public:
    virtual ~KVcacheBase() = default;
    virtual void reset() = 0;
    virtual void init(llaisys::KVcache::CacheMeta meta_,
                      llaisysDeviceType_t device_, llaisysDataType_t dtype_,
                      llaisys::KVcache::IKVAllocator *allocator_)
        = 0;
    size_t seq_len();                         // 当前缓存长度(seq长度，而非字节)
    size_t capacity();                        // 最大缓存长度(seq长度，而非字节)
    virtual void reserve(size_t max_seq) = 0; // 扩容
    virtual bool ensure(size_t seq_len) = 0;  // 返回当前容量是否支持再加seq_len长度
    virtual void append(size_t layer, llaisys::tensor_t &k,
                        llaisys::tensor_t &v,
                        size_t token_idx = 0)
        = 0; // 写入新的cache
    virtual void get(llaisys::tensor_t &k, llaisys::tensor_t &v,
                     size_t layer)
        = 0; // 读取cache
    llaisys::KVcache::CacheMeta meta();
    llaisysDataType_t dtype();
    llaisysDeviceType_t device();
};
}; // namespace llaisys::KVcache
