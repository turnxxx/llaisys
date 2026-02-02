#include "KVcacheBase.hpp"
namespace llaisys::KVcache {
size_t KVcacheBase::seq_len() {
    switch (this->dtype()) {
    case LLAISYS_DTYPE_BF16:
        return this->used_bytes_ / 2;
        break;
    case LLAISYS_DTYPE_F16:
        return this->used_bytes_ / 2;
        break;
    case LLAISYS_DTYPE_F32:
        return this->used_bytes_ / 4;
        break;
    default:
        throw "KVcache::seq_len(): Unsupported data type";
        break;
    }
}
size_t KVcacheBase::capacity() {
    switch (this->dtype()) {
    case LLAISYS_DTYPE_BF16:
        return this->total_bytes_ / 2;
        break;
    case LLAISYS_DTYPE_F16:
        return this->total_bytes_ / 2;
        break;
    case LLAISYS_DTYPE_F32:
        return this->total_bytes_ / 4;
        break;
    default:
        throw "KVcache::seq_len(): Unsupported data type";
        break;
    }
}
CacheMeta KVcacheBase::meta() {
    return this->meta_;
}
llaisysDataType_t KVcacheBase::dtype() {
    return this->dtype_;
}
llaisysDeviceType_t KVcacheBase::device() {
    return this->device_;
}
} // namespace llaisys::KVcache