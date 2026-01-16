#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <numeric>
#include <sstream>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}
// 检查张量的形状和步长，判断它在内存中是否连续
bool Tensor::isContiguous() const {
    // TO_BE_IMPLEMENTED();
    for (size_t i = 0; i < this->ndim() - 1; i++) {
        if (this->strides()[i] != this->strides()[i + 1] * static_cast<ptrdiff_t>(this->shape()[i + 1])) {
            return false;
        }
    }
    if (this->strides()[this->ndim() - 1] != 1) {
        return false;
    }
    return true;
}

/* 创建一个新张量，通过拆分或合并原始维度将原始张量重塑为给定形状。不涉及数据传输。
例如，通过合并最后两个维度，将形状为(2, 3, 5)的张量更改为(2, 15)。
这个函数不是简单地改变张量的形状那么简单，尽管测试会通过。
如果新视图与原始张量不兼容，它应该引发错误。
想想一个形状为(2, 3, 5)、步长为(30, 10, 1)的张量。
你还能在不传输数据的情况下将其重塑为(2, 15)吗？ */
tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    // 要求元素总数一致
    size_t new_numel = 1;
    for (auto dim : shape) {
        new_numel *= dim;
    }
    ASSERT(new_numel == this->numel(), "view error! numel mismatch");

    // 仅支持连续内存的 view（非连续需要更复杂的兼容性检查）
    ASSERT(isContiguous(), "view error! tensor is not contiguous");

    // 计算新的连续步长
    std::vector<ptrdiff_t> new_strides(shape.size());
    ptrdiff_t stride = 1;
    for (size_t i = shape.size(); i-- > 0;) {
        new_strides[i] = stride;
        stride *= static_cast<ptrdiff_t>(shape[i]);
    }
    TensorMeta new_meta{this->dtype(), shape, std::move(new_strides)};
    return std::shared_ptr<Tensor>(new Tensor(new_meta, this->_storage, this->_offset));
}
/* 创建一个新张量，改变原始张量维度的顺序。转置可以通过这个函数实现，而无需移动数据。 */
tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    //  TO_BE_IMPLEMENTED();
    // 先计算新的shape与stride
    std::vector<size_t> new_shape(this->shape().size());
    std::vector<ptrdiff_t> new_strides(this->strides().size());
    // 计算新的shape与stride
    // 新的shape计算
    for (size_t i = 0; i < this->shape().size(); i++) {
        new_shape[i] = this->shape()[order[i]];
        new_strides[i] = this->strides()[order[i]];
    }
    TensorMeta new_meta{this->dtype(), new_shape, new_strides};
    return std::shared_ptr<Tensor>(new Tensor(new_meta, this->_storage, this->_offset));
}
// 创建一个新张量，沿给定维度，start（包含）和end（不包含）索引对原始张量进行切片操作。
tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    // 计算新的shape
    std::vector<size_t> new_shape = this->shape();
    new_shape[dim] = end - start;
    // 计算新的offset
    size_t new_offset = this->_offset;
    new_offset += start * this->strides()[dim] * this->elementSize();
    TensorMeta new_meta{this->dtype(), new_shape, this->strides()};
    return std::shared_ptr<Tensor>(new Tensor(new_meta, this->_storage, new_offset));
}

// 将主机（cpu）数据加载到张量（可以在设备上）。
// 查看构造函数了解如何获取当前设备上下文的运行时API，并执行从主机到设备的内存复制。
void Tensor::load(const void *src_) {
    // step1:获取运行时设备类型
    llaisysDeviceType_t device_type = core::context().runtime().deviceType();
    if (device_type == LLAISYS_DEVICE_CPU) {
        // 当前上下文在cpu上
        std::memcpy(this->data(), src_, this->numel() * this->elementSize());
    } else {
        // 当前上下文在设备上，同步拷贝
        core::context().runtime().api()->memcpy_sync(this->data(), src_, this->numel() * this->elementSize(), LLAISYS_MEMCPY_H2D);
    }
}

tensor_t Tensor::contiguous() const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

} // namespace llaisys
