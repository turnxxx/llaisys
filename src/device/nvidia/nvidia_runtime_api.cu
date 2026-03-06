#include "../runtime_api.hpp"

#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>

namespace llaisys::device::nvidia {

namespace runtime_api {
namespace {
// 将框架内存拷贝方向映射到 CUDA 的 memcpy 方向。
cudaMemcpyKind toCudaMemcpyKind(llaisysMemcpyKind_t kind) {
    switch (kind) {
    case LLAISYS_MEMCPY_H2H:
        return cudaMemcpyHostToHost;
    case LLAISYS_MEMCPY_H2D:
        return cudaMemcpyHostToDevice;
    case LLAISYS_MEMCPY_D2H:
        return cudaMemcpyDeviceToHost;
    case LLAISYS_MEMCPY_D2D:
        return cudaMemcpyDeviceToDevice;
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

// 统一 CUDA 错误检查，保证失败时有清晰报错信息。
void checkCuda(cudaError_t status, const char *op) {
    if (status != cudaSuccess) {
        std::cerr << "[ERROR] CUDA runtime call failed: " << op
                  << ", message: " << cudaGetErrorString(status) << "."
                  << std::endl;
        throw std::runtime_error("CUDA runtime call failed");
    }
}
} // namespace

int getDeviceCount() {
    // 获取当前进程可见的 CUDA 设备数量。
    int count = 0;
    checkCuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    return count;
}

void setDevice(int device_id) {
    // 切换当前线程使用的 CUDA 设备（上下文切换核心入口）。
    checkCuda(cudaSetDevice(device_id), "cudaSetDevice");
}

void deviceSynchronize() {
    // 同步当前设备，确保此前提交的工作都已完成。
    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

llaisysStream_t createStream() {
    // 创建非阻塞 stream，便于后续算子异步执行。
    cudaStream_t stream = nullptr;
    checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
              "cudaStreamCreateWithFlags");
    return reinterpret_cast<llaisysStream_t>(stream);
}

void destroyStream(llaisysStream_t stream) {
    // 框架里 CPU 使用空 stream；GPU 如果是空则直接返回。
    if (stream == nullptr) {
        return;
    }
    checkCuda(cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream)),
              "cudaStreamDestroy");
}
void streamSynchronize(llaisysStream_t stream) {
    // 同步指定 stream；空 stream 退化为设备级同步。
    if (stream == nullptr) {
        return deviceSynchronize();
    }
    checkCuda(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)),
              "cudaStreamSynchronize");
}

void *mallocDevice(size_t size) {
    // 分配 GPU 显存，供 Tensor/Storage 创建设备张量使用。
    if (size == 0) {
        return nullptr;
    }
    void *ptr = nullptr;
    checkCuda(cudaMalloc(&ptr, size), "cudaMalloc");
    return ptr;
}

void freeDevice(void *ptr) {
    if (ptr == nullptr) {
        return;
    }
    checkCuda(cudaFree(ptr), "cudaFree");
}

void *mallocHost(size_t size) {
    // 分配 pinned host 内存，加速 H2D/D2H 数据搬运。
    if (size == 0) {
        return nullptr;
    }
    void *ptr = nullptr;
    checkCuda(cudaMallocHost(&ptr, size), "cudaMallocHost");
    return ptr;
}

void freeHost(void *ptr) {
    if (ptr == nullptr) {
        return;
    }
    checkCuda(cudaFreeHost(ptr), "cudaFreeHost");
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    // 同步拷贝：用于 Tensor::load / Tensor::to 等立即可见的数据迁移。
    if (size == 0) {
        return;
    }
    checkCuda(cudaMemcpy(dst, src, size, toCudaMemcpyKind(kind)), "cudaMemcpy");
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    // 异步拷贝：绑定到指定 stream，便于和 kernel 并行流水。
    if (size == 0) {
        return;
    }
    checkCuda(cudaMemcpyAsync(dst,
                              src,
                              size,
                              toCudaMemcpyKind(kind),
                              reinterpret_cast<cudaStream_t>(stream)),
              "cudaMemcpyAsync");
}

static const LlaisysRuntimeAPI RUNTIME_API = {
    &getDeviceCount,
    &setDevice,
    &deviceSynchronize,
    &createStream,
    &destroyStream,
    &streamSynchronize,
    &mallocDevice,
    &freeDevice,
    &mallocHost,
    &freeHost,
    &memcpySync,
    &memcpyAsync};

} // namespace runtime_api

const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}
} // namespace llaisys::device::nvidia
