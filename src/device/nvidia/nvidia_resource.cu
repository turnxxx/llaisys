#include "nvidia_resource.cuh"

#include "../runtime_api.hpp"

#include <cuda_runtime.h>

#include <iostream>
#include <memory>
#include <stdexcept>
#include <unordered_map>

namespace llaisys::device::nvidia {
namespace {
void checkCuda(cudaError_t status, const char *op) {
    if (status != cudaSuccess) {
        std::cerr << "[ERROR] CUDA runtime call failed: " << op
                  << ", message: " << cudaGetErrorString(status) << "."
                  << std::endl;
        throw std::runtime_error("CUDA runtime call failed");
    }
}

void checkCublas(cublasStatus_t status, const char *op) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "[ERROR] cuBLAS call failed: " << op
                  << ", status=" << static_cast<int>(status) << "."
                  << std::endl;
        throw std::runtime_error("cuBLAS call failed");
    }
}

void checkCublasLt(cublasStatus_t status, const char *op) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "[ERROR] cuBLASLt call failed: " << op
                  << ", status=" << static_cast<int>(status) << "."
                  << std::endl;
        throw std::runtime_error("cuBLASLt call failed");
    }
}
} // namespace

Resource::Resource(int device_id)
    : llaisys::device::DeviceResource(LLAISYS_DEVICE_NVIDIA, device_id) {
    _op_ctx.device_id = device_id;
    _op_ctx.stream = nullptr;
    checkCuda(cudaSetDevice(device_id), "cudaSetDevice");
    checkCublas(cublasCreate(&_op_ctx.cublas_handle), "cublasCreate");
    checkCublasLt(cublasLtCreate(&_op_ctx.cublaslt_handle), "cublasLtCreate");
    checkCublas(cublasSetStream(_op_ctx.cublas_handle,
                                reinterpret_cast<cudaStream_t>(_op_ctx.stream)),
                "cublasSetStream");
}

Resource::~Resource() {
    if (_op_ctx.cublas_handle != nullptr) {
        checkCublas(cublasDestroy(_op_ctx.cublas_handle), "cublasDestroy");
        _op_ctx.cublas_handle = nullptr;
    }
    if (_op_ctx.cublaslt_handle != nullptr) {
        checkCublasLt(cublasLtDestroy(_op_ctx.cublaslt_handle), "cublasLtDestroy");
        _op_ctx.cublaslt_handle = nullptr;
    }
}

const OpExecutionContext &Resource::opContext() const {
    return _op_ctx;
}

void Resource::setStream(llaisysStream_t stream) {
    _op_ctx.stream = stream;
    checkCublas(cublasSetStream(_op_ctx.cublas_handle,
                                reinterpret_cast<cudaStream_t>(_op_ctx.stream)),
                "cublasSetStream");
}

Resource &getResource(int device_id) {
    // 每个线程独立一份资源池，避免跨线程共享 handle
    thread_local std::unordered_map<int, std::unique_ptr<Resource>> resources;

    auto it = resources.find(device_id);
    if (it != resources.end()) {
        return *(it->second);
    }

    // 基本合法性检查
    int ndev = 0;
    checkCuda(cudaGetDeviceCount(&ndev), "cudaGetDeviceCount");
    CHECK_ARGUMENT(device_id >= 0 && device_id < ndev, "invalid nvidia device id");

    // 懒创建
    auto res = std::make_unique<Resource>(device_id);
    Resource &ref = *res;
    resources.emplace(device_id, std::move(res));
    return ref;
}

} // namespace llaisys::device::nvidia
