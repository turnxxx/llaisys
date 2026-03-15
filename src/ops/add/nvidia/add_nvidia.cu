#include "add_nvidia.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../../../core/llaisys_core.hpp"
#include "../../../utils.hpp"

namespace llaisys::ops::nvidia {
template <typename T>
__global__ void add_kernel(T *c, const T *a, const T *b, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void add_kernel_fp16(llaisys::fp16_t *c,
                                const llaisys::fp16_t *a,
                                const llaisys::fp16_t *b,
                                size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        __half ha = __ushort_as_half(a[idx]._v);
        __half hb = __ushort_as_half(b[idx]._v);
        __half hc = __float2half(__half2float(ha) + __half2float(hb));
        c[idx]._v = __half_as_ushort(hc);
    }
}

__global__ void add_kernel_bf16(llaisys::bf16_t *c,
                                const llaisys::bf16_t *a,
                                const llaisys::bf16_t *b,
                                size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        __nv_bfloat16 ba = __ushort_as_bfloat16(a[idx]._v);
        __nv_bfloat16 bb = __ushort_as_bfloat16(b[idx]._v);
        __nv_bfloat16 bc = __float2bfloat16(__bfloat162float(ba) + __bfloat162float(bb));
        c[idx]._v = __bfloat16_as_ushort(bc);
    }
}

template <typename T>
void add_(tensor_t c, tensor_t a, tensor_t b) {
    constexpr int block_size = 256;
    int num_blocks = static_cast<int>((c->numel() + block_size - 1) / block_size);
    auto *c_ptr = reinterpret_cast<T *>(c->data());
    auto *a_ptr = reinterpret_cast<const T *>(a->data());
    auto *b_ptr = reinterpret_cast<const T *>(b->data());
    auto &runtime = llaisys::core::context().runtime();
    auto stream = reinterpret_cast<cudaStream_t>(runtime.stream());
    add_kernel<T><<<num_blocks, block_size, 0, stream>>>(c_ptr, a_ptr, b_ptr, c->numel());
    runtime.api()->stream_synchronize(runtime.stream());
}

void add_fp16(tensor_t c, tensor_t a, tensor_t b) {
    constexpr int block_size = 256;
    int num_blocks = static_cast<int>((c->numel() + block_size - 1) / block_size);
    auto *c_ptr = reinterpret_cast<llaisys::fp16_t *>(c->data());
    auto *a_ptr = reinterpret_cast<const llaisys::fp16_t *>(a->data());
    auto *b_ptr = reinterpret_cast<const llaisys::fp16_t *>(b->data());
    auto &runtime = llaisys::core::context().runtime();
    auto stream = reinterpret_cast<cudaStream_t>(runtime.stream());
    add_kernel_fp16<<<num_blocks, block_size, 0, stream>>>(c_ptr, a_ptr, b_ptr, c->numel());
    runtime.api()->stream_synchronize(runtime.stream());
}

void add_bf16(tensor_t c, tensor_t a, tensor_t b) {
    constexpr int block_size = 256;
    int num_blocks = static_cast<int>((c->numel() + block_size - 1) / block_size);
    auto *c_ptr = reinterpret_cast<llaisys::bf16_t *>(c->data());
    auto *a_ptr = reinterpret_cast<const llaisys::bf16_t *>(a->data());
    auto *b_ptr = reinterpret_cast<const llaisys::bf16_t *>(b->data());
    auto &runtime = llaisys::core::context().runtime();
    auto stream = reinterpret_cast<cudaStream_t>(runtime.stream());
    add_kernel_bf16<<<num_blocks, block_size, 0, stream>>>(c_ptr, a_ptr, b_ptr, c->numel());
    runtime.api()->stream_synchronize(runtime.stream());
}
void add_cublas_fp16(tensor_t c, tensor_t a, tensor_t b) {
    CHECK_SAME_DEVICE(c, a, b);
    auto &runtime = llaisys::core::context().runtime();
    auto stream = runtime.stream();
    auto &res = llaisys::device::nvidia::getResource(c->deviceId());
    res.setStream(stream);
    cublasHandle_t handle = res.opContext().cublas_handle;
    cublasStatus_t status = cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    ASSERT(status == CUBLAS_STATUS_SUCCESS, "add_cublas_fp16: cublasSetPointerMode failed");

    const int n = static_cast<int>(c->numel());
    const __half *a_ptr = reinterpret_cast<const __half *>(a->data());
    const __half *b_ptr = reinterpret_cast<const __half *>(b->data());
    __half *c_ptr = reinterpret_cast<__half *>(c->data());
    const float alpha = 1.0f;

    cudaStream_t cu_stream;
    cublasGetStream(handle, &cu_stream);

    cudaMemcpyAsync(c_ptr, b_ptr, c->numel() * c->elementSize(),
                    cudaMemcpyDeviceToDevice, cu_stream);

    status = cublasAxpyEx(handle, n, &alpha, CUDA_R_32F,
                          a_ptr, CUDA_R_16F, 1,
                          c_ptr, CUDA_R_16F, 1,
                          CUDA_R_32F);
    ASSERT(status == CUBLAS_STATUS_SUCCESS, "add_cublas_fp16: cublasAxpyEx failed");
    auto api = runtime.api();
    api->stream_synchronize(stream);
}
void add_cublas_fp32(tensor_t c, tensor_t a, tensor_t b) {
    CHECK_SAME_DEVICE(c, a, b);
    auto &runtime = llaisys::core::context().runtime();
    auto stream = runtime.stream();
    auto &res = llaisys::device::nvidia::getResource(c->deviceId());
    res.setStream(stream);
    cublasHandle_t handle = res.opContext().cublas_handle;
    cublasStatus_t status = cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    ASSERT(status == CUBLAS_STATUS_SUCCESS, "add_cublas_fp32: cublasSetPointerMode failed");

    const int n = static_cast<int>(c->numel());
    const float *a_ptr = reinterpret_cast<const float *>(a->data());
    const float *b_ptr = reinterpret_cast<const float *>(b->data());
    float *c_ptr = reinterpret_cast<float *>(c->data());
    const float alpha = 1.0f;

    cudaStream_t cu_stream;
    cublasGetStream(handle, &cu_stream);

    cudaMemcpyAsync(c_ptr, b_ptr, c->numel() * c->elementSize(),
                    cudaMemcpyDeviceToDevice, cu_stream);

    status = cublasAxpyEx(handle, n, &alpha, CUDA_R_32F,
                          a_ptr, CUDA_R_32F, 1,
                          c_ptr, CUDA_R_32F, 1,
                          CUDA_R_32F);
    ASSERT(status == CUBLAS_STATUS_SUCCESS, "add_cublas_fp32: cublasAxpyEx failed");
    auto api = runtime.api();
    api->stream_synchronize(stream);
}
void add_cublas_bf16(tensor_t c, tensor_t a, tensor_t b) {
    CHECK_SAME_DEVICE(c, a, b);
    auto &runtime = llaisys::core::context().runtime();
    auto stream = runtime.stream();
    auto &res = llaisys::device::nvidia::getResource(c->deviceId());
    res.setStream(stream);
    cublasHandle_t handle = res.opContext().cublas_handle;
    cublasStatus_t status = cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    ASSERT(status == CUBLAS_STATUS_SUCCESS, "add_cublas_bf16: cublasSetPointerMode failed");

    const int n = static_cast<int>(c->numel());
    const __nv_bfloat16 *a_ptr = reinterpret_cast<const __nv_bfloat16 *>(a->data());
    const __nv_bfloat16 *b_ptr = reinterpret_cast<const __nv_bfloat16 *>(b->data());
    __nv_bfloat16 *c_ptr = reinterpret_cast<__nv_bfloat16 *>(c->data());
    const float alpha = 1.0f;

    cudaStream_t cu_stream;
    cublasGetStream(handle, &cu_stream);

    cudaMemcpyAsync(c_ptr, b_ptr, c->numel() * c->elementSize(),
                    cudaMemcpyDeviceToDevice, cu_stream);

    status = cublasAxpyEx(handle, n, &alpha, CUDA_R_32F,
                          a_ptr, CUDA_R_16BF, 1,
                          c_ptr, CUDA_R_16BF, 1,
                          CUDA_R_32F);
    ASSERT(status == CUBLAS_STATUS_SUCCESS, "add_cublas_bf16: cublasAxpyEx failed");
    auto api = runtime.api();
    api->stream_synchronize(stream);
}
// 不拷贝的版本：直接在 b 上做 b = alpha*a + b，然后 swap c 和 b 的存储
// 调用后 b 的原始数据被破坏，c 持有结果
void add_cublas_bf16_nocpy(tensor_t c, tensor_t a, tensor_t b) {
    CHECK_SAME_DEVICE(c, a, b);
    auto &runtime = llaisys::core::context().runtime();
    auto stream = runtime.stream();
    auto &res = llaisys::device::nvidia::getResource(c->deviceId());
    res.setStream(stream);
    cublasHandle_t handle = res.opContext().cublas_handle;
    cublasStatus_t status = cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    ASSERT(status == CUBLAS_STATUS_SUCCESS, "add_cublas_bf16_nocpy: cublasSetPointerMode failed");

    const int n = static_cast<int>(b->numel());
    const __nv_bfloat16 *a_ptr = reinterpret_cast<const __nv_bfloat16 *>(a->data());
    __nv_bfloat16 *b_ptr = reinterpret_cast<__nv_bfloat16 *>(b->data());
    const float alpha = 1.0f;

    status = cublasAxpyEx(handle, n, &alpha, CUDA_R_32F,
                          a_ptr, CUDA_R_16BF, 1,
                          b_ptr, CUDA_R_16BF, 1,
                          CUDA_R_32F);
    ASSERT(status == CUBLAS_STATUS_SUCCESS, "add_cublas_bf16_nocpy: cublasAxpyEx failed");

    auto api = runtime.api();
    api->stream_synchronize(stream);

    c->swapStorage(*b);
}
void add_cublas_fp32_nocpy(tensor_t c, tensor_t a, tensor_t b) {
    CHECK_SAME_DEVICE(c, a, b);
    auto &runtime = llaisys::core::context().runtime();
    auto stream = runtime.stream();
    auto &res = llaisys::device::nvidia::getResource(c->deviceId());
    res.setStream(stream);
    cublasHandle_t handle = res.opContext().cublas_handle;
    cublasStatus_t status = cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    ASSERT(status == CUBLAS_STATUS_SUCCESS, "add_cublas_bf16_nocpy: cublasSetPointerMode failed");

    const int n = static_cast<int>(b->numel());
    const float *a_ptr = reinterpret_cast<const float *>(a->data());
    float *b_ptr = reinterpret_cast<float *>(b->data());
    const float alpha = 1.0f;

    status = cublasAxpyEx(handle, n, &alpha, CUDA_R_32F,
                          a_ptr, CUDA_R_32F, 1,
                          b_ptr, CUDA_R_32F, 1,
                          CUDA_R_32F);
    ASSERT(status == CUBLAS_STATUS_SUCCESS, "add_cublas_bf16_nocpy: cublasAxpyEx failed");

    auto api = runtime.api();
    api->stream_synchronize(stream);

    c->swapStorage(*b);
}
void add_cublas_fp16_nocpy(tensor_t c, tensor_t a, tensor_t b) {
    CHECK_SAME_DEVICE(c, a, b);
    auto &runtime = llaisys::core::context().runtime();
    auto stream = runtime.stream();
    auto &res = llaisys::device::nvidia::getResource(c->deviceId());
    res.setStream(stream);
    cublasHandle_t handle = res.opContext().cublas_handle;
    cublasStatus_t status = cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    ASSERT(status == CUBLAS_STATUS_SUCCESS, "add_cublas_fp16_nocpy: cublasSetPointerMode failed");

    const int n = static_cast<int>(b->numel());
    const __half *a_ptr = reinterpret_cast<const __half *>(a->data());
    __half *b_ptr = reinterpret_cast<__half *>(b->data());
    const float alpha = 1.0f;

    status = cublasAxpyEx(handle, n, &alpha, CUDA_R_32F,
                          a_ptr, CUDA_R_16F, 1,
                          b_ptr, CUDA_R_16F, 1,
                          CUDA_R_32F);
    ASSERT(status == CUBLAS_STATUS_SUCCESS, "add_cublas_fp16_nocpy: cublasAxpyEx failed");

    auto api = runtime.api();
    api->stream_synchronize(stream);

    c->swapStorage(*b);
}
// 模板分发接口
void add(tensor_t c, tensor_t a, tensor_t b) {
    llaisys::core::context().setDevice(LLAISYS_DEVICE_NVIDIA, c->deviceId());
    switch (c->dtype()) {
    case LLAISYS_DTYPE_F32:
        return add_cublas_fp32_nocpy(c, a, b);
    case LLAISYS_DTYPE_BF16:
        return add_cublas_bf16_nocpy(c, a, b);
    case LLAISYS_DTYPE_F16:
        return add_cublas_fp16_nocpy(c, a, b);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(c->dtype());
    }
}
} // namespace llaisys::ops::nvidia
