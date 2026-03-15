#include "linear_nvidia.cuh"

#include "../../../core/llaisys_core.hpp"
#include "../../../device/nvidia/nvidia_resource.cuh"
#include "../../../utils.hpp"

#include <cublasLt.h>
#include <cuda_runtime.h>

namespace llaisys::ops::nvidia {

namespace {
void linear_cublaslt_impl(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias,
                          cudaDataType_t data_type, cublasComputeType_t compute_type) {
    auto &runtime = llaisys::core::context().runtime();
    auto &res = llaisys::device::nvidia::getResource(out->deviceId());
    res.setStream(runtime.stream());
    auto stream = reinterpret_cast<cudaStream_t>(runtime.stream());
    cublasLtHandle_t lt_handle = res.opContext().cublaslt_handle;
    ASSERT(lt_handle != nullptr, "Linear(NVIDIA): cublasLt handle is null");

    const int m = static_cast<int>(in->shape()[0]);
    const int k = static_cast<int>(in->shape()[1]);
    const int n = static_cast<int>(weight->shape()[0]);

    cublasOperation_t op_a = CUBLAS_OP_T; // weight row-major [n, k] -> op(A) gives [n, k] in col-major view
    cublasOperation_t op_b = CUBLAS_OP_N; // in row-major [m, k] -> [k, m] in col-major view

    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasStatus_t status = cublasLtMatmulDescCreate(&op_desc, compute_type, CUDA_R_32F);
    ASSERT(status == CUBLAS_STATUS_SUCCESS, "Linear(NVIDIA): cublasLtMatmulDescCreate failed");
    status = cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(op_a));
    ASSERT(status == CUBLAS_STATUS_SUCCESS, "Linear(NVIDIA): set TRANSA failed");
    status = cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(op_b));
    ASSERT(status == CUBLAS_STATUS_SUCCESS, "Linear(NVIDIA): set TRANSB failed");

    if (bias != nullptr) {
        cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
        status = cublasLtMatmulDescSetAttribute(
            op_desc,
            CUBLASLT_MATMUL_DESC_EPILOGUE,
            &epilogue,
            sizeof(epilogue));
        ASSERT(status == CUBLAS_STATUS_SUCCESS, "Linear(NVIDIA): set EPILOGUE_BIAS failed");
        void *bias_ptr = bias->data();
        status = cublasLtMatmulDescSetAttribute(
            op_desc,
            CUBLASLT_MATMUL_DESC_BIAS_POINTER,
            &bias_ptr,
            sizeof(bias_ptr));
        ASSERT(status == CUBLAS_STATUS_SUCCESS, "Linear(NVIDIA): set bias pointer failed");
    }

    cublasLtMatrixLayout_t a_desc = nullptr;
    cublasLtMatrixLayout_t b_desc = nullptr;
    cublasLtMatrixLayout_t c_desc = nullptr;
    status = cublasLtMatrixLayoutCreate(&a_desc, data_type, k, n, k); // weight pointer as col-major [k, n]
    ASSERT(status == CUBLAS_STATUS_SUCCESS, "Linear(NVIDIA): create A layout failed");
    status = cublasLtMatrixLayoutCreate(&b_desc, data_type, k, m, k); // in pointer as col-major [k, m]
    ASSERT(status == CUBLAS_STATUS_SUCCESS, "Linear(NVIDIA): create B layout failed");
    status = cublasLtMatrixLayoutCreate(&c_desc, data_type, n, m, n); // out pointer as col-major [n, m]
    ASSERT(status == CUBLAS_STATUS_SUCCESS, "Linear(NVIDIA): create C layout failed");

    cublasLtMatmulPreference_t pref = nullptr;
    status = cublasLtMatmulPreferenceCreate(&pref);
    ASSERT(status == CUBLAS_STATUS_SUCCESS, "Linear(NVIDIA): create matmul preference failed");
    size_t workspace_size = 0;
    status = cublasLtMatmulPreferenceSetAttribute(
        pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_size,
        sizeof(workspace_size));
    ASSERT(status == CUBLAS_STATUS_SUCCESS, "Linear(NVIDIA): set workspace preference failed");

    cublasLtMatmulHeuristicResult_t heuristic_result{};
    int returned_results = 0;
    status = cublasLtMatmulAlgoGetHeuristic(
        lt_handle,
        op_desc,
        a_desc,
        b_desc,
        c_desc,
        c_desc,
        pref,
        1,
        &heuristic_result,
        &returned_results);
    ASSERT(status == CUBLAS_STATUS_SUCCESS && returned_results > 0,
           "Linear(NVIDIA): no cublasLt matmul heuristic available");

    const float alpha = 1.0f;
    const float beta = 0.0f;
    status = cublasLtMatmul(
        lt_handle,
        op_desc,
        &alpha,
        weight->data(),
        a_desc,
        in->data(),
        b_desc,
        &beta,
        out->data(),
        c_desc,
        out->data(),
        c_desc,
        &heuristic_result.algo,
        nullptr,
        0,
        stream);
    ASSERT(status == CUBLAS_STATUS_SUCCESS, "Linear(NVIDIA): cublasLtMatmul failed");

    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(c_desc);
    cublasLtMatrixLayoutDestroy(b_desc);
    cublasLtMatrixLayoutDestroy(a_desc);
    cublasLtMatmulDescDestroy(op_desc);
}
} // namespace

void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    llaisys::core::context().setDevice(LLAISYS_DEVICE_NVIDIA, out->deviceId());
    auto &runtime = llaisys::core::context().runtime();
    if (bias != nullptr) {
        ASSERT(bias->numel() == out->shape()[1],
               "Linear(NVIDIA): bias numel must equal out.shape[1]");
    }
    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
        linear_cublaslt_impl(out, in, weight, bias, CUDA_R_32F, CUBLAS_COMPUTE_32F);
        break;
    case LLAISYS_DTYPE_F16:
        linear_cublaslt_impl(out, in, weight, bias, CUDA_R_16F, CUBLAS_COMPUTE_32F);
        break;
    case LLAISYS_DTYPE_BF16:
        linear_cublaslt_impl(out, in, weight, bias, CUDA_R_16BF, CUBLAS_COMPUTE_32F);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
    runtime.api()->stream_synchronize(runtime.stream());
}
} // namespace llaisys::ops::nvidia
