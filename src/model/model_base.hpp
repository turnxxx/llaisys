/*
模型推理后端的通用基类接口。
职责：定义模型生命周期、设备/并行配置、权重加载、会话与推理入口。
*/
#pragma once

#include "../KVcache/KVcacheBase.hpp"
#include "../KVcache/NaiveCache/NaiveCache.hpp"
#include "../tensor/tensor.hpp"
#include "../weights/base_weights.hpp"
#include "llaisys.h"
#include "model_utils.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
namespace llaisys::model {
class ModelBase;
using llaisys::KVcache::KVcache_t;
using model_t = std::shared_ptr<ModelBase>;
// 设备规格：目标设备类型与设备列表，预留多设备/分布式字段
struct DeviceSpec {
    llaisysDeviceType_t device_type = LLAISYS_DEVICE_CPU;
    std::vector<int> device_ids;
    int rank = 0;
    int world_size = 1;
};
// 并行规格：预留张量/流水/数据并行配置
struct ParallelSpec {
    size_t tensor_parallel = 0;
    size_t pipeline_parallel = 0;
    size_t data_parallel = 0;
};
// 一次推理请求输出：next_token 是本次生成的 token_id
struct InferenceOutputs {
    int64_t next_token = -1;
    tensor_t logits;
};

// 权重映射：键为权重指针
using WeightsMap = std::unordered_map<std::string, Weights_t>;

// 推理会话：保存 KV-cache 等与会话相关的状态

class ModelSession {
public:
    virtual ~ModelSession() = default;
    virtual const std::vector<int64_t> &tokens() const = 0;
    virtual void append(int64_t next_token) = 0;
    virtual size_t seq_len() const = 0;
    virtual size_t token_pos() const = 0;
    virtual KVcache_t kv_cache() const = 0;
};
using session_t = std::shared_ptr<ModelSession>;
// 模型基类：定义通用接口，具体模型需继承实现
class ModelBase {
public:
    virtual ~ModelBase() = default;

    ModelBase(const ModelBase &) = delete;
    ModelBase &operator=(const ModelBase &) = delete;
    ModelBase(ModelBase &&) = delete;
    ModelBase &operator=(ModelBase &&) = delete;

    // 只读访问模型配置与运行参数
    const meta_data &config() const { return _config; }
    const DeviceSpec &deviceSpec() const { return _device; }
    const ParallelSpec &parallelSpec() const { return _parallel; }

    // 运行参数设置（允许在创建后调整）
    void setDeviceSpec(const DeviceSpec &device);
    void setParallelSpec(const ParallelSpec &parallel);

    // 权重加载/卸载（建议在构造后调用）
    virtual void loadWeights(WeightsMap &weights) = 0;
    virtual void unloadWeights() = 0;

    // 会话管理（用于多请求或 KV-cache 管理）
    virtual std::unique_ptr<ModelSession> createSession() = 0;
    virtual void resetSession(ModelSession &session) = 0;

    // 推理入口：单轮推理（生成一个 token）
    virtual InferenceOutputs inferStep(session_t session) = 0;
    // 调试功能，打印模型信息
    virtual void show() = 0;

protected:
    // 受保护构造：仅供派生类调用
    explicit ModelBase(const meta_data &config, const DeviceSpec &device, const ParallelSpec &parallel)
        : _config(config), _device(device), _parallel(parallel) {}

    // 基础配置与运行参数
    meta_data _config;
    DeviceSpec _device;
    ParallelSpec _parallel;
};

} // namespace llaisys::model
