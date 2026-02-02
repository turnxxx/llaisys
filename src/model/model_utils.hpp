#pragma once
/*
模型加载工具
*/
#include "../tensor/tensor.hpp"
#include "../weights/base_weights.hpp"
#include "llaisys.h"
#include <memory>
#include <string>
#include <unordered_map>
namespace llaisys::model {
typedef enum {
    LLAISYS_SILU = 0
} llaisysActivateFunctions;

// 先定义模型meta

struct meta_data {
    std::string architercutres;
    float attention_dropout;
    size_t bos_token_id;
    size_t eos_token_id;
    llaisys::model::llaisysActivateFunctions hidden_act;
    size_t hidden_size;
    float initializer_range;
    size_t intermediate_size;
    size_t max_position_embeddings;
    size_t max_window_layers;
    std::string model_type;
    size_t num_attention_heads;
    size_t num_hidden_layers;
    size_t num_key_value_heads;
    float rms_norm_eps;
    size_t rope_theta;
    size_t sliding_window;
    bool tie_word_embeddings;
    llaisysDataType_t torch_type;
    std::string transformers_version;
    bool use_cache;
    bool use_mrope;
    bool use_sliding_window;
    size_t vocab_size;
};
// 从config解析模型参数
class Model_Config {
private:
    llaisys::model::meta_data meta_data;

public:
    void read_from_config(const std::string &config_path);
    llaisys::model::meta_data get_meta_data() const;
};
// 从python接口获取权重储存到缓冲区的功能
struct Weight_buffer {
    std::unordered_map<std::string, Weights_t> weights_t_map_;
    void add(const std::string &name, const Weights_t &weights);
    void add_tensor(const std::string &name, const llaisys::tensor_t &tensor);
    bool has(const std::string &name) const;
    Weights_t get(const std::string &name) const;
    size_t size() const;
    void clear();
    std::unordered_map<std::string, Weights_t> &&move_map();
};

} // namespace llaisys::model