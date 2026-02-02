#include "model_utils.hpp"

#include <cctype>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
using json = nlohmann::json;
namespace llaisys::model {
llaisys::model::meta_data Model_Config::get_meta_data() const {
    return meta_data;
}
void Model_Config::read_from_config(const std::string &config_path)
// 解析config.json，设置模型超参
{
    // 从文件读取 JSON
    std::ifstream ifs(config_path);
    if (!ifs) {
        throw std::runtime_error("failed to open config file: " + config_path);
    }

    json config_json = json::parse(ifs);

    auto to_lower = [](std::string s) {
        for (auto &ch : s) {
            ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        }
        return s;
    };

    auto parse_hidden_act = [&](const std::string &act) {
        const auto lower = to_lower(act);
        if (lower == "silu" || lower == "swish") {
            return llaisys::model::LLAISYS_SILU;
        }
        throw std::runtime_error("unsupported hidden_act: " + act);
    };

    auto parse_torch_dtype = [&](const std::string &dtype) {
        const auto lower = to_lower(dtype);
        if (lower == "bfloat16" || lower == "bf16") {
            return LLAISYS_DTYPE_BF16;
        }
        if (lower == "float16" || lower == "fp16" || lower == "f16") {
            return LLAISYS_DTYPE_F16;
        }
        if (lower == "float32" || lower == "fp32" || lower == "f32") {
            return LLAISYS_DTYPE_F32;
        }
        if (lower == "float64" || lower == "fp64" || lower == "f64") {
            return LLAISYS_DTYPE_F64;
        }
        throw std::runtime_error("unsupported torch_dtype: " + dtype);
    };

    // architectures 可能是数组或字符串
    if (config_json.contains("architectures")) {
        const auto &arch = config_json.at("architectures");
        if (arch.is_array() && !arch.empty()) {
            meta_data.architercutres = arch.at(0).get<std::string>();
        } else {
            meta_data.architercutres = arch.get<std::string>();
        }
    }

    meta_data.attention_dropout = config_json.value("attention_dropout", 0.0f);
    meta_data.bos_token_id = config_json.at("bos_token_id").get<size_t>();
    meta_data.eos_token_id = config_json.at("eos_token_id").get<size_t>();
    meta_data.hidden_act = parse_hidden_act(config_json.at("hidden_act").get<std::string>());
    meta_data.hidden_size = config_json.at("hidden_size").get<size_t>();
    meta_data.initializer_range = config_json.value("initializer_range", 0.0f);
    meta_data.intermediate_size = config_json.at("intermediate_size").get<size_t>();
    meta_data.max_position_embeddings = config_json.at("max_position_embeddings").get<size_t>();
    meta_data.max_window_layers = config_json.value("max_window_layers", size_t{0});
    meta_data.model_type = config_json.at("model_type").get<std::string>();
    meta_data.num_attention_heads = config_json.at("num_attention_heads").get<size_t>();
    meta_data.num_hidden_layers = config_json.at("num_hidden_layers").get<size_t>();
    meta_data.num_key_value_heads = config_json.at("num_key_value_heads").get<size_t>();
    meta_data.rms_norm_eps = config_json.at("rms_norm_eps").get<float>();
    meta_data.rope_theta = config_json.value("rope_theta", size_t{0});
    meta_data.sliding_window = config_json.value("sliding_window", size_t{0});
    meta_data.tie_word_embeddings = config_json.value("tie_word_embeddings", false);
    meta_data.torch_type = parse_torch_dtype(config_json.at("torch_dtype").get<std::string>());
    meta_data.transformers_version = config_json.value("transformers_version", std::string{});
    meta_data.use_cache = config_json.value("use_cache", false);
    meta_data.use_mrope = config_json.value("use_mrope", false);
    meta_data.use_sliding_window = config_json.value("use_sliding_window", false);
    meta_data.vocab_size = config_json.at("vocab_size").get<size_t>();
}

void Weight_buffer::add(const std::string &name, const Weights_t &weights) {
    weights_t_map_[name] = weights;
}

void Weight_buffer::add_tensor(const std::string &name, const llaisys::tensor_t &tensor) {
    weights_t_map_[name] = std::make_shared<llaisys::Weights>(name, tensor);
}

bool Weight_buffer::has(const std::string &name) const {
    return weights_t_map_.find(name) != weights_t_map_.end();
}

Weights_t Weight_buffer::get(const std::string &name) const {
    auto it = weights_t_map_.find(name);
    return it == weights_t_map_.end() ? nullptr : it->second;
}

size_t Weight_buffer::size() const {
    return weights_t_map_.size();
}

void Weight_buffer::clear() {
    weights_t_map_.clear();
}

std::unordered_map<std::string, Weights_t> &&Weight_buffer::move_map() {
    return std::move(weights_t_map_);
}
} // namespace llaisys::model