#include "model_utils.hpp"

#include <cctype>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>

namespace {
std::string read_all(const std::string &path) {
    std::ifstream ifs(path);
    if (!ifs) {
        throw std::runtime_error("failed to open config file: " + path);
    }
    std::stringstream buffer;
    buffer << ifs.rdbuf();
    return buffer.str();
}

size_t skip_ws(const std::string &s, size_t i) {
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) {
        ++i;
    }
    return i;
}

size_t find_key(const std::string &s, const std::string &key) {
    const std::string pattern = "\"" + key + "\"";
    return s.find(pattern);
}

std::string parse_string(const std::string &s, size_t &i) {
    i = skip_ws(s, i);
    if (i >= s.size() || s[i] != '"') {
        throw std::runtime_error("expected string");
    }
    ++i;
    std::string out;
    while (i < s.size()) {
        char c = s[i++];
        if (c == '\\') {
            if (i < s.size()) {
                out.push_back(s[i++]);
            }
        } else if (c == '"') {
            break;
        } else {
            out.push_back(c);
        }
    }
    return out;
}

double parse_number(const std::string &s, size_t &i) {
    i = skip_ws(s, i);
    size_t start = i;
    if (i < s.size() && (s[i] == '-' || s[i] == '+')) {
        ++i;
    }
    while (i < s.size() && (std::isdigit(static_cast<unsigned char>(s[i])) || s[i] == '.' || s[i] == 'e' || s[i] == 'E' || s[i] == '+' || s[i] == '-')) {
        ++i;
    }
    return std::stod(s.substr(start, i - start));
}

bool parse_bool(const std::string &s, size_t &i) {
    i = skip_ws(s, i);
    if (s.compare(i, 4, "true") == 0) {
        i += 4;
        return true;
    }
    if (s.compare(i, 5, "false") == 0) {
        i += 5;
        return false;
    }
    throw std::runtime_error("expected bool");
}

std::optional<size_t> find_value_pos(const std::string &s, const std::string &key) {
    size_t pos = find_key(s, key);
    if (pos == std::string::npos) {
        return std::nullopt;
    }
    pos = s.find(':', pos);
    if (pos == std::string::npos) {
        throw std::runtime_error("missing ':' for key: " + key);
    }
    return pos + 1;
}

std::string get_required_string(const std::string &s, const std::string &key) {
    auto pos = find_value_pos(s, key);
    if (!pos) {
        throw std::runtime_error("missing key: " + key);
    }
    size_t i = *pos;
    return parse_string(s, i);
}

std::optional<std::string> get_optional_string(const std::string &s, const std::string &key) {
    auto pos = find_value_pos(s, key);
    if (!pos) {
        return std::nullopt;
    }
    size_t i = *pos;
    return parse_string(s, i);
}

size_t get_required_size_t(const std::string &s, const std::string &key) {
    auto pos = find_value_pos(s, key);
    if (!pos) {
        throw std::runtime_error("missing key: " + key);
    }
    size_t i = *pos;
    return static_cast<size_t>(parse_number(s, i));
}

float get_required_float(const std::string &s, const std::string &key) {
    auto pos = find_value_pos(s, key);
    if (!pos) {
        throw std::runtime_error("missing key: " + key);
    }
    size_t i = *pos;
    return static_cast<float>(parse_number(s, i));
}

float get_optional_float(const std::string &s, const std::string &key, float def) {
    auto pos = find_value_pos(s, key);
    if (!pos) {
        return def;
    }
    size_t i = *pos;
    return static_cast<float>(parse_number(s, i));
}

size_t get_optional_size_t(const std::string &s, const std::string &key, size_t def) {
    auto pos = find_value_pos(s, key);
    if (!pos) {
        return def;
    }
    size_t i = *pos;
    return static_cast<size_t>(parse_number(s, i));
}

bool get_optional_bool(const std::string &s, const std::string &key, bool def) {
    auto pos = find_value_pos(s, key);
    if (!pos) {
        return def;
    }
    size_t i = *pos;
    return parse_bool(s, i);
}

std::optional<std::string> get_architecture(const std::string &s) {
    auto pos = find_value_pos(s, "architectures");
    if (!pos) {
        return std::nullopt;
    }
    size_t i = skip_ws(s, *pos);
    if (i >= s.size()) {
        return std::nullopt;
    }
    if (s[i] == '[') {
        ++i;
        i = skip_ws(s, i);
        if (i < s.size() && s[i] == ']') {
            return std::nullopt;
        }
        return parse_string(s, i);
    }
    if (s[i] == '"') {
        return parse_string(s, i);
    }
    return std::nullopt;
}
} // namespace
namespace llaisys::model {
llaisys::model::meta_data Model_Config::get_meta_data() const {
    return meta_data;
}
void Model_Config::read_from_config(const std::string &config_path)
// 解析config.json，设置模型超参
{
    const std::string config_json = read_all(config_path);

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

    if (const auto arch = get_architecture(config_json)) {
        meta_data.architercutres = *arch;
    }

    meta_data.attention_dropout = get_optional_float(config_json, "attention_dropout", 0.0f);
    meta_data.bos_token_id = get_required_size_t(config_json, "bos_token_id");
    meta_data.eos_token_id = get_required_size_t(config_json, "eos_token_id");
    meta_data.hidden_act = parse_hidden_act(get_required_string(config_json, "hidden_act"));
    meta_data.hidden_size = get_required_size_t(config_json, "hidden_size");
    meta_data.initializer_range = get_optional_float(config_json, "initializer_range", 0.0f);
    meta_data.intermediate_size = get_required_size_t(config_json, "intermediate_size");
    meta_data.max_position_embeddings = get_required_size_t(config_json, "max_position_embeddings");
    meta_data.max_window_layers = get_optional_size_t(config_json, "max_window_layers", size_t{0});
    meta_data.model_type = get_required_string(config_json, "model_type");
    meta_data.num_attention_heads = get_required_size_t(config_json, "num_attention_heads");
    meta_data.num_hidden_layers = get_required_size_t(config_json, "num_hidden_layers");
    meta_data.num_key_value_heads = get_required_size_t(config_json, "num_key_value_heads");
    meta_data.rms_norm_eps = get_required_float(config_json, "rms_norm_eps");
    meta_data.rope_theta = get_optional_size_t(config_json, "rope_theta", size_t{0});
    meta_data.sliding_window = get_optional_size_t(config_json, "sliding_window", size_t{0});
    meta_data.tie_word_embeddings = get_optional_bool(config_json, "tie_word_embeddings", false);
    meta_data.torch_type = parse_torch_dtype(get_required_string(config_json, "torch_dtype"));
    if (const auto ver = get_optional_string(config_json, "transformers_version")) {
        meta_data.transformers_version = *ver;
    }
    meta_data.use_cache = get_optional_bool(config_json, "use_cache", false);
    meta_data.use_mrope = get_optional_bool(config_json, "use_mrope", false);
    meta_data.use_sliding_window = get_optional_bool(config_json, "use_sliding_window", false);
    meta_data.vocab_size = get_required_size_t(config_json, "vocab_size");
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