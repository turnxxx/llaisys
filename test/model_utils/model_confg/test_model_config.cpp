#include "src/model/model_utils.hpp"

#include <iostream>
#include <string>

namespace {
#ifndef LLAISYS_PROJECT_DIR
std::string dir_of_file(const char *path) {
    std::string p(path);
    const auto pos = p.find_last_of("/\\");
    if (pos == std::string::npos) {
        return ".";
    }
    return p.substr(0, pos);
}
#endif
} // namespace

int main() {
    std::string config_path;
#ifdef LLAISYS_PROJECT_DIR
    config_path = std::string(LLAISYS_PROJECT_DIR);
    if (!config_path.empty() && config_path.back() != '/') {
        config_path += "/";
    }
    config_path += "test/model_utils/model_confg/test_config.json";
#else
    config_path = dir_of_file(__FILE__) + "/test_config.json";
#endif

    llaisys::model::Model_Config cfg;
    cfg.read_from_config(config_path);
    const auto meta = cfg.get_meta_data();

    std::cout << "architercutres: " << meta.architercutres << "\n";
    std::cout << "attention_dropout: " << meta.attention_dropout << "\n";
    std::cout << "bos_token_id: " << meta.bos_token_id << "\n";
    std::cout << "eos_token_id: " << meta.eos_token_id << "\n";
    std::cout << "hidden_act: " << static_cast<int>(meta.hidden_act) << "\n";
    std::cout << "hidden_size: " << meta.hidden_size << "\n";
    std::cout << "initializer_range: " << meta.initializer_range << "\n";
    std::cout << "intermediate_size: " << meta.intermediate_size << "\n";
    std::cout << "max_position_embeddings: " << meta.max_position_embeddings << "\n";
    std::cout << "max_window_layers: " << meta.max_window_layers << "\n";
    std::cout << "model_type: " << meta.model_type << "\n";
    std::cout << "num_attention_heads: " << meta.num_attention_heads << "\n";
    std::cout << "num_hidden_layers: " << meta.num_hidden_layers << "\n";
    std::cout << "num_key_value_heads: " << meta.num_key_value_heads << "\n";
    std::cout << "rms_norm_eps: " << meta.rms_norm_eps << "\n";
    std::cout << "rope_theta: " << meta.rope_theta << "\n";
    std::cout << "sliding_window: " << meta.sliding_window << "\n";
    std::cout << "tie_word_embeddings: " << std::boolalpha << meta.tie_word_embeddings << "\n";
    std::cout << "torch_type: " << static_cast<int>(meta.torch_type) << "\n";
    std::cout << "transformers_version: " << meta.transformers_version << "\n";
    std::cout << "use_cache: " << std::boolalpha << meta.use_cache << "\n";
    std::cout << "use_mrope: " << std::boolalpha << meta.use_mrope << "\n";
    std::cout << "use_sliding_window: " << std::boolalpha << meta.use_sliding_window << "\n";
    std::cout << "vocab_size: " << meta.vocab_size << "\n";

    return 0;
}
