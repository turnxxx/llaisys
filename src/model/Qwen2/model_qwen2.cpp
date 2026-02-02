#include "model_qwen2.hpp"
#include "../../ops/ops.hpp"
#include "../../utils.hpp"
#include "naive_session.hpp"
#include <cstring>
#include <string>

namespace llaisys::model {
// 加载权重，根据运行时的不同设备来把权重加载到不同文件上
void Model_Qwen2::loadWeights(WeightsMap &weights) {
    switch (this->_device.device_type) {
    case LLAISYS_DEVICE_CPU:
        // TODO: load weights for CPU
        // 根据并行类型选择权重加载
        // 不并行的情况
        if (this->_parallel.tensor_parallel == 0
            && this->_parallel.pipeline_parallel == 0
            && this->_parallel.data_parallel == 0) {
            this->weights_ = std::move(weights);
        } else {
            throw "Parallel is not supported yet";
        }
        break;
    case LLAISYS_DEVICE_NVIDIA:
        // TODO: load weights for NVIDIA
        // Comming Soon
        throw "Nvidia is not supported yet";
        break;
    default:
        break;
    }
    parseWeight();
    this->show();
    LOG_INFO("Model_Qwen2::loadWeights: complete");
}
model_t Model_Qwen2::create(WeightsMap &weights, const meta_data &meta_data,
                            const DeviceSpec &device, const ParallelSpec &parallel) {
    auto model = std::make_shared<Model_Qwen2>(meta_data, device, parallel);
    // 加载权重
    model->loadWeights(weights);
    model->bos_token_id = static_cast<int64_t>(meta_data.bos_token_id);
    model->eos_token_id = static_cast<int64_t>(meta_data.eos_token_id);
    LOG_INFO("Model_Qwen2::create: complete");
    return model;
}
void Model_Qwen2::unloadWeights() {
    this->weights_.clear();
}

std::unique_ptr<ModelSession> Model_Qwen2::createSession() {
    auto session = std::make_unique<naive_session>();
    std::vector<int64_t> tokens;
    session->init(_config, tokens);
    return session;
}

void Model_Qwen2::resetSession(ModelSession &session) {
    auto *naive = dynamic_cast<naive_session *>(&session);
    if (!naive) {
        return;
    }
    std::vector<int64_t> tokens;
    naive->init(_config, tokens);
}

void Model_Qwen2::destroy() {
    unloadWeights();
    qwen2_weights.embed_tokens.reset();
    qwen2_weights.final_norm.reset();
    qwen2_weights.lm_head.reset();
    qwen2_weights.layers.clear();
    bos_token_id = -1;
    eos_token_id = -1;
}

InferenceOutputs Model_Qwen2::inferStep(session_t session) {
    LOG_INFO("Model_Qwen2::inferStep:begin");
    LOG_INFO("Model_Qwen2::inferStep:session" << session->seq_len());
    ASSERT(session != nullptr, "Model_Qwen2::inferStep: session is null");
    ASSERT(session->kv_cache() != nullptr, "Model_Qwen2::inferStep: kv_cache is null");

    const auto &tokens = session->tokens();
    ASSERT(!tokens.empty(), "Model_Qwen2::inferStep: tokens is empty");
    ASSERT(qwen2_weights.embed_tokens != nullptr,
           "Model_Qwen2::inferStep: embed_tokens weight is null");
    ASSERT(qwen2_weights.final_norm != nullptr,
           "Model_Qwen2::inferStep: final_norm weight is null");
    ASSERT(qwen2_weights.lm_head != nullptr,
           "Model_Qwen2::inferStep: lm_head weight is null");
    ASSERT(qwen2_weights.layers.size() == _config.num_hidden_layers,
           "Model_Qwen2::inferStep: layers size mismatch");

    size_t token_pos = 0;
    tensor_t hidden_states;
    if (session->kv_cache()->seq_len() == 0) {
        // prefill: process full sequence
        token_pos = 0;
        hidden_states = inferInit(session);
    } else {
        // decode: only process last token
        token_pos = tokens.size() - 1;
        std::vector<int64_t> last_token{tokens.back()};
        std::vector<size_t> token_shape{1};
        tensor_t token_ids = Tensor::create(token_shape, LLAISYS_DTYPE_I64, _device.device_type);
        token_ids->load(last_token.data());
        std::vector<size_t> out_shape{1, _config.hidden_size};
        hidden_states = Tensor::create(out_shape, _config.torch_type, _device.device_type);
        ops::embedding(hidden_states, token_ids, qwen2_weights.embed_tokens->weights());
    }
    if (token_pos == 0) {
        LOG_INFO("Model::Qwen2:prefill: begin");
    }
    for (size_t i = 0; i < _config.num_hidden_layers; ++i) {
        const auto &layer = qwen2_weights.layers[i];
        ASSERT(layer.input_layernorm.weight != nullptr,
               "Model_Qwen2::inferStep: input_layernorm weight is null");
        ASSERT(layer.post_attention_layernorm.weight != nullptr,
               "Model_Qwen2::inferStep: post_attention_layernorm weight is null");
        ASSERT(layer.attention.q != nullptr && layer.attention.k != nullptr
                   && layer.attention.v != nullptr && layer.attention.o != nullptr,
               "Model_Qwen2::inferStep: attention weights are null");
        ASSERT(layer.attention.bias_q != nullptr && layer.attention.bias_k != nullptr
                   && layer.attention.bias_v != nullptr,
               "Model_Qwen2::inferStep: attention bias weights are null");
        ASSERT(layer.mlp.gate != nullptr && layer.mlp.up != nullptr && layer.mlp.down != nullptr,
               "Model_Qwen2::inferStep: mlp weights are null");
        hidden_states = llaisys::Qwen2::qwen2_decoder(
            hidden_states, qwen2_weights.layers[i], session->kv_cache(), _config, token_pos, i);
    }

    tensor_t normed = Tensor::create(hidden_states->shape(), _config.torch_type, _device.device_type);
    ops::rms_norm(normed, hidden_states, qwen2_weights.final_norm->weights(), _config.rms_norm_eps);

    std::vector<size_t> logits_shape{hidden_states->shape()[0], _config.vocab_size};
    tensor_t logits = Tensor::create(logits_shape, _config.torch_type, _device.device_type);
    ops::linear(logits, normed, qwen2_weights.lm_head->weights(), nullptr);

    const size_t seq_len = logits->shape()[0];
    tensor_t last_row = logits->slice(0, seq_len - 1, seq_len)->reshape({_config.vocab_size});
    tensor_t max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, _device.device_type);
    tensor_t max_val = Tensor::create({1}, _config.torch_type, _device.device_type);
    ops::argmax(max_idx, max_val, last_row);

    int64_t next_token = 0;
    std::memcpy(&next_token, max_idx->data(), sizeof(int64_t));

    session->append(next_token);

    InferenceOutputs outputs;
    outputs.next_token = next_token;
    outputs.logits = logits;
    return outputs;
}

std::vector<int64_t> Model_Qwen2::inferDialog(std::vector<int64_t> &tokens,
                                              size_t max_steps) {
    LOG_INFO("Model_Qwen2::inferDialog:begin");
    ASSERT(max_steps > 0, "Model_Qwen2::inferDialog: max_steps must be > 0");
    if (tokens.empty()) {
        tokens.push_back(bos_token_id);
    }
    auto session = naive_session::create(_config, tokens);
    for (size_t i = 0; i < max_steps; ++i) {
        auto outputs = inferStep(session);
        LOG_INFO("step=" << i
                         << " next=" << outputs.next_token
                         << " eos=" << eos_token_id);
        if (outputs.next_token == eos_token_id) {
            break;
        }
    }

    LOG_INFO("Model_Qwen2::inferDialog:end");
    return std::vector<int64_t>(session->tokens());
}

// input_embedding阶段
tensor_t Model_Qwen2::inferInit(session_t session) {
    ASSERT(session != nullptr, "Model_Qwen2::inferInit: session is null");
    const auto &tokens = session->tokens();
    ASSERT(!tokens.empty(), "Model_Qwen2::inferInit: tokens is empty");
    ASSERT(qwen2_weights.embed_tokens != nullptr,
           "Model_Qwen2::inferInit: embed_tokens weight is null");

    std::vector<size_t> token_shape{tokens.size()};
    tensor_t token_ids = Tensor::create(token_shape, LLAISYS_DTYPE_I64, _device.device_type);
    token_ids->load(tokens.data());

    std::vector<size_t> out_shape{tokens.size(), _config.hidden_size};
    tensor_t hidden_states = Tensor::create(out_shape, _config.torch_type, _device.device_type);
    ops::embedding(hidden_states, token_ids, qwen2_weights.embed_tokens->weights());
    LOG_TENSOR_DEBUG("Model_Qwen2::inferInit::hidden_states", hidden_states);
    LOG_INFO("Model_Qwen2::inferInit: complete");
    return hidden_states;
}
// 解析权重
void Model_Qwen2::parseWeight() {
    auto get_weight = [this](const std::string &name) -> Weights_t {
        auto it = weights_.find(name);
        ASSERT(it != weights_.end(), "Model_Qwen2::parseWeight: missing weight " + name);
        return it->second;
    };

    qwen2_weights.embed_tokens = get_weight("model.embed_tokens.weight");
    qwen2_weights.final_norm = get_weight("model.norm.weight");
    qwen2_weights.lm_head = get_weight("lm_head.weight");

    qwen2_weights.layers.clear();
    qwen2_weights.layers.resize(_config.num_hidden_layers);
    for (size_t i = 0; i < _config.num_hidden_layers; ++i) {
        auto &layer = qwen2_weights.layers[i];
        const std::string prefix = "model.layers." + std::to_string(i) + ".";

        layer.input_layernorm.weight = get_weight(prefix + "input_layernorm.weight");
        layer.post_attention_layernorm.weight = get_weight(prefix + "post_attention_layernorm.weight");

        layer.attention.q = get_weight(prefix + "self_attn.q_proj.weight");
        layer.attention.k = get_weight(prefix + "self_attn.k_proj.weight");
        layer.attention.v = get_weight(prefix + "self_attn.v_proj.weight");
        layer.attention.o = get_weight(prefix + "self_attn.o_proj.weight");
        layer.attention.bias_q = get_weight(prefix + "self_attn.q_proj.bias");
        layer.attention.bias_k = get_weight(prefix + "self_attn.k_proj.bias");
        layer.attention.bias_v = get_weight(prefix + "self_attn.v_proj.bias");

        layer.mlp.gate = get_weight(prefix + "mlp.gate_proj.weight");
        layer.mlp.up = get_weight(prefix + "mlp.up_proj.weight");
        layer.mlp.down = get_weight(prefix + "mlp.down_proj.weight");
    }
}
void Model_Qwen2::show() {
    LOG_INFO("Model_Qwen2::show: begin");
    LOG_INFO("  hidden_size=" << _config.hidden_size
                              << " nlayer=" << _config.num_hidden_layers
                              << " nh=" << _config.num_attention_heads
                              << " nkvh=" << _config.num_key_value_heads
                              << " dh=" << (_config.num_attention_heads > 0 ? (_config.hidden_size / _config.num_attention_heads) : 0)
                              << " di=" << _config.intermediate_size
                              << " maxseq=" << _config.max_position_embeddings
                              << " vocab=" << _config.vocab_size);
    LOG_INFO("  rms_eps=" << _config.rms_norm_eps
                          << " rope_theta=" << _config.rope_theta
                          << " use_cache=" << _config.use_cache
                          << " use_mrope=" << _config.use_mrope
                          << " use_sliding_window=" << _config.use_sliding_window
                          << " sliding_window=" << _config.sliding_window
                          << " max_window_layers=" << _config.max_window_layers
                          << " tie_word_embeddings=" << _config.tie_word_embeddings);
    LOG_INFO("  bos=" << _config.bos_token_id << " eos=" << _config.eos_token_id
                      << " dtype=" << static_cast<int>(_config.torch_type));
    LOG_INFO("  device_type=" << static_cast<int>(_device.device_type)
                              << " device_ids=" << _device.device_ids.size()
                              << " rank=" << _device.rank
                              << " world_size=" << _device.world_size);
    LOG_INFO("  parallel tp=" << _parallel.tensor_parallel
                              << " pp=" << _parallel.pipeline_parallel
                              << " dp=" << _parallel.data_parallel);
    LOG_INFO("  weights_map_size=" << weights_.size());
    LOG_INFO("  embed_tokens=" << (qwen2_weights.embed_tokens ? "ok" : "null")
                               << " final_norm=" << (qwen2_weights.final_norm ? "ok" : "null")
                               << " lm_head=" << (qwen2_weights.lm_head ? "ok" : "null"));

    size_t layers_missing = 0;
    size_t total_missing = 0;
    for (size_t i = 0; i < qwen2_weights.layers.size(); ++i) {
        const auto &layer = qwen2_weights.layers[i];
        size_t missing = 0;
        if (!layer.input_layernorm.weight) {
            missing++;
        }
        if (!layer.post_attention_layernorm.weight) {
            missing++;
        }
        if (!layer.attention.q) {
            missing++;
        }
        if (!layer.attention.k) {
            missing++;
        }
        if (!layer.attention.v) {
            missing++;
        }
        if (!layer.attention.o) {
            missing++;
        }
        if (!layer.attention.bias_q) {
            missing++;
        }
        if (!layer.attention.bias_k) {
            missing++;
        }
        if (!layer.attention.bias_v) {
            missing++;
        }
        if (!layer.mlp.gate) {
            missing++;
        }
        if (!layer.mlp.up) {
            missing++;
        }
        if (!layer.mlp.down) {
            missing++;
        }
        if (missing > 0) {
            layers_missing++;
            total_missing += missing;
        }
    }
    LOG_INFO("  layers=" << qwen2_weights.layers.size()
                         << " layers_missing=" << layers_missing
                         << " total_missing=" << total_missing);
    LOG_INFO("Model_Qwen2::show: end");
}
} // namespace llaisys::model