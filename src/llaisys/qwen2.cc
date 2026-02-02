
#include "llaisys/models/qwen2.h"
#include "../model/Qwen2/model_qwen2.hpp"
#include "../model/Qwen2/naive_session.hpp"
#include "../model/model_utils.hpp"
#include "llaisys_tensor.hpp"
#include <cstring>
#include <vector>

extern "C" {
// implement C API wrappers for Qwen2
struct LlaisysQwen2Model {
    llaisys::model::model_t qwen2_model;
    size_t nlayer = 0;
    bool weights_ready = false;
    LlaisysQwen2Weights c_weights{};
    std::vector<llaisysTensor_t> attn_norm_w;
    std::vector<llaisysTensor_t> attn_q_w;
    std::vector<llaisysTensor_t> attn_q_b;
    std::vector<llaisysTensor_t> attn_k_w;
    std::vector<llaisysTensor_t> attn_k_b;
    std::vector<llaisysTensor_t> attn_v_w;
    std::vector<llaisysTensor_t> attn_v_b;
    std::vector<llaisysTensor_t> attn_o_w;
    std::vector<llaisysTensor_t> mlp_norm_w;
    std::vector<llaisysTensor_t> mlp_gate_w;
    std::vector<llaisysTensor_t> mlp_up_w;
    std::vector<llaisysTensor_t> mlp_down_w;
};

struct LlaisysWeightBuffer {
    llaisys::model::Weight_buffer buffer;
};

__export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(
    const LlaisysQwen2Meta *meta,
    llaisysDeviceType_t device,
    int *device_ids,
    int ndevice) {
    if (!meta) {
        return nullptr;
    }

    llaisys::model::meta_data cfg{};
    cfg.hidden_size = meta->hs;
    cfg.num_hidden_layers = meta->nlayer;
    cfg.num_attention_heads = meta->nh;
    cfg.num_key_value_heads = meta->nkvh;
    cfg.intermediate_size = meta->di;
    cfg.max_position_embeddings = meta->maxseq;
    cfg.vocab_size = meta->voc;
    cfg.rms_norm_eps = meta->epsilon;
    cfg.rope_theta = static_cast<size_t>(meta->theta);
    cfg.torch_type = meta->dtype;
    cfg.hidden_act = llaisys::model::LLAISYS_SILU;
    cfg.bos_token_id = static_cast<size_t>(meta->end_token);
    cfg.eos_token_id = static_cast<size_t>(meta->end_token);
    cfg.attention_dropout = meta->attention_dropout;
    cfg.initializer_range = meta->initializer_range;
    cfg.max_window_layers = meta->max_window_layers;
    cfg.sliding_window = meta->sliding_window;
    cfg.tie_word_embeddings = meta->tie_word_embeddings != 0;
    cfg.use_cache = meta->use_cache != 0;
    cfg.use_mrope = meta->use_mrope != 0;
    cfg.use_sliding_window = meta->use_sliding_window != 0;

    llaisys::model::DeviceSpec spec{};
    spec.device_type = device;
    if (device_ids && ndevice > 0) {
        spec.device_ids.assign(device_ids, device_ids + ndevice);
    }

    llaisys::model::ParallelSpec parallel{};

    auto wrapper = new LlaisysQwen2Model{};
    wrapper->nlayer = meta->nlayer;
    wrapper->qwen2_model = std::make_shared<llaisys::model::Model_Qwen2>(cfg, spec, parallel);

    LOG_INFO("C_wrapper::llaisysQwen2ModelCreate:complete");

    return wrapper;
}

static llaisysTensor_t wrap_tensor(const llaisys::Weights_t &w) {
    if (!w) {
        return nullptr;
    }
    return new LlaisysTensor{w->weights()};
}

static void release_tensor(llaisysTensor_t &t) {
    delete t;
    t = nullptr;
}

static void clear_weights(LlaisysQwen2Model *model) {
    if (!model) {
        return;
    }
    release_tensor(model->c_weights.in_embed);
    release_tensor(model->c_weights.out_embed);
    release_tensor(model->c_weights.out_norm_w);
    for (auto &t : model->attn_norm_w) {
        release_tensor(t);
    }
    for (auto &t : model->attn_q_w) {
        release_tensor(t);
    }
    for (auto &t : model->attn_q_b) {
        release_tensor(t);
    }
    for (auto &t : model->attn_k_w) {
        release_tensor(t);
    }
    for (auto &t : model->attn_k_b) {
        release_tensor(t);
    }
    for (auto &t : model->attn_v_w) {
        release_tensor(t);
    }
    for (auto &t : model->attn_v_b) {
        release_tensor(t);
    }
    for (auto &t : model->attn_o_w) {
        release_tensor(t);
    }
    for (auto &t : model->mlp_norm_w) {
        release_tensor(t);
    }
    for (auto &t : model->mlp_gate_w) {
        release_tensor(t);
    }
    for (auto &t : model->mlp_up_w) {
        release_tensor(t);
    }
    for (auto &t : model->mlp_down_w) {
        release_tensor(t);
    }
    model->attn_norm_w.clear();
    model->attn_q_w.clear();
    model->attn_q_b.clear();
    model->attn_k_w.clear();
    model->attn_k_b.clear();
    model->attn_v_w.clear();
    model->attn_v_b.clear();
    model->attn_o_w.clear();
    model->mlp_norm_w.clear();
    model->mlp_gate_w.clear();
    model->mlp_up_w.clear();
    model->mlp_down_w.clear();
    model->weights_ready = false;
}

__export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
    if (!model) {
        return;
    }
    clear_weights(model);
    if (model->qwen2_model) {
        auto impl = std::dynamic_pointer_cast<llaisys::model::Model_Qwen2>(model->qwen2_model);
        if (impl) {
            impl->destroy();
        }
        model->qwen2_model.reset();
    }
    delete model;
}

__export void llaisysQwen2ModelLoadWeights(struct LlaisysQwen2Model *model,
                                           llaisysWeightBuffer_t buffer) {
    if (!model || !buffer || !model->qwen2_model) {
        return;
    }
    auto impl = std::dynamic_pointer_cast<llaisys::model::Model_Qwen2>(model->qwen2_model);
    if (!impl) {
        return;
    }
    LOG_INFO("llaisysQwen2ModelLoadWeights: buffer size=" << buffer->buffer.size());
    auto map = buffer->buffer.move_map();
    LOG_INFO("llaisysQwen2ModelLoadWeights: moved map size=" << map.size());
    impl->loadWeights(map);
    clear_weights(model);
}

__export struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model) {
    if (!model || !model->qwen2_model) {
        return nullptr;
    }
    auto impl = std::dynamic_pointer_cast<llaisys::model::Model_Qwen2>(model->qwen2_model);
    if (!impl) {
        return nullptr;
    }
    if (model->weights_ready) {
        return &model->c_weights;
    }
    const auto &qw = impl->weights();
    const size_t nlayer = qw.layers.size();
    if (nlayer == 0) {
        return nullptr;
    }
    clear_weights(model);
    model->nlayer = nlayer;

    model->c_weights.in_embed = wrap_tensor(qw.embed_tokens);
    model->c_weights.out_embed = wrap_tensor(qw.lm_head);
    model->c_weights.out_norm_w = wrap_tensor(qw.final_norm);

    model->attn_norm_w.resize(nlayer);
    model->attn_q_w.resize(nlayer);
    model->attn_q_b.resize(nlayer);
    model->attn_k_w.resize(nlayer);
    model->attn_k_b.resize(nlayer);
    model->attn_v_w.resize(nlayer);
    model->attn_v_b.resize(nlayer);
    model->attn_o_w.resize(nlayer);
    model->mlp_norm_w.resize(nlayer);
    model->mlp_gate_w.resize(nlayer);
    model->mlp_up_w.resize(nlayer);
    model->mlp_down_w.resize(nlayer);

    for (size_t i = 0; i < nlayer; ++i) {
        const auto &layer = qw.layers[i];
        model->attn_norm_w[i] = wrap_tensor(layer.input_layernorm.weight);
        model->attn_q_w[i] = wrap_tensor(layer.attention.q);
        model->attn_q_b[i] = wrap_tensor(layer.attention.bias_q);
        model->attn_k_w[i] = wrap_tensor(layer.attention.k);
        model->attn_k_b[i] = wrap_tensor(layer.attention.bias_k);
        model->attn_v_w[i] = wrap_tensor(layer.attention.v);
        model->attn_v_b[i] = wrap_tensor(layer.attention.bias_v);
        model->attn_o_w[i] = wrap_tensor(layer.attention.o);
        model->mlp_norm_w[i] = wrap_tensor(layer.post_attention_layernorm.weight);
        model->mlp_gate_w[i] = wrap_tensor(layer.mlp.gate);
        model->mlp_up_w[i] = wrap_tensor(layer.mlp.up);
        model->mlp_down_w[i] = wrap_tensor(layer.mlp.down);
    }

    model->c_weights.attn_norm_w = model->attn_norm_w.data();
    model->c_weights.attn_q_w = model->attn_q_w.data();
    model->c_weights.attn_q_b = model->attn_q_b.data();
    model->c_weights.attn_k_w = model->attn_k_w.data();
    model->c_weights.attn_k_b = model->attn_k_b.data();
    model->c_weights.attn_v_w = model->attn_v_w.data();
    model->c_weights.attn_v_b = model->attn_v_b.data();
    model->c_weights.attn_o_w = model->attn_o_w.data();
    model->c_weights.mlp_norm_w = model->mlp_norm_w.data();
    model->c_weights.mlp_gate_w = model->mlp_gate_w.data();
    model->c_weights.mlp_up_w = model->mlp_up_w.data();
    model->c_weights.mlp_down_w = model->mlp_down_w.data();
    model->weights_ready = true;
    return &model->c_weights;
}

__export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model *model, int64_t *token_ids,
                                        size_t ntoken) {
    if (!model || !model->qwen2_model || !token_ids || ntoken == 0) {
        return -1;
    }
    auto impl = std::dynamic_pointer_cast<llaisys::model::Model_Qwen2>(model->qwen2_model);
    if (!impl) {
        return -1;
    }
    std::vector<int64_t> tokens(token_ids, token_ids + ntoken);
    auto session = llaisys::model::naive_session::create(impl->config(), tokens);
    auto outputs = impl->inferStep(session);
    return outputs.next_token;
}

__export int64_t llaisysQwen2ModelInferDialog(struct LlaisysQwen2Model *model,
                                              int64_t *token_ids,
                                              size_t ntoken,
                                              size_t max_steps,
                                              int64_t *out_tokens,
                                              size_t out_ntoken) {
    if (!model || !model->qwen2_model || !token_ids || ntoken == 0) {
        return -1;
    }
    auto impl = std::dynamic_pointer_cast<llaisys::model::Model_Qwen2>(model->qwen2_model);
    if (!impl) {
        return -1;
    }
    std::vector<int64_t> tokens(token_ids, token_ids + ntoken);
    auto outputs = impl->inferDialog(tokens, max_steps);
    const size_t total = outputs.size();
    if (!out_tokens || out_ntoken == 0) {
        return static_cast<int64_t>(total);
    }
    const size_t to_copy = std::min(total, out_ntoken);
    std::memcpy(out_tokens, outputs.data(), to_copy * sizeof(int64_t));
    return static_cast<int64_t>(total);
}
}