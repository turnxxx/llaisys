#ifndef LLAISYS_MODELS_QWEN2_H
#define LLAISYS_MODELS_QWEN2_H

#include "../tensor.h"
#include "../weights_buffer.h"
__C {
    struct LlaisysQwen2Meta {
        llaisysDataType_t dtype;
        size_t nlayer, hs, nh, nkvh, dh, di, maxseq, voc;
        float epsilon, theta;
        int64_t end_token;
        float attention_dropout;
        float initializer_range;
        size_t max_window_layers;
        size_t sliding_window;
        int tie_word_embeddings;
        int use_cache;
        int use_mrope;
        int use_sliding_window;
    };

    struct LlaisysQwen2Weights {
        llaisysTensor_t in_embed;
        llaisysTensor_t out_embed;
        llaisysTensor_t out_norm_w;   // a.k.a. model.norm.weight
        llaisysTensor_t *attn_norm_w; // a.k.a. input_layernorm.weight
        llaisysTensor_t *attn_q_w;
        llaisysTensor_t *attn_q_b;
        llaisysTensor_t *attn_k_w;
        llaisysTensor_t *attn_k_b;
        llaisysTensor_t *attn_v_w;
        llaisysTensor_t *attn_v_b;
        llaisysTensor_t *attn_o_w;
        llaisysTensor_t *mlp_norm_w; // a.k.a. post_attention_layernorm.weight
        llaisysTensor_t *mlp_gate_w;
        llaisysTensor_t *mlp_up_w;
        llaisysTensor_t *mlp_down_w;
    };

    struct LlaisysQwen2Model;

    __export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice);

    __export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model);

    __export void llaisysQwen2ModelLoadWeights(struct LlaisysQwen2Model * model,
                                               llaisysWeightBuffer_t buffer);

    __export struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model);

    __export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t *token_ids, size_t ntoken);

    // out_tokens: caller-allocated buffer (can be nullptr). out_ntoken: capacity.
    // return: <0 error, >=0 actual token count required/returned.
    __export int64_t llaisysQwen2ModelInferDialog(struct LlaisysQwen2Model * model,
                                                  int64_t *token_ids,
                                                  size_t ntoken,
                                                  size_t max_steps,
                                                  int64_t *out_tokens,
                                                  size_t out_ntoken);
}
#endif // LLAISYS_MODELS_QWEN2_H
