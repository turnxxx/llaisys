#include "Decoder.hpp"
#include <cmath>
namespace llaisys::Qwen2 {
// 一次decoder计算
tensor_t qwen2_decoder(tensor_t &hidden_states,
                       const layer_weights &weights,
                       llaisys::KVcache::KVcache_t kv_cache,
                       const llaisys::model::meta_data &meta_data,
                       size_t token_pos,
                       size_t layer) {
    LOG_INFO("qwen2_decoder::begin:token_pos:" << token_pos);
    LOG_INFO("qwen2_decoder::begin:nlayer: " << layer);
    LOG_TENSOR_META_AT("qwen2_decoder::begin:hidden_states", hidden_states);
    // step1：shape检查
    size_t hidden_size = meta_data.hidden_size;                 // 1536
    size_t num_attention_heads = meta_data.num_attention_heads; // 12
    size_t num_key_value_heads = meta_data.num_key_value_heads; // 2
    size_t seq_len = hidden_states->shape()[0];
    size_t head_dim = hidden_size / num_attention_heads; // 128
    size_t kv_dim = num_key_value_heads * head_dim;      // 256
    ASSERT(hidden_states->shape()[1] == hidden_size, "qwen2_decoder: hidden_size mismatch");
    // 读取权重指针
    Weights_t Wq = weights.attention.q;
    Weights_t Wk = weights.attention.k;
    Weights_t Wv = weights.attention.v;
    Weights_t Wo = weights.attention.o;
    Weights_t bias_q = weights.attention.bias_q;
    Weights_t bias_k = weights.attention.bias_k;
    Weights_t bias_v = weights.attention.bias_v;
    Weights_t input_lm_weight = weights.input_layernorm.weight;
    Weights_t post_attn_weight = weights.post_attention_layernorm.weight;
    Weights_t gate = weights.mlp.gate;
    Weights_t up = weights.mlp.up;
    Weights_t down = weights.mlp.down;
    // 读取meta数据
    llaisysDataType_t dtype = meta_data.torch_type;
    llaisysDeviceType_t device_type = hidden_states->deviceType();
    float rms_norm_eps = meta_data.rms_norm_eps;
    float rope_theta = meta_data.rope_theta;
    // input_layernorm计算
    std::vector<size_t> input_normed_shape(hidden_states->shape());
    tensor_t input_normed_states = llaisys::Tensor::create(input_normed_shape, dtype, device_type);
    ops::rms_norm(input_normed_states, hidden_states, input_lm_weight->weights(), rms_norm_eps);
    LOG_TENSOR_META_AT("input_normed_states:", input_normed_states);
    // Q,K,V投影,调用Linear算子
    std::vector<size_t> Q_shape{seq_len, hidden_size};
    std::vector<size_t> K_shape{seq_len, kv_dim};
    std::vector<size_t> V_shape{seq_len, kv_dim};
    tensor_t q = llaisys::Tensor::create(Q_shape, dtype, device_type);
    tensor_t k = llaisys::Tensor::create(K_shape, dtype, device_type);
    tensor_t v = llaisys::Tensor::create(V_shape, dtype, device_type);
    ops::linear(q, input_normed_states, Wq->weights(), bias_q->weights());
    ops::linear(k, input_normed_states, Wk->weights(), bias_k->weights());
    ops::linear(v, input_normed_states, Wv->weights(), bias_v->weights());
    LOG_TENSOR_META_AT("q:", q);
    LOG_TENSOR_META_AT("k:", k);
    LOG_TENSOR_META_AT("v:", v);
    std::vector<size_t> ghq_q_shape{seq_len, num_attention_heads, head_dim};
    std::vector<size_t> ghq_k_shape{seq_len, num_key_value_heads, head_dim};
    std::vector<size_t> ghq_v_shape{seq_len, num_key_value_heads, head_dim};
    tensor_t q_3d = q->reshape(ghq_q_shape);
    tensor_t k_3d = k->reshape(ghq_k_shape);
    tensor_t v_3d = v->reshape(ghq_v_shape);
    // rope
    tensor_t q_rope = llaisys::Tensor::create(ghq_q_shape, dtype, device_type);
    tensor_t k_rope = llaisys::Tensor::create(ghq_k_shape, dtype, device_type);
    std::vector<int64_t> pos_ids_host(seq_len);
    for (size_t i = 0; i < seq_len; ++i) {
        pos_ids_host[i] = static_cast<int64_t>(token_pos + i);
    }
    tensor_t pos_ids = llaisys::Tensor::create({seq_len}, LLAISYS_DTYPE_I64, device_type);
    pos_ids->load(pos_ids_host.data());
    ops::rope(q_rope, q_3d, pos_ids, rope_theta);
    ops::rope(k_rope, k_3d, pos_ids, rope_theta);
    LOG_TENSOR_META_AT("q_rope:", q_rope);
    LOG_TENSOR_META_AT("k_rope:", k_rope);
    // 存入KVcache
    kv_cache->append(layer, k_rope, v_3d);
    // GQA
    tensor_t attn_val = Tensor::create(q_rope->shape(), dtype, device_type);
    float scale = 1 / sqrt(head_dim);
    // 从KV_cache里取出张量
    tensor_t k_attn;
    tensor_t v_attn;
    kv_cache->get(k_attn, v_attn, layer);
    LOG_TENSOR_META_AT("k_attn:", k_attn);
    LOG_TENSOR_META_AT("v_attn:", v_attn);
    ops::self_attention(attn_val, q_rope, k_attn, v_attn, scale);
    LOG_TENSOR_META_AT("attn_val", attn_val);
    tensor_t attn_val_2d = attn_val->reshape({seq_len, hidden_size});
    LOG_TENSOR_META_AT("attn_val_2d", attn_val_2d);
    tensor_t attn_output = Tensor::create({seq_len, hidden_size}, dtype, device_type);
    LOG_TENSOR_META_AT("attn_output", attn_output);
    LOG_TENSOR_META_AT("Wo:", Wo->weights());
    ops::linear(attn_output, attn_val_2d, Wo->weights(), nullptr);
    LOG_TENSOR_META_AT("attn_output", attn_output);
    // 残差连接
    tensor_t self_attn_output = Tensor::create({seq_len, hidden_size}, dtype, device_type);
    ops::add(self_attn_output, hidden_states, attn_output);
    LOG_TENSOR_META_AT("self_attn_output", self_attn_output);
    // MLP层
    tensor_t post_attn_normed = Tensor::create({seq_len, hidden_size}, dtype, device_type);
    ops::rms_norm(post_attn_normed, self_attn_output, post_attn_weight->weights(), rms_norm_eps);
    LOG_TENSOR_META_AT("post_attn_normed", post_attn_normed);
    size_t intermediate_size = meta_data.intermediate_size;
    tensor_t gate_proj = Tensor::create({seq_len, intermediate_size}, dtype, device_type);
    tensor_t up_proj = Tensor::create({seq_len, intermediate_size}, dtype, device_type);
    ops::linear(gate_proj, post_attn_normed, gate->weights(), nullptr);
    ops::linear(up_proj, post_attn_normed, up->weights(), nullptr);

    tensor_t mlp_hidden = Tensor::create({seq_len, intermediate_size}, dtype, device_type);
    ops::swiglu(mlp_hidden, gate_proj, up_proj);

    tensor_t mlp_out = Tensor::create({seq_len, hidden_size}, dtype, device_type);
    ops::linear(mlp_out, mlp_hidden, down->weights(), nullptr);

    tensor_t output = Tensor::create({seq_len, hidden_size}, dtype, device_type);

    ops::add(output, self_attn_output, mlp_out);
    LOG_TENSOR_META_AT("decoder_output:", output);
    LOG_INFO("qwen2_decoder::token_pos" << token_pos);
    LOG_INFO("qwen2_decoder::nlayer: " << layer);
    return output;
}
} // namespace llaisys::Qwen2