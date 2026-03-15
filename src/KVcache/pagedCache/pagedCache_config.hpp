#pragma once
#include <cstdint>

struct KVCacheConfig {
    int num_layers;          // 模型层数
    int num_kv_heads;        // KV attention 头数
    int head_size;           // 每个头的维度 (如 128)
    int block_size;          // 每个块容纳的 token 数 (如 16)
    int max_num_reqs;        // 最大并发请求数 (单对话=1)
    int max_model_len;       // 最大序列长度

    // 每块在单层上的字节数: 2(K+V) * block_size * num_kv_heads * head_size * dtype_size
    int dtype_size;          // 数据类型字节数 (fp16=2, bf16=2, fp8=1)

    int64_t page_size_bytes() const {
        return 2LL * block_size * num_kv_heads * head_size * dtype_size;
    }

    int64_t page_size_bytes_all_layers() const {
        return page_size_bytes() * num_layers;
    }

    int max_num_blocks_per_req() const {
        return (max_model_len + block_size - 1) / block_size;
    }
};