#pragma once
#include "pagedCache_config.hpp"
#include "block_pool.hpp"
#include "block_table.hpp"
#include <vector>
#include <cstdint>
#include <unordered_map>
#include <optional>

// 分配结果
struct AllocResult {
    bool success;                     // 是否分配成功
    std::vector<int> new_block_ids;   // 新分配的物理块 ID
    int num_free_blocks_after;        // 分配后剩余空闲块数
};

// 单个请求的 KV cache 状态
struct RequestKVState {
    int request_id;
    int num_allocated_tokens = 0;  // 已分配 slot 的 token 数
    int num_blocks = 0;            // 当前使用的块数
    int row_idx = -1;              // 在 BlockTable 中的行索引
    std::vector<KVCacheBlock*> blocks;  // 持有的块指针（按分配顺序）
};

// ============================================================
// KVCacheManager: 顶层接口
//
// 使用方式（单对话）:
//   auto mgr = KVCacheManager(config, num_blocks);
//   int req = mgr.add_request();
//
//   // Prefill
//   auto result = mgr.allocate_tokens(req, num_prompt_tokens);
//   auto slots = mgr.get_slot_mapping(req, positions);
//   // -> 调用 reshape_and_cache(k, v, k_cache, v_cache, slots)
//   // -> 调用 attention(q, k, v)  [prefill 不需要 paged attn]
//
//   // Decode loop
//   while (!done) {
//       auto result = mgr.allocate_tokens(req, 1);
//       auto slot = mgr.get_slot_for_token(req, new_position);
//       // -> reshape_and_cache 写 1 个 token
//       // -> paged_attention 读历史 (用 mgr.get_block_table_row(req))
//   }
//
//   mgr.remove_request(req);
//
// ============================================================

class KVCacheManager {
public:
    // 构造: config 含模型参数, num_blocks 由 GPU 剩余显存计算得出
    KVCacheManager(const KVCacheConfig& config, int num_blocks);
    ~KVCacheManager();

    // ============ 请求生命周期 ============

    // 注册一个新请求，返回 request_id
    int add_request();

    // 移除请求，释放其所有 KV cache 块
    void remove_request(int request_id);

    // 请求是否存在
    bool has_request(int request_id) const;

    // ============ 块分配 ============

    // 为请求分配 num_tokens 个 token 的 KV cache 空间
    // - Prefill 时: num_tokens = prompt_length
    // - Decode 时:  num_tokens = 1（通常）
    // 返回 AllocResult: success=true 时包含新分配的 block_ids
    // success=false 时表示空闲块不足，调用方可做抢占后重试
    AllocResult allocate_tokens(int request_id, int num_tokens);

    // 查询是否能为某请求分配 num_tokens 的空间
    bool can_allocate(int request_id, int num_tokens) const;

    // ============ Slot Mapping（写入 KV cache 用）============

    // 计算一批 token 的 slot mapping
    // positions: 这些 token 在序列中的绝对位置
    // 输出: slot_mapping，长度 = positions.size()
    std::vector<int64_t> get_slot_mapping(
        int request_id,
        const std::vector<int>& positions
    ) const;

    // 计算单个 token 的 slot（decode 时常用）
    int64_t get_slot_for_token(int request_id, int position) const;

    // ============ Block Table（读取 KV cache 用）============

    // 获取请求的 block table（物理块 ID 列表）
    // 传给 paged_attention kernel
    std::vector<int32_t> get_block_table_row(int request_id) const;

    // 获取请求的上下文长度（已分配的 token 数）
    int get_context_len(int request_id) const;
    int get_row_idx(int request_id) const;

    // ============ 底层数据访问（给 GPU kernel 用）============

    // 获取完整 block table 二维数组的指针
    // shape: [max_num_reqs, max_num_blocks_per_req], int32
    const int32_t* block_table_data() const;

    // 获取 BlockTable 对象（如需自定义操作）
    const BlockTable& block_table() const;

    // ============ 全局状态查询 ============

    int num_free_blocks() const;
    int num_total_blocks() const;
    float usage() const;                    // KV cache 利用率
    int num_active_requests() const;
    int block_size() const;

    // ============ 未来扩展接口（暂为空实现）============

    // Prefix caching: 查找缓存命中的块
    // int find_cached_blocks(int request_id, const std::vector<int>& token_ids);

    // Preemption: 释放一个请求的所有块但保留状态，以便后续恢复
    // void preempt_request(int request_id);
    // void resume_request(int request_id);

private:
    KVCacheConfig config_;
    BlockPool pool_;
    BlockTable block_table_;

    int next_request_id_ = 0;
    int next_row_idx_ = 0;  // 简单的行分配器

    // request_id -> 请求 KV 状态
    std::unordered_map<int, RequestKVState> requests_;

    // 释放的行索引回收池
    std::vector<int> free_row_indices_;

    int alloc_row_idx();
    void free_row_idx(int idx);
};