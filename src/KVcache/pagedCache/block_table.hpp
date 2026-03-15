#pragma once
#include <vector>
#include <cstdint>
#include <cassert>

// 管理多个请求的 block table 和 slot mapping 计算
class BlockTable {
public:
    BlockTable(int block_size, int max_num_reqs, int max_num_blocks_per_req);

    // ---- 行操作（每行 = 一个请求的 block table）----

    // 为请求 row_idx 追加新的物理块 ID
    void append_row(int row_idx, const std::vector<int>& block_ids);

    // 重置并设置请求 row_idx 的全部块 ID
    void set_row(int row_idx, const std::vector<int>& block_ids);

    // 清空请求 row_idx 的 block table
    void clear_row(int row_idx);

    // 复制/交换行（用于 scheduler 中请求重排序）
    void move_row(int src, int dst);
    void swap_row(int src, int dst);

    // ---- 查询 ----

    // 获取请求 row_idx 的物理块 ID 列表
    std::vector<int32_t> get_row(int row_idx) const;

    // 获取单个逻辑块对应的物理块 ID
    int32_t get_physical_block(int row_idx, int logical_block_idx) const;

    // 请求当前的块数
    int num_blocks_of(int row_idx) const;

    // ---- Slot Mapping 计算 ----

    // 为一批 token 计算 slot mapping
    // req_indices[i]: 第 i 个 token 属于哪个请求
    // positions[i]:   第 i 个 token 在序列中的位置
    // 输出: slot_mapping[i] = physical_block * block_size + offset
    void compute_slot_mapping(
        const int* req_indices,
        const int* positions,
        int num_tokens,
        int64_t* out_slot_mapping  // 输出数组
    ) const;

    // ---- 为 attention kernel 导出 ----

    // 获取整个 block table 的底层数据指针（用于传给 GPU kernel）
    // 形状: [max_num_reqs, max_num_blocks_per_req]，row-major
    const int32_t* data() const;

    int block_size() const { return block_size_; }
    int max_num_reqs() const { return max_num_reqs_; }
    int max_num_blocks_per_req() const { return max_num_blocks_per_req_; }

private:
    int block_size_;
    int max_num_reqs_;
    int max_num_blocks_per_req_;
    std::vector<int32_t> table_;       // 展平的 2D 数组
    std::vector<int32_t> num_blocks_;  // 每行的块数
};