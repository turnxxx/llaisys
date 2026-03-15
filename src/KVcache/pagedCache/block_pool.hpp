#pragma once
#include "pagedCache_block.hpp"
#include "free_block_queue.hpp"
#include <vector>
#include <cassert>

class BlockPool {
public:
    // 创建含 num_blocks 个块的池，所有块初始为空闲
    explicit BlockPool(int num_blocks);

    // ---- 分配 ----

    // 分配 n 个新块，ref_cnt 设为 1
    // 若空闲块不足则返回空 vector（调用方可据此决定是否抢占）
    std::vector<KVCacheBlock*> allocate(int n);

    // 检查是否能分配 n 个块
    bool can_allocate(int n) const;

    // ---- 释放 ----

    // 释放一组块（ref_cnt--，归零则回到空闲队列）
    // 调用方负责按逆序传入（尾部块先释放 = 先被驱逐）
    void free_blocks(const std::vector<KVCacheBlock*>& blocks);

    // ---- Touch (prefix caching 扩展用) ----

    // 增加引用计数；若块在空闲队列中则移除（防止被驱逐）
    void touch(const std::vector<KVCacheBlock*>& blocks);

    // ---- 查询 ----

    int num_free_blocks() const;
    int num_total_blocks() const;
    float usage() const;  // 已用比例 [0.0, 1.0]

    // 按 block_id 获取块（用于从外部恢复状态等）
    KVCacheBlock* get_block(int block_id);

private:
    int num_blocks_;
    std::vector<KVCacheBlock> blocks_;       // 所有块，按 block_id 索引
    FreeKVCacheBlockQueue free_queue_;
};