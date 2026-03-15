#pragma once
#include "pagedCache_block.hpp"
#include <vector>
#include <stdexcept>
#include <cassert>

// 带哨兵节点的双向链表，实现 LRU 空闲块队列
// 队头 = 最久未使用（最先被分配/驱逐）
// 队尾 = 最近释放（最后被分配/驱逐）
class FreeKVCacheBlockQueue {
public:
    FreeKVCacheBlockQueue() {
        fake_head_.next = &fake_tail_;
        fake_tail_.prev = &fake_head_;
    }

    // 用一组块初始化队列，块按 block_id 顺序入队
    explicit FreeKVCacheBlockQueue(std::vector<KVCacheBlock>& blocks)
        : FreeKVCacheBlockQueue() {
        for (auto& b : blocks) {
            append(&b);
        }
    }

    // 从队头弹出一个块（分配时调用）O(1)
    KVCacheBlock* popleft();

    // 从队头弹出 n 个块 O(n)
    std::vector<KVCacheBlock*> popleft_n(int n);

    // 从队列中间移除指定块（touch 时调用）O(1)
    void remove(KVCacheBlock* block);

    // 追加到队尾（释放时调用）O(1)
    void append(KVCacheBlock* block);

    // 批量追加 O(n)
    void append_n(const std::vector<KVCacheBlock*>& blocks);

    int num_free_blocks() const { return num_free_blocks_; }

private:
    KVCacheBlock fake_head_{-1};   // 哨兵头节点
    KVCacheBlock fake_tail_{-1};   // 哨兵尾节点
    int num_free_blocks_ = 0;
};