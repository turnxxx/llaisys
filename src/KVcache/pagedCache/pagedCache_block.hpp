#pragma once

#include <cstdint>

struct KVCacheBlock {
    int block_id;                    // 物理块 ID [0, num_blocks)
    int ref_cnt = 0;                 // 引用计数
    KVCacheBlock* prev = nullptr;    // 空闲队列前驱
    KVCacheBlock* next = nullptr;    // 空闲队列后继

    explicit KVCacheBlock(int id) : block_id(id) {}

    // 是否在空闲队列中（prev/next 任一非空即可）
    bool in_free_queue() const { return prev != nullptr || next != nullptr; }
};