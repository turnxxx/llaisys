#include "free_block_queue.hpp"

KVCacheBlock* FreeKVCacheBlockQueue::popleft() {
    if (num_free_blocks_ == 0) {
        throw std::runtime_error("No free blocks in the queue");
    }

    KVCacheBlock* block = fake_head_.next;
    remove(block);
    return block;
}

std::vector<KVCacheBlock*> FreeKVCacheBlockQueue::popleft_n(int n) {
    if (n < 0) {
        throw std::invalid_argument("n must be non-negative");
    }
    if (n > num_free_blocks_) {
        throw std::runtime_error("Not enough free blocks in the queue");
    }

    std::vector<KVCacheBlock*> out;
    out.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        out.push_back(popleft());
    }
    return out;
}

void FreeKVCacheBlockQueue::remove(KVCacheBlock* block) {
    assert(block != nullptr);
    assert(block != &fake_head_ && block != &fake_tail_);
    assert(block->prev != nullptr);
    assert(block->next != nullptr);

    KVCacheBlock* prev = block->prev;
    KVCacheBlock* next = block->next;
    prev->next = next;
    next->prev = prev;

    block->prev = nullptr;
    block->next = nullptr;
    --num_free_blocks_;
}

void FreeKVCacheBlockQueue::append(KVCacheBlock* block) {
    assert(block != nullptr);
    assert(block != &fake_head_ && block != &fake_tail_);
    assert(block->prev == nullptr);
    assert(block->next == nullptr);

    KVCacheBlock* last = fake_tail_.prev;
    assert(last != nullptr);

    last->next = block;
    block->prev = last;
    block->next = &fake_tail_;
    fake_tail_.prev = block;
    ++num_free_blocks_;
}

void FreeKVCacheBlockQueue::append_n(const std::vector<KVCacheBlock*>& blocks) {
    for (KVCacheBlock* block : blocks) {
        append(block);
    }
}