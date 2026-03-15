#include "block_pool.hpp"

#include <stdexcept>

BlockPool::BlockPool(int num_blocks) : num_blocks_(num_blocks) {
    if (num_blocks_ < 0) {
        throw std::invalid_argument("num_blocks must be non-negative");
    }

    blocks_.reserve(static_cast<size_t>(num_blocks_));
    for (int i = 0; i < num_blocks_; ++i) {
        blocks_.emplace_back(i);
    }
    for (int i = 0; i < num_blocks_; ++i) {
        free_queue_.append(&blocks_[i]);
    }
}

std::vector<KVCacheBlock*> BlockPool::allocate(int n) {
    if (n < 0) {
        throw std::invalid_argument("n must be non-negative");
    }
    if (n == 0) {
        return {};
    }
    if (!can_allocate(n)) {
        return {};
    }

    std::vector<KVCacheBlock*> allocated;
    allocated.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        KVCacheBlock* block = free_queue_.popleft();
        block->ref_cnt = 1;
        allocated.push_back(block);
    }
    return allocated;
}

bool BlockPool::can_allocate(int n) const {
    if (n < 0) {
        return false;
    }
    return free_queue_.num_free_blocks() >= n;
}

void BlockPool::free_blocks(const std::vector<KVCacheBlock*>& blocks) {
    for (KVCacheBlock* block : blocks) {
        assert(block != nullptr);
        assert(block->block_id >= 0 && block->block_id < num_blocks_);
        assert(block->ref_cnt > 0);

        --block->ref_cnt;
        if (block->ref_cnt == 0) {
            free_queue_.append(block);
        }
    }
}

void BlockPool::touch(const std::vector<KVCacheBlock*>& blocks) {
    for (KVCacheBlock* block : blocks) {
        assert(block != nullptr);
        assert(block->block_id >= 0 && block->block_id < num_blocks_);

        if (block->in_free_queue()) {
            free_queue_.remove(block);
        }
        ++block->ref_cnt;
    }
}

int BlockPool::num_free_blocks() const {
    return free_queue_.num_free_blocks();
}

int BlockPool::num_total_blocks() const {
    return num_blocks_;
}

float BlockPool::usage() const {
    if (num_blocks_ == 0) {
        return 0.0f;
    }
    return static_cast<float>(num_blocks_ - num_free_blocks()) /
           static_cast<float>(num_blocks_);
}

KVCacheBlock* BlockPool::get_block(int block_id) {
    if (block_id < 0 || block_id >= num_blocks_) {
        return nullptr;
    }
    return &blocks_[static_cast<size_t>(block_id)];
}