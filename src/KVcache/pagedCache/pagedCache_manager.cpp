#include "pagedCache_manager.hpp"

#include <stdexcept>
#include <utility>

KVCacheManager::KVCacheManager(const KVCacheConfig& config, int num_blocks)
    : config_(config),
      pool_(num_blocks),
      block_table_(config.block_size, config.max_num_reqs, config.max_num_blocks_per_req()) {
    if (config_.block_size <= 0 || config_.max_num_reqs <= 0 || config_.max_model_len <= 0) {
        throw std::invalid_argument("Invalid KVCacheConfig");
    }
}

KVCacheManager::~KVCacheManager() = default;

int KVCacheManager::add_request() {
    const int request_id = next_request_id_++;
    const int row_idx = alloc_row_idx();

    RequestKVState state;
    state.request_id = request_id;
    state.row_idx = row_idx;
    requests_.emplace(request_id, std::move(state));
    return request_id;
}

void KVCacheManager::remove_request(int request_id) {
    auto it = requests_.find(request_id);
    if (it == requests_.end()) {
        return;
    }

    RequestKVState& state = it->second;
    if (!state.blocks.empty()) {
        pool_.free_blocks(state.blocks);
    }
    free_row_idx(state.row_idx);
    requests_.erase(it);
}

bool KVCacheManager::has_request(int request_id) const {
    return requests_.find(request_id) != requests_.end();
}

AllocResult KVCacheManager::allocate_tokens(int request_id, int num_tokens) {
    if (num_tokens < 0) {
        throw std::invalid_argument("num_tokens must be non-negative");
    }

    auto it = requests_.find(request_id);
    if (it == requests_.end()) {
        throw std::out_of_range("request_id not found");
    }

    RequestKVState& state = it->second;
    if (num_tokens == 0) {
        return AllocResult{true, {}, pool_.num_free_blocks()};
    }

    const int old_tokens = state.num_allocated_tokens;
    const int old_blocks = state.num_blocks;
    const int new_total_tokens = old_tokens + num_tokens;
    const int new_total_blocks =
        (new_total_tokens + config_.block_size - 1) / config_.block_size;
    const int need_new_blocks = new_total_blocks - old_blocks;

    if (need_new_blocks <= 0) {
        state.num_allocated_tokens = new_total_tokens;
        return AllocResult{true, {}, pool_.num_free_blocks()};
    }

    if (!pool_.can_allocate(need_new_blocks)) {
        return AllocResult{false, {}, pool_.num_free_blocks()};
    }

    std::vector<KVCacheBlock*> new_blocks = pool_.allocate(need_new_blocks);
    if (static_cast<int>(new_blocks.size()) != need_new_blocks) {
        return AllocResult{false, {}, pool_.num_free_blocks()};
    }

    std::vector<int> new_block_ids;
    new_block_ids.reserve(static_cast<size_t>(need_new_blocks));
    for (KVCacheBlock* block : new_blocks) {
        state.blocks.push_back(block);
        new_block_ids.push_back(block->block_id);
    }

    block_table_.append_row(state.row_idx, new_block_ids);
    state.num_blocks = new_total_blocks;
    state.num_allocated_tokens = new_total_tokens;

    return AllocResult{true, std::move(new_block_ids), pool_.num_free_blocks()};
}

bool KVCacheManager::can_allocate(int request_id, int num_tokens) const {
    if (num_tokens < 0) {
        return false;
    }

    auto it = requests_.find(request_id);
    if (it == requests_.end()) {
        return false;
    }

    const RequestKVState& state = it->second;
    const int new_total_tokens = state.num_allocated_tokens + num_tokens;
    const int new_total_blocks =
        (new_total_tokens + config_.block_size - 1) / config_.block_size;
    const int need_new_blocks = new_total_blocks - state.num_blocks;
    if (need_new_blocks <= 0) {
        return true;
    }
    return pool_.can_allocate(need_new_blocks);
}

std::vector<int64_t> KVCacheManager::get_slot_mapping(
    int request_id,
    const std::vector<int>& positions) const {
    auto it = requests_.find(request_id);
    if (it == requests_.end()) {
        throw std::out_of_range("request_id not found");
    }

    const RequestKVState& state = it->second;
    const int num_tokens = static_cast<int>(positions.size());
    std::vector<int64_t> slot_mapping(static_cast<size_t>(num_tokens), -1);
    if (num_tokens == 0) {
        return slot_mapping;
    }

    std::vector<int> req_indices(static_cast<size_t>(num_tokens), state.row_idx);
    block_table_.compute_slot_mapping(
        req_indices.data(),
        positions.data(),
        num_tokens,
        slot_mapping.data());
    return slot_mapping;
}

int64_t KVCacheManager::get_slot_for_token(int request_id, int position) const {
    const std::vector<int> positions{position};
    const std::vector<int64_t> slots = get_slot_mapping(request_id, positions);
    return slots[0];
}

std::vector<int32_t> KVCacheManager::get_block_table_row(int request_id) const {
    auto it = requests_.find(request_id);
    if (it == requests_.end()) {
        throw std::out_of_range("request_id not found");
    }
    return block_table_.get_row(it->second.row_idx);
}

int KVCacheManager::get_context_len(int request_id) const {
    auto it = requests_.find(request_id);
    if (it == requests_.end()) {
        throw std::out_of_range("request_id not found");
    }
    return it->second.num_allocated_tokens;
}

int KVCacheManager::get_row_idx(int request_id) const {
    auto it = requests_.find(request_id);
    if (it == requests_.end()) {
        throw std::out_of_range("request_id not found");
    }
    return it->second.row_idx;
}

const int32_t* KVCacheManager::block_table_data() const {
    return block_table_.data();
}

const BlockTable& KVCacheManager::block_table() const {
    return block_table_;
}

int KVCacheManager::num_free_blocks() const {
    return pool_.num_free_blocks();
}

int KVCacheManager::num_total_blocks() const {
    return pool_.num_total_blocks();
}

float KVCacheManager::usage() const {
    return pool_.usage();
}

int KVCacheManager::num_active_requests() const {
    return static_cast<int>(requests_.size());
}

int KVCacheManager::block_size() const {
    return config_.block_size;
}

int KVCacheManager::alloc_row_idx() {
    if (!free_row_indices_.empty()) {
        const int idx = free_row_indices_.back();
        free_row_indices_.pop_back();
        return idx;
    }

    if (next_row_idx_ >= config_.max_num_reqs) {
        throw std::runtime_error("No free row in block table");
    }
    return next_row_idx_++;
}

void KVCacheManager::free_row_idx(int idx) {
    if (idx < 0 || idx >= config_.max_num_reqs) {
        throw std::out_of_range("row_idx out of range");
    }
    block_table_.clear_row(idx);
    free_row_indices_.push_back(idx);
}
