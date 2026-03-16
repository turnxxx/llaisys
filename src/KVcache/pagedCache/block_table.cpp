#include "block_table.hpp"

#include <algorithm>
#include <stdexcept>

BlockTable::BlockTable(int block_size, int max_num_reqs, int max_num_blocks_per_req)
    : block_size_(block_size), max_num_reqs_(max_num_reqs), max_num_blocks_per_req_(max_num_blocks_per_req),
      table_(static_cast<size_t>(max_num_reqs) * static_cast<size_t>(max_num_blocks_per_req), -1),
      num_blocks_(static_cast<size_t>(max_num_reqs), 0) {
    if (block_size_ <= 0 || max_num_reqs_ < 0 || max_num_blocks_per_req_ < 0) {
        throw std::invalid_argument("Invalid BlockTable shape arguments");
    }
}

void BlockTable::append_row(int row_idx, const std::vector<int>& block_ids) {
    assert(row_idx >= 0 && row_idx < max_num_reqs_);
    const int old_size = num_blocks_[static_cast<size_t>(row_idx)];
    const int append_size = static_cast<int>(block_ids.size());
    assert(old_size + append_size <= max_num_blocks_per_req_);

    const int base = row_idx * max_num_blocks_per_req_;
    for (int i = 0; i < append_size; ++i) {
        table_[static_cast<size_t>(base + old_size + i)] = static_cast<int32_t>(block_ids[static_cast<size_t>(i)]);
    }
    num_blocks_[static_cast<size_t>(row_idx)] = old_size + append_size;
}

void BlockTable::set_row(int row_idx, const std::vector<int>& block_ids) {
    assert(row_idx >= 0 && row_idx < max_num_reqs_);
    assert(static_cast<int>(block_ids.size()) <= max_num_blocks_per_req_);

    clear_row(row_idx);
    append_row(row_idx, block_ids);
}

void BlockTable::clear_row(int row_idx) {
    assert(row_idx >= 0 && row_idx < max_num_reqs_);
    const int base = row_idx * max_num_blocks_per_req_;
    std::fill_n(table_.begin() + base, max_num_blocks_per_req_, static_cast<int32_t>(-1));
    num_blocks_[static_cast<size_t>(row_idx)] = 0;
}

void BlockTable::move_row(int src, int dst) {
    assert(src >= 0 && src < max_num_reqs_);
    assert(dst >= 0 && dst < max_num_reqs_);
    if (src == dst) {
        return;
    }

    const int src_base = src * max_num_blocks_per_req_;
    const int dst_base = dst * max_num_blocks_per_req_;
    std::copy_n(table_.begin() + src_base, max_num_blocks_per_req_, table_.begin() + dst_base);
    num_blocks_[static_cast<size_t>(dst)] = num_blocks_[static_cast<size_t>(src)];
    clear_row(src);
}

void BlockTable::swap_row(int src, int dst) {
    assert(src >= 0 && src < max_num_reqs_);
    assert(dst >= 0 && dst < max_num_reqs_);
    if (src == dst) {
        return;
    }

    const int src_base = src * max_num_blocks_per_req_;
    const int dst_base = dst * max_num_blocks_per_req_;
    std::swap_ranges(table_.begin() + src_base, table_.begin() + src_base + max_num_blocks_per_req_,
                     table_.begin() + dst_base);
    std::swap(num_blocks_[static_cast<size_t>(src)], num_blocks_[static_cast<size_t>(dst)]);
}

std::vector<int32_t> BlockTable::get_row(int row_idx) const {
    assert(row_idx >= 0 && row_idx < max_num_reqs_);
    const int count = num_blocks_[static_cast<size_t>(row_idx)];
    const int base = row_idx * max_num_blocks_per_req_;
    return std::vector<int32_t>(table_.begin() + base, table_.begin() + base + count);
}

int32_t BlockTable::get_physical_block(int row_idx, int logical_block_idx) const {
    assert(row_idx >= 0 && row_idx < max_num_reqs_);
    assert(logical_block_idx >= 0);
    assert(logical_block_idx < num_blocks_[static_cast<size_t>(row_idx)]);

    const int idx = row_idx * max_num_blocks_per_req_ + logical_block_idx;
    return table_[static_cast<size_t>(idx)];
}

int BlockTable::num_blocks_of(int row_idx) const {
    assert(row_idx >= 0 && row_idx < max_num_reqs_);
    return num_blocks_[static_cast<size_t>(row_idx)];
}

void BlockTable::compute_slot_mapping(const int* req_indices, const int* positions, int num_tokens,
                                      int64_t* out_slot_mapping) const {
    assert(req_indices != nullptr);
    assert(positions != nullptr);
    assert(out_slot_mapping != nullptr);
    assert(num_tokens >= 0);

    for (int i = 0; i < num_tokens; ++i) {
        const int row_idx = req_indices[i];
        const int pos = positions[i];

        assert(row_idx >= 0 && row_idx < max_num_reqs_);
        assert(pos >= 0);

        const int logical_block_idx = pos / block_size_;
        const int offset = pos % block_size_;
        const int num_blocks = num_blocks_[static_cast<size_t>(row_idx)];
        assert(logical_block_idx < num_blocks);
        (void)num_blocks; // 避免 Release 下 assert 被去掉时的未使用变量警告

        const int32_t physical_block = get_physical_block(row_idx, logical_block_idx);
        assert(physical_block >= 0);
        out_slot_mapping[i] = static_cast<int64_t>(physical_block) * static_cast<int64_t>(block_size_) + offset;
    }
}

const int32_t* BlockTable::data() const {
    return table_.data();
}