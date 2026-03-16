#include "naive_session.hpp"

namespace llaisys::model {

void naive_session::init(std::vector<int64_t>& tokens, CacheHandle_t handle) {
    tokens_ = tokens;
    seq_len_ = tokens.size();
    token_pos_ = 0;
    cache_ = std::move(handle);
}
void naive_session::append(int64_t next_token) {
    tokens_.push_back(next_token);
    seq_len_ = tokens_.size();
    token_pos_ = seq_len_;
}
const std::vector<int64_t> &naive_session::tokens() const {
    return tokens_;
}

size_t naive_session::seq_len() const {
    return seq_len_;
}

size_t naive_session::token_pos() const {
    return token_pos_;
}
} // namespace llaisys::model
