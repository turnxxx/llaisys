#include "../model_base.hpp"

namespace llaisys::model {

class naive_session : public ModelSession {
public:
    ~naive_session() override = default;
    void init(std::vector<int64_t>& tokens, CacheHandle_t handle);
    const std::vector<int64_t>& tokens() const override;
    size_t seq_len() const override;
    size_t token_pos() const override;
    CacheHandle_t cache() const override { return cache_; }
    void append(int64_t next_token) override;

private:
    std::vector<int64_t> tokens_;
    size_t seq_len_ = 0;
    size_t token_pos_ = 0;
    CacheHandle_t cache_;
};

} // namespace llaisys::model
