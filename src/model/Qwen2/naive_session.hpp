#include "../model_base.hpp"
// 会话类，管理会话自己的状态
namespace llaisys::model {
class naive_session : public ModelSession {
public:
    ~naive_session() override = default;
    void init(const llaisys::model::meta_data &meta_data, std::vector<int64_t> &tokens);
    const std::vector<int64_t> &tokens() const override;
    size_t seq_len() const override;
    size_t token_pos() const override;
    KVcache_t kv_cache() const override;
    static session_t create(const llaisys::model::meta_data &meta_data,
                            std::vector<int64_t> &tokens) {
        auto session_t = std::make_shared<naive_session>();
        session_t->init(meta_data, tokens);
        return session_t;
    }
    void append(int64_t next_token) override;

private:
    std::vector<int64_t> tokens_;
    size_t seq_len_;
    size_t token_pos_;
    KVcache_t kv_cache_;
};
} // namespace llaisys::model