#include "../../layer/Qwen2/Decoder.hpp"
#include "src/model/model_base.hpp"
namespace llaisys::model {
// 新的模型继承自ModelBase
class Model_Qwen2 : public ModelBase {
public:
    explicit Model_Qwen2(const meta_data &config, const DeviceSpec &device,
                         const ParallelSpec &parallel)
        : ModelBase(config, device, parallel),
          bos_token_id(static_cast<int64_t>(config.bos_token_id)),
          eos_token_id(static_cast<int64_t>(config.eos_token_id)) {}
    void loadWeights(WeightsMap &weights) override;
    void unloadWeights() override;
    std::unique_ptr<ModelSession> createSession() override;
    void resetSession(ModelSession &session) override;

    InferenceOutputs inferStep(session_t session) override;
    std::vector<int64_t> inferDialog(std::vector<int64_t> &tokens,
                                     size_t max_steps = 128);
    void destroy();
    void show() override;
    const llaisys::Qwen2::qwen2_weights &weights() const { return qwen2_weights; }
    static model_t create(WeightsMap &weights, const meta_data &meta_data,
                          const DeviceSpec &device, const ParallelSpec &parallel);

private:
    WeightsMap weights_;
    // 推理初始化功能，实现token_idx->input_embedding以及KV_cache资源分配
    tensor_t inferInit(session_t session);
    llaisys::Qwen2::qwen2_weights qwen2_weights;
    void parseWeight();
    int64_t bos_token_id;
    int64_t eos_token_id;
};
} // namespace llaisys::model