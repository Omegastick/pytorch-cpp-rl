#pragma once

#include <vector>

#include <torch/torch.h>

namespace cpprl
{
// https://github.com/openai/baselines/blob/master/baselines/common/running_mean_std.py
class RunningMeanStdImpl : public torch::nn::Module
{
  private:
    torch::Tensor count, mean, variance;

    void update_from_moments(torch::Tensor batch_mean,
                             torch::Tensor batch_var,
                             int batch_count);

  public:
    explicit RunningMeanStdImpl(int size);
    RunningMeanStdImpl(std::vector<float> means, std::vector<float> variances);

    void update(torch::Tensor observation);

    inline int get_count() const { return static_cast<int>(count.item().toFloat()); }
    inline torch::Tensor get_mean() const { return mean.clone(); }
    inline torch::Tensor get_variance() const { return variance.clone(); }
    inline void set_count(int count) { this->count[0] = count + 1e-8; }
};
TORCH_MODULE(RunningMeanStd);
}