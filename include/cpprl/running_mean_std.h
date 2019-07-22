#pragma once

#include <vector>

#include <torch/torch.h>

namespace SingularityTrainer
{
// https://github.com/openai/baselines/blob/master/baselines/common/running_mean_std.py
class RunningMeanStd
{
  private:
    float count;
    torch::Tensor mean, variance;

    void update_from_moments(torch::Tensor batch_mean,
                             torch::Tensor batch_var,
                             int batch_count);

  public:
    explicit RunningMeanStd(int size);
    RunningMeanStd(std::vector<float> means, std::vector<float> variances);

    void update(torch::Tensor observation);

    inline int get_count() const { return static_cast<int>(count); }
    inline torch::Tensor get_mean() const { return mean.clone(); }
    inline torch::Tensor get_variance() const { return variance.clone(); }
    inline void set_count(int count) { this->count = count + 1e-8; }
};
}