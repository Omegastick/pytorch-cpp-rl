#pragma once

#include <vector>

#include <torch/torch.h>

#include "cpprl/running_mean_std.h"

namespace SingularityTrainer
{
class ObservationNormalizer;

class ObservationNormalizerImpl : public torch::nn::Module
{
  private:
    torch::Tensor clip;
    RunningMeanStd rms;

  public:
    explicit ObservationNormalizerImpl(int size, float clip = 10.);
    ObservationNormalizerImpl(const std::vector<float> &means,
                              const std::vector<float> &variances,
                              float clip = 10.);
    explicit ObservationNormalizerImpl(const std::vector<ObservationNormalizer> &others);

    torch::Tensor process_observation(torch::Tensor observation);
    std::vector<float> get_mean() const;
    std::vector<float> get_variance() const;
    void update(torch::Tensor observations);

    inline float get_clip_value() const { return clip.item().toFloat(); }
    inline int get_step_count() const { return rms->get_count(); }
};
TORCH_MODULE(ObservationNormalizer);
}