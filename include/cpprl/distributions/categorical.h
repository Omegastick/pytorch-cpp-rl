#pragma once

#include <torch/torch.h>

#include "cpprl/distributions/distribution.h"

namespace cpprl
{
class Categorical : public Distribution
{
  private:
    torch::Tensor probs;
    torch::Tensor logits;
    std::vector<long> batch_shape;
    std::vector<long> event_shape;
    torch::Tensor param;
    int num_events;

    std::vector<long> extended_shape(torch::IntArrayRef sample_shape);

  public:
    Categorical(const torch::Tensor *probs, const torch::Tensor *logits);

    torch::Tensor sample(torch::IntArrayRef sample_shape = {});
    torch::Tensor log_prob(torch::Tensor value);
    torch::Tensor entropy();
};
}