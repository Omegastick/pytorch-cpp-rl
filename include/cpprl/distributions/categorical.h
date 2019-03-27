#pragma once

#include <torch/torch.h>

namespace cpprl
{
class Categorical
{
  private:
    torch::Tensor probs;
    torch::Tensor logits;

  public:
    Categorical(const torch::Tensor *probs, const torch::Tensor *logits);

    torch::Tensor sample(c10::IntArrayRef sample_shape);
    torch::Tensor log_prob(torch::Tensor value);
    torch::Tensor entropy();
};
}