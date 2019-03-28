#pragma once

#include <torch/torch.h>

namespace cpprl
{
class Distribution
{
  public:
    virtual ~Distribution() = 0;

    virtual torch::Tensor sample(torch::IntArrayRef sample_shape) = 0;
    virtual torch::Tensor log_prob(torch::Tensor value) = 0;
    virtual torch::Tensor entropy() = 0;
};

inline Distribution::~Distribution() {}
}