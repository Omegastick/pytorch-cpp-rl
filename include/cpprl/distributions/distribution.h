#pragma once

#include <torch/torch.h>

namespace cpprl
{
class Distribution
{
  protected:
    std::vector<int64_t> batch_shape, event_shape;

    std::vector<int64_t> extended_shape(c10::ArrayRef<int64_t> sample_shape);

  public:
    virtual ~Distribution() = 0;

    virtual torch::Tensor entropy() = 0;
    virtual torch::Tensor log_prob(torch::Tensor value) = 0;
    virtual torch::Tensor sample(c10::ArrayRef<int64_t> sample_shape = {}) = 0;
};

inline Distribution::~Distribution() {}
}