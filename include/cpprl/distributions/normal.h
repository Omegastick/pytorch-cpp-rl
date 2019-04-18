#pragma once

#include <c10/util/ArrayRef.h>
#include <torch/torch.h>

#include "cpprl/distributions/distribution.h"

namespace cpprl
{
class Normal : public Distribution
{
  private:
    torch::Tensor loc, scale;
    std::vector<int64_t> batch_shape, event_shape;

    std::vector<int64_t> extended_shape(c10::ArrayRef<int64_t> sample_shape);

  public:
    Normal(const torch::Tensor loc, const torch::Tensor scale);

    torch::Tensor entropy();
    torch::Tensor log_prob(torch::Tensor value);
    torch::Tensor sample(c10::ArrayRef<int64_t> sample_shape = {});

    inline torch::Tensor get_loc() { return loc; }
    inline torch::Tensor get_scale() { return scale; }
};
}