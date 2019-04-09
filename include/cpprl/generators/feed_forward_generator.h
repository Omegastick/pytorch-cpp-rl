#pragma once

#include <torch/torch.h>

#include "cpprl/generators/generator.h"

namespace cpprl
{
class FeedForwardGenerator : Generator
{
  private:
    torch::Tensor observations, hidden_states, actions, value_predictions,
        returns, masks, old_action_log_probs, advantages;

  public:
    FeedForwardGenerator(int num_mini_batch,
                         torch::Tensor observations,
                         torch::Tensor hidden_states,
                         torch::Tensor actions,
                         torch::Tensor value_predictions,
                         torch::Tensor returns,
                         torch::Tensor masks,
                         torch::Tensor old_action_log_probs,
                         torch::Tensor advantages);

    virtual bool done() const;
    virtual MiniBatch next();
};
}