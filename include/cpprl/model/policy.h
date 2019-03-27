#pragma once

#include <vector>
#include <memory>

#include <torch/torch.h>

#include "cpprl/model/nnbase.h"

using namespace torch;

namespace cpprl
{
class ActionSpace;

class PolicyImpl : public nn::Module
{
  private:
    std::shared_ptr<NNBase> base;
    // std::unique_ptr<Distribution> dist;

    std::vector<torch::Tensor> forward_gru(torch::Tensor x,
                                           torch::Tensor hxs,
                                           torch::Tensor masks);

  public:
    PolicyImpl(ActionSpace action_space, std::shared_ptr<NNBase> base);

    std::vector<torch::Tensor> act(torch::Tensor inputs,
                                   torch::Tensor rnn_hxs,
                                   torch::Tensor masks);
    std::vector<torch::Tensor> evaluate_actions(torch::Tensor inputs,
                                                torch::Tensor rnn_hxs,
                                                torch::Tensor masks,
                                                torch::Tensor actions);
    torch::Tensor get_values(torch::Tensor inputs,
                             torch::Tensor rnn_hxs,
                             torch::Tensor masks);

    inline bool is_recurrent() const { return base->is_recurrent(); }
    inline unsigned int get_hidden_size() const
    {
        return base->get_hidden_size();
    }
};
TORCH_MODULE(Policy);
}