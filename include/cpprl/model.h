#pragma once

#include <vector>
#include <memory>

#include <torch/torch.h>

using namespace torch;

namespace cpprl
{
class ActionSpace;
class Distribution;

class NNBase : nn::Module
{
  private:
    int hidden_size;
    bool recurrent;

  public:
    NNBase(bool recurrent,
           int recurrent_input_size,
           int hidden_size);

    int recurrent_hidden_state_size() const;

    inline bool is_recurrent() const { return recurrent; }
};

class Policy : nn::Module
{
  private:
    std::unique_ptr<NNBase> base;
    std::unique_ptr<Distribution> dist;

    std::vector<torch::Tensor> forward_gru(torch::Tensor &x,
                                           torch::Tensor &hxs,
                                           torch::Tensor &masks);

  public:
    Policy(IntArrayRef observation_shape,
           ActionSpace action_space,
           NNBase &base);

    std::vector<torch::Tensor> act(torch::Tensor &inputs,
                                   torch::Tensor &rnn_hxs,
                                   torch::Tensor &masks) const;
    std::vector<torch::Tensor> evaluate_actions(torch::Tensor &inputs,
                                                torch::Tensor &rnn_hxs,
                                                torch::Tensor &masks,
                                                torch::Tensor &actions) const;
    torch::Tensor get_values(torch::Tensor &inputs,
                             torch::Tensor &rnn_hxs,
                             torch::Tensor &masks) const;

    inline bool is_recurrent() const { return base->is_recurrent(); }
    inline int recurrent_hidden_state_size() const
    {
        return base->recurrent_hidden_state_size();
    }
};

class CNNBase : NNBase
{
  private:
    nn::Module main;
    nn::Module critic_linear;

  public:
    CNNBase(int num_inputs, bool recurrent = false, int hidden_size = 512);

    std::vector<torch::Tensor> forward(torch::Tensor &inputs,
                                       torch::Tensor &hxs,
                                       torch::Tensor &masks) const;
};

class MLPBase : NNBase
{
  private:
    nn::Module actor;
    nn::Module critic;
    nn::Module critic_linear;

  public:
    MLPBase(int num_inputs, bool recurrent = false, int hidden_size = 64);

    std::vector<torch::Tensor> forward(torch::Tensor &inputs,
                                       torch::Tensor &hxs,
                                       torch::Tensor &masks) const;
};
}