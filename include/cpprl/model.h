#pragma once

#include <vector>
#include <memory>

#include <torch/torch.h>

using namespace torch;

namespace cpprl
{
class ActionSpace;
class Distribution;

class NNBase : public nn::Module
{
  private:
    bool recurrent;
    unsigned int hidden_size;

  public:
    NNBase(bool recurrent,
           unsigned int recurrent_input_size,
           unsigned int hidden_size);

    int recurrent_hidden_state_size() const;

    inline bool is_recurrent() const { return recurrent; }
};

class PolicyImpl : public nn::Module
{
  private:
    std::shared_ptr<NNBase> base;
    // std::unique_ptr<Distribution> dist;

    std::vector<torch::Tensor> forward_gru(torch::Tensor &x,
                                           torch::Tensor &hxs,
                                           torch::Tensor &masks);

  public:
    PolicyImpl(c10::IntArrayRef observation_shape,
               ActionSpace action_space,
               std::shared_ptr<NNBase> base);

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
TORCH_MODULE(Policy);

class CnnBase : public NNBase
{
  private:
    std::unique_ptr<nn::Module> main;
    std::unique_ptr<nn::Module> critic_linear;

  public:
    CnnBase(unsigned int num_inputs,
            bool recurrent = false,
            unsigned int hidden_size = 512);

    std::vector<torch::Tensor> forward(torch::Tensor &inputs,
                                       torch::Tensor &hxs,
                                       torch::Tensor &masks) const;
};

class MlpBase : public NNBase
{
  private:
    std::unique_ptr<nn::Module> actor;
    std::unique_ptr<nn::Module> critic;
    std::unique_ptr<nn::Module> critic_linear;

  public:
    MlpBase(unsigned int num_inputs,
            bool recurrent = false,
            unsigned int hidden_size = 64);

    std::vector<torch::Tensor> forward(torch::Tensor &inputs,
                                       torch::Tensor &hxs,
                                       torch::Tensor &masks) const;
};
}