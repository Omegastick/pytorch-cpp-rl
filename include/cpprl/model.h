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
    nn::GRU gru;

  public:
    NNBase(bool recurrent,
           unsigned int recurrent_input_size,
           unsigned int hidden_size);

    std::vector<torch::Tensor> forward_gru(torch::Tensor x,
                                           torch::Tensor hxs,
                                           torch::Tensor masks);
    unsigned int get_hidden_size() const;

    inline bool is_recurrent() const { return recurrent; }
};

class PolicyImpl : public nn::Module
{
  private:
    std::shared_ptr<NNBase> base;
    // std::unique_ptr<Distribution> dist;

    std::vector<torch::Tensor> forward_gru(torch::Tensor x,
                                           torch::Tensor hxs,
                                           torch::Tensor masks);

  public:
    PolicyImpl(c10::IntArrayRef observation_shape,
               ActionSpace action_space,
               std::shared_ptr<NNBase> base);

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

class CnnBase : public NNBase
{
  private:
    nn::Sequential main;
    nn::Sequential critic_linear;

  public:
    CnnBase(unsigned int num_inputs,
            bool recurrent = false,
            unsigned int hidden_size = 512);

    std::vector<torch::Tensor> forward(torch::Tensor inputs,
                                       torch::Tensor hxs,
                                       torch::Tensor masks);
};

class MlpBase : public NNBase
{
  private:
    nn::Sequential actor;
    nn::Sequential critic;
    nn::Linear critic_linear;

  public:
    MlpBase(unsigned int num_inputs,
            bool recurrent = false,
            unsigned int hidden_size = 64);

    std::vector<torch::Tensor> forward(torch::Tensor inputs,
                                       torch::Tensor hxs,
                                       torch::Tensor masks);
};

struct FlattenImpl : nn::Module
{
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(Flatten);

void init_weights(torch::OrderedDict<std::string, torch::Tensor> parameters, double gain);
}