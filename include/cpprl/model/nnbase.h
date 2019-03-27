#pragma once

#include <vector>

#include <torch/torch.h>

using namespace torch;

namespace cpprl
{
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
}