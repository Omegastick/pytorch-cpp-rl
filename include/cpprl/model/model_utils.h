#pragma once

#include <vector>

#include <torch/torch.h>

using namespace torch;

namespace cpprl
{
struct FlattenImpl : nn::Module
{
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(Flatten);

void init_weights(torch::OrderedDict<std::string, torch::Tensor> parameters,
                  double gain);
}