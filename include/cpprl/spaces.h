#pragma once

#include <string>

#include <torch/torch.h>

namespace cpprl
{
struct ActionSpace
{
    std::string type;
    std::vector<int64_t> shape;
};
}