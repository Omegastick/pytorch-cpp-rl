#pragma once

#include <string>

#include <torch/torch.h>

namespace cpprl
{
struct ActionSpace
{
    std::string type;
    c10::IntArrayRef shape;
};
}