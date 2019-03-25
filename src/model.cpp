#include <iostream>

#include <torch/torch.h>

#include "cpprl/model.h"

int main()
{
    torch::Tensor x = torch::rand({100, 100});
    torch::Tensor y = torch::rand({100, 100});
    std::cout << x * y << std::endl;

    return 0;
}