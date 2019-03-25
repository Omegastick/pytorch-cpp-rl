#include <iostream>

#include <torch/torch.h>

#include "cpprl/model.h"

int main(int /*argc*/, char * /*argv*/ [])
{
    torch::Tensor x = torch::rand({100, 100}, c10::TensorOptions());
    torch::Tensor y = torch::rand({100, 100});
    std::cout << x * y << std::endl;

    return 0;
}