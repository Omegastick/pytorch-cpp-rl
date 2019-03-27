#include <torch/torch.h>

#include "cpprl/model/model_utils.h"
#include "third_party/doctest.h"

using namespace torch;

namespace cpprl
{
torch::Tensor FlattenImpl::forward(torch::Tensor x)
{
    return x.view({x.size(0), -1});
}

void init_weights(torch::OrderedDict<std::string, torch::Tensor> parameters, double gain)
{
    for (const auto &parameter : parameters)
    {
        if (parameter.key().find("bias") != std::string::npos)
        {
            nn::init::constant_(parameter.value(), gain);
        }
        else if (parameter.key().find("weight") != std::string::npos)
        {
            nn::init::orthogonal_(parameter.value());
        }
    }
}

TEST_CASE("Flatten")
{
    auto flatten = Flatten();

    SUBCASE("Flatten converts 3 dimensional vector to 2 dimensional")
    {
        auto input = torch::rand({5, 5, 5});
        auto output = flatten->forward(input);

        CHECK(output.size(0) == 5);
        CHECK(output.size(1) == 25);
    }

    SUBCASE("Flatten converts 5 dimensional vector to 2 dimensional")
    {
        auto input = torch::rand({2, 2, 2, 2, 2});
        auto output = flatten->forward(input);

        CHECK(output.size(0) == 2);
        CHECK(output.size(1) == 16);
    }

    SUBCASE("Flatten converts 1 dimensional vector to 2 dimensional")
    {
        auto input = torch::rand({10});
        auto output = flatten->forward(input);

        CHECK(output.size(0) == 10);
        CHECK(output.size(1) == 1);
    }
}

TEST_CASE("init_weights()")
{
    auto module = nn::Sequential(
        nn::Linear(5, 10),
        nn::Functional(torch::relu),
        nn::Linear(10, 8));

    init_weights(module->named_parameters(), 0);

    SUBCASE("Bias weights are initialized to 0")
    {
        for (const auto &parameter : module->named_parameters())
        {
            if (parameter.key().find("bias") != std::string::npos)
            {
                CHECK(parameter.value()[0].item().toDouble() == doctest::Approx(0));
            }
        }
    }
}
}