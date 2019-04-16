#include <memory>

#include <torch/torch.h>

#include "cpprl/model/output_layers.h"
#include "cpprl/model/model_utils.h"
#include "cpprl/distributions/distribution.h"
#include "cpprl/distributions/categorical.h"
#include "cpprl/distributions/normal.h"
#include "third_party/doctest.h"

using namespace torch;

namespace cpprl
{
CategoricalOutput::CategoricalOutput(unsigned int num_inputs,
                                     unsigned int num_outputs)
    : linear(num_inputs, num_outputs)
{
    register_module("linear", linear);
    init_weights(linear->named_parameters(), 0.01, 0);
}

std::unique_ptr<Distribution> CategoricalOutput::forward(torch::Tensor x)
{
    x = linear(x);
    return std::make_unique<Categorical>(nullptr, &x);
}

NormalOutput::NormalOutput(unsigned int num_inputs,
                           unsigned int num_outputs)
    : linear_loc(num_inputs, num_outputs),
      linear_scale(num_inputs, num_outputs)
{
    register_module("linear_loc", linear_loc);
    register_module("linear_scale", linear_scale);
    init_weights(linear_loc->named_parameters(), 0.01, 0);
    init_weights(linear_scale->named_parameters(), 0.1, 0);
}

std::unique_ptr<Distribution> NormalOutput::forward(torch::Tensor x)
{
    auto loc = linear_loc(x);
    auto scale = torch::relu(linear_scale(x));
    return std::make_unique<Normal>(loc, scale);
}

TEST_CASE("CategoricalOutput")
{
    auto output_layer = CategoricalOutput(3, 5);

    SUBCASE("Output distribution has correct output shape")
    {
        float input_array[2][3] = {{0, 1, 2}, {3, 4, 5}};
        auto input_tensor = torch::from_blob(input_array,
                                             {2, 3},
                                             TensorOptions(torch::kFloat));
        auto dist = output_layer.forward(input_tensor);

        auto output = dist->sample();

        CHECK(output.sizes().vec() == std::vector<int64_t>{2});
    }
}

TEST_CASE("NormalOutput")
{
    auto output_layer = NormalOutput(3, 5);

    SUBCASE("Output distribution has correct output shape")
    {
        float input_array[2][3] = {{0, 1, 2}, {3, 4, 5}};
        auto input_tensor = torch::from_blob(input_array,
                                             {2, 3},
                                             TensorOptions(torch::kFloat));
        auto dist = output_layer.forward(input_tensor);

        auto output = dist->sample();

        CHECK(output.sizes().vec() == std::vector<int64_t>{2, 5});
    }
}
}