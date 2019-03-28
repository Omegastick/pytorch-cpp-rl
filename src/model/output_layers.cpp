#include <memory>

#include <torch/torch.h>

#include "cpprl/model/output_layers.h"
#include "cpprl/distributions/distribution.h"
#include "third_party/doctest.h"

using namespace torch;

namespace cpprl
{
CategoricalOutput::CategoricalOutput(unsigned int /*num_inputs*/,
                                     unsigned int /*num_outputs*/)
    : linear(nullptr) {}

std::unique_ptr<Distribution> CategoricalOutput::forward(torch::Tensor /*x*/)
{
    return std::unique_ptr<Distribution>();
}

TEST_CASE("CategoricalOutput")
{
    auto output_layer = CategoricalOutput(3, 5);

    SUBCASE("Output distribution has correct output shape")
    {
        float input_array[2][3] = {{0, 1, 2}, {3, 4, 5}};
        auto input_tensor = torch::from_blob(input_array,
                                             {2, 3},
                                             TensorOptions(torch::kLong));
        auto dist = output_layer.forward(input_tensor);

        auto output = dist->sample({1});

        CHECK(output.sizes().vec() == std::vector<long>{2});
    }
}
}