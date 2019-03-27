#include <torch/torch.h>

#include "cpprl/distributions/categorical.h"
#include "third_party/doctest.h"

namespace cpprl
{
Categorical::Categorical(const torch::Tensor * /*probs*/,
                         const torch::Tensor * /*logits*/) {}

torch::Tensor Categorical::sample(c10::IntArrayRef /*sample_shape*/)
{
    return torch::Tensor();
}
torch::Tensor Categorical::log_prob(torch::Tensor /*value*/)
{
    return torch::Tensor();
}
torch::Tensor Categorical::entropy()
{
    return torch::Tensor();
}

TEST_CASE("Categorical")
{
    SUBCASE("Throws when provided both probs and logits")
    {
        auto tensor = torch::Tensor();
        CHECK_THROWS(Categorical(&tensor, &tensor));
    }

    SUBCASE("Generated numbers are in the right range")
    {
        float probabilities[] = {0.2, 0.2, 0.2, 0.2, 0.2};
        auto probabilities_tensor = torch::from_blob(probabilities, {5});
        auto dist = Categorical(&probabilities_tensor, nullptr);

        auto output = dist.sample({100});
        auto more_than_4 = output > 4;
        auto less_than_0 = output < 0;
        CHECK(!more_than_4.any().item().toBool());
        CHECK(!less_than_0.any().item().toBool());
    }
}
}