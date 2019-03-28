#include <torch/torch.h>

#include "cpprl/distributions/categorical.h"
#include "third_party/doctest.h"

namespace cpprl
{
Categorical::Categorical(const torch::Tensor *probs,
                         const torch::Tensor *logits)
{
    if ((probs == nullptr) == (logits == nullptr))
    {
        throw std::exception();
    }

    if (probs != nullptr)
    {
        if (probs->dim() < 1)
        {
            throw std::exception();
        }
        this->probs = *probs / probs->sum(-1, true);
    }
    else
    {
        if (logits->dim() < 1)
        {
            throw std::exception();
        }
        this->logits = *logits - logits->logsumexp(-1, true);
    }

    param = probs != nullptr ? *probs : *logits;
    num_events = param.size(-1);
    if (param.dim() > 1)
    {
        batch_shape = param.sizes().vec();
        batch_shape.resize(batch_shape.size() - 1);
    }
}

torch::Tensor Categorical::entropy()
{
    return torch::Tensor();
}

std::vector<long> Categorical::extended_shape(torch::IntArrayRef sample_shape)
{
    std::vector<long> output_shape;
    output_shape.insert(output_shape.end(),
                        sample_shape.begin(),
                        sample_shape.end());
    output_shape.insert(output_shape.end(),
                        batch_shape.begin(),
                        batch_shape.end());
    output_shape.insert(output_shape.end(),
                        event_shape.begin(),
                        event_shape.end());
    return output_shape;
}

torch::Tensor Categorical::log_prob(torch::Tensor /*value*/)
{
    return torch::Tensor();
}

torch::Tensor Categorical::sample(torch::IntArrayRef sample_shape)
{
    auto ext_sample_shape = extended_shape(sample_shape);
    auto param_shape = ext_sample_shape;
    param_shape.insert(param_shape.end(), {num_events});
    auto exp_probs = probs.expand(param_shape);
    torch::Tensor probs_2d;
    if (probs.dim() == 1 or probs.size(0) == 1)
    {
        probs_2d = exp_probs.view({-1, num_events});
    }
    else
    {
        probs_2d = exp_probs.contiguous().view({-1, num_events});
    }
    auto sample_2d = torch::multinomial(probs_2d, 1, true);
    return sample_2d.contiguous().view(ext_sample_shape);
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

    SUBCASE("Generated tensors are of the right shape")
    {
        float probabilities[] = {0.2, 0.2, 0.2, 0.2, 0.2};
        auto probabilities_tensor = torch::from_blob(probabilities, {5});
        auto dist = Categorical(&probabilities_tensor, nullptr);

        CHECK(dist.sample({20}).sizes().vec() == std::vector<long>{20});
        CHECK(dist.sample({2, 20}).sizes().vec() == std::vector<long>{2, 20});
        CHECK(dist.sample({1, 2, 3, 4, 5}).sizes().vec() == std::vector<long>{1, 2, 3, 4, 5});
    }

    SUBCASE("Multi-dimensional input probabilities are handled correctly")
    {
        SUBCASE("Generated tensors are of the right shape")
        {
            float probabilities[2][4] = {{0.5, 0.5, 0.0, 0.0}, {0.25, 0.25, 0.25, 0.25}};
            auto probabilities_tensor = torch::from_blob(probabilities, {2, 4});
            auto dist = Categorical(&probabilities_tensor, nullptr);

            CHECK(dist.sample({20}).sizes().vec() == std::vector<long>{20, 2});
            CHECK(dist.sample({10, 5}).sizes().vec() == std::vector<long>{10, 5, 2});
        }

        SUBCASE("Generated tensors have correct probabilities")
        {
            float probabilities[2][4] = {{0, 1, 0, 0}, {0, 0, 0, 1}};
            auto probabilities_tensor = torch::from_blob(probabilities, {2, 4});
            auto dist = Categorical(&probabilities_tensor, nullptr);

            auto output = dist.sample({20});
            auto sum = output.sum({0});

            CHECK(sum[0].item().toInt() == 20);
            CHECK(sum[1].item().toInt() == 60);
        }
    }
}
}