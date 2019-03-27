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
        batch_shape = param.sizes().slice(param.dim() - 1).vec();
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

torch::Tensor Categorical::sample(c10::IntArrayRef sample_shape)
{
    auto ext_sample_shape = extended_shape(sample_shape);
    ext_sample_shape.insert(ext_sample_shape.end(), {num_events});
    auto exp_probs = probs.expand(ext_sample_shape);
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
    return sample_2d.contiguous().view(sample_shape);
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