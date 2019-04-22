#include <ATen/core/Reduction.h>
#include <c10/util/ArrayRef.h>
#include <spdlog/spdlog.h>
#include <torch/torch.h>

#include "cpprl/distributions/bernoulli.h"
#include "third_party/doctest.h"

namespace cpprl
{
Bernoulli::Bernoulli(const torch::Tensor *probs,
                     const torch::Tensor *logits)
{
    if ((probs == nullptr) == (logits == nullptr))
    {
        spdlog::error("Either probs or logits is required, but not both");
        throw std::exception();
    }

    if (probs != nullptr)
    {
        if (probs->dim() < 1)
        {
            throw std::exception();
        }
        this->probs = *probs;
        // 1.21e-7 is used as the epsilon to match PyTorch's Python results as closely
        // as possible
        auto clamped_probs = this->probs.clamp(1.21e-7, 1. - 1.21e-7);
        this->logits = torch::log(clamped_probs) - torch::log1p(-clamped_probs);
    }
    else
    {
        if (logits->dim() < 1)
        {
            throw std::exception();
        }
        this->logits = *logits;
        this->probs = torch::sigmoid(*logits);
    }

    param = probs != nullptr ? *probs : *logits;
    num_events = param.size(-1);
    batch_shape = param.sizes().vec();
}

torch::Tensor Bernoulli::entropy()
{
    return torch::binary_cross_entropy_with_logits(logits, probs, torch::Tensor(), torch::Tensor(), Reduction::None);
}

std::vector<int64_t> Bernoulli::extended_shape(c10::ArrayRef<int64_t> sample_shape)
{
    std::vector<int64_t> output_shape;
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

torch::Tensor Bernoulli::log_prob(torch::Tensor value)
{
    auto broadcasted_tensors = torch::broadcast_tensors({logits, value});
    return -torch::binary_cross_entropy_with_logits(broadcasted_tensors[0], broadcasted_tensors[1], torch::Tensor(), torch::Tensor(), Reduction::None);
}

torch::Tensor Bernoulli::sample(c10::ArrayRef<int64_t> sample_shape)
{
    auto ext_sample_shape = extended_shape(sample_shape);
    torch::NoGradGuard no_grad_guard;
    return torch::bernoulli(probs.expand(ext_sample_shape));
}

TEST_CASE("Bernoulli")
{
    SUBCASE("Throws when provided both probs and logits")
    {
        auto tensor = torch::Tensor();
        CHECK_THROWS(Bernoulli(&tensor, &tensor));
    }

    SUBCASE("Sampled numbers are in the right range")
    {
        float probabilities[] = {0.2, 0.2, 0.2, 0.2, 0.2};
        auto probabilities_tensor = torch::from_blob(probabilities, {5});
        auto dist = Bernoulli(&probabilities_tensor, nullptr);

        auto output = dist.sample({100});
        auto more_than_1 = output > 1;
        auto less_than_0 = output < 0;
        CHECK(!more_than_1.any().item().toInt());
        CHECK(!less_than_0.any().item().toInt());
    }

    SUBCASE("Sampled tensors are of the right shape")
    {
        float probabilities[] = {0.2, 0.2, 0.2, 0.2, 0.2};
        auto probabilities_tensor = torch::from_blob(probabilities, {5});
        auto dist = Bernoulli(&probabilities_tensor, nullptr);

        CHECK(dist.sample({20}).sizes().vec() == std::vector<int64_t>{20, 5});
        CHECK(dist.sample({2, 20}).sizes().vec() == std::vector<int64_t>{2, 20, 5});
        CHECK(dist.sample({1, 2, 3, 4}).sizes().vec() == std::vector<int64_t>{1, 2, 3, 4, 5});
    }

    SUBCASE("Multi-dimensional input probabilities are handled correctly")
    {
        SUBCASE("Sampled tensors are of the right shape")
        {
            float probabilities[2][4] = {{0.5, 0.5, 0.0, 0.0},
                                         {0.25, 0.25, 0.25, 0.25}};
            auto probabilities_tensor = torch::from_blob(probabilities, {2, 4});
            auto dist = Bernoulli(&probabilities_tensor, nullptr);

            CHECK(dist.sample({20}).sizes().vec() == std::vector<int64_t>{20, 2, 4});
            CHECK(dist.sample({10, 5}).sizes().vec() == std::vector<int64_t>{10, 5, 2, 4});
        }
    }

    SUBCASE("entropy()")
    {
        float probabilities[2][2] = {{0.5, 0.0},
                                     {0.25, 0.25}};
        auto probabilities_tensor = torch::from_blob(probabilities, {2, 2});
        auto dist = Bernoulli(&probabilities_tensor, nullptr);

        auto entropies = dist.entropy();

        SUBCASE("Returns correct values")
        {
            CHECK(entropies[0][0].item().toDouble() ==
                  doctest::Approx(0.6931).epsilon(1e-3));
            CHECK(entropies[0][1].item().toDouble() ==
                  doctest::Approx(0.0000).epsilon(1e-3));
            CHECK(entropies[1][0].item().toDouble() ==
                  doctest::Approx(0.5623).epsilon(1e-3));
            CHECK(entropies[1][1].item().toDouble() ==
                  doctest::Approx(0.5623).epsilon(1e-3));
        }

        SUBCASE("Output tensor is the correct size")
        {
            CHECK(entropies.sizes().vec() == std::vector<int64_t>{2, 2});
        }
    }

    SUBCASE("log_prob()")
    {
        float probabilities[2][2] = {{0.5, 0.0},
                                     {0.25, 0.25}};
        auto probabilities_tensor = torch::from_blob(probabilities, {2, 2});
        auto dist = Bernoulli(&probabilities_tensor, nullptr);

        float actions[2][2] = {{1, 0},
                               {1, 0}};
        auto actions_tensor = torch::from_blob(actions, {2, 2});
        auto log_probs = dist.log_prob(actions_tensor);

        INFO(log_probs << "\n");
        SUBCASE("Returns correct values")
        {
            CHECK(log_probs[0][0].item().toDouble() ==
                  doctest::Approx(-0.6931).epsilon(1e-3));
            CHECK(log_probs[0][1].item().toDouble() ==
                  doctest::Approx(0.0000).epsilon(1e-3));
            CHECK(log_probs[1][0].item().toDouble() ==
                  doctest::Approx(-1.3863).epsilon(1e-3));
            CHECK(log_probs[1][1].item().toDouble() ==
                  doctest::Approx(-0.2876).epsilon(1e-3));
        }

        SUBCASE("Output tensor is correct size")
        {
            CHECK(log_probs.sizes().vec() == std::vector<int64_t>{2, 2});
        }
    }
}
}