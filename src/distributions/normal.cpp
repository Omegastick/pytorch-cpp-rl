#define _USE_MATH_DEFINES
#include <math.h>
#include <cmath>
#include <limits>

#include <c10/util/ArrayRef.h>
#include <torch/torch.h>

#include "cpprl/distributions/normal.h"
#include "third_party/doctest.h"

namespace cpprl
{
Normal::Normal(const torch::Tensor loc,
               const torch::Tensor scale)
{
    auto broadcasted_tensors = torch::broadcast_tensors({loc, scale});
    this->loc = broadcasted_tensors[0];
    this->scale = broadcasted_tensors[1];
    batch_shape = this->loc.sizes().vec();
    event_shape = {};
}

torch::Tensor Normal::entropy()
{
    return (0.5 + 0.5 * std::log(2 * M_PI) + torch::log(scale)).sum(-1);
}

std::vector<int64_t> Normal::extended_shape(c10::ArrayRef<int64_t> sample_shape)
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

torch::Tensor Normal::log_prob(torch::Tensor value)
{
    auto variance = scale.pow(2);
    auto log_scale = scale.log();
    return (-(value - loc).pow(2) /
                (2 * variance) -
            log_scale -
            std::log(std::sqrt(2 * M_PI)));
}

torch::Tensor Normal::sample(c10::ArrayRef<int64_t> sample_shape)
{
    auto shape = extended_shape(sample_shape);
    auto no_grad_guard = torch::NoGradGuard();
    return torch::normal(loc.expand(shape), scale.expand(shape));
}

TEST_CASE("Normal")
{
    float locs_array[] = {0, 1, 2, 3, 4, 5};
    float scales_array[] = {5, 4, 3, 2, 1, 0};
    auto locs = torch::from_blob(locs_array, {2, 3});
    auto scales = torch::from_blob(scales_array, {2, 3});
    auto dist = Normal(locs, scales);

    SUBCASE("Sampled tensors have correct shape")
    {
        CHECK(dist.sample().sizes().vec() == std::vector<int64_t>{2, 3});
        CHECK(dist.sample({20}).sizes().vec() == std::vector<int64_t>{20, 2, 3});
        CHECK(dist.sample({2, 20}).sizes().vec() == std::vector<int64_t>{2, 20, 2, 3});
        CHECK(dist.sample({1, 2, 3, 4, 5}).sizes().vec() == std::vector<int64_t>{1, 2, 3, 4, 5, 2, 3});
    }

    SUBCASE("entropy()")
    {
        auto entropies = dist.entropy();

        SUBCASE("Returns correct values")
        {
            INFO("Entropies: \n"
                 << entropies);

            CHECK(entropies[0].item().toDouble() ==
                  doctest::Approx(8.3512).epsilon(1e-3));
            CHECK(entropies[1].item().toDouble() ==
                  -std::numeric_limits<float>::infinity());
        }

        SUBCASE("Output tensor is the correct size")
        {
            CHECK(entropies.sizes().vec() == std::vector<int64_t>{2});
        }
    }

    SUBCASE("log_prob()")
    {
        float actions[2][3] = {{0, 1, 2},
                               {0, 1, 2}};
        auto actions_tensor = torch::from_blob(actions, {2, 3});
        auto log_probs = dist.log_prob(actions_tensor);

        INFO(log_probs << "\n");
        SUBCASE("Returns correct values")
        {
            CHECK(log_probs[0][0].item().toDouble() ==
                  doctest::Approx(-2.5284).epsilon(1e-3));
            CHECK(log_probs[0][1].item().toDouble() ==
                  doctest::Approx(-2.3052).epsilon(1e-3));
            CHECK(log_probs[0][2].item().toDouble() ==
                  doctest::Approx(-2.0176).epsilon(1e-3));
            CHECK(log_probs[1][0].item().toDouble() ==
                  doctest::Approx(-2.7371).epsilon(1e-3));
            CHECK(log_probs[1][1].item().toDouble() ==
                  doctest::Approx(-5.4189).epsilon(1e-3));
            CHECK(std::isnan(log_probs[1][2].item().toDouble()));
        }

        SUBCASE("Output tensor is correct size")
        {
            CHECK(log_probs.sizes().vec() == std::vector<int64_t>{2, 3});
        }
    }
}
}