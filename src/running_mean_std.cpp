#include <torch/torch.h>

#include "cpprl/running_mean_std.h"
#include "third_party/doctest.h"

namespace cpprl
{
RunningMeanStdImpl::RunningMeanStdImpl(int size)
    : count(register_buffer("count", torch::full({1}, 1e-4, torch::kFloat))),
      mean(register_buffer("mean", torch::zeros({size}))),
      variance(register_buffer("variance", torch::ones({size}))) {}

RunningMeanStdImpl::RunningMeanStdImpl(std::vector<float> means, std::vector<float> variances)
    : count(register_buffer("count", torch::full({1}, 1e-4, torch::kFloat))),
      mean(register_buffer("mean", torch::from_blob(means.data(), {static_cast<long>(means.size())})
                                       .clone())),
      variance(register_buffer("variance", torch::from_blob(variances.data(), {static_cast<long>(variances.size())})
                                               .clone())) {}

void RunningMeanStdImpl::update(torch::Tensor observation)
{
    observation = observation.reshape({-1, mean.size(0)});
    auto batch_mean = observation.mean(0);
    auto batch_var = observation.var(0, false, false);
    auto batch_count = observation.size(0);

    update_from_moments(batch_mean, batch_var, batch_count);
}

void RunningMeanStdImpl::update_from_moments(torch::Tensor batch_mean,
                                             torch::Tensor batch_var,
                                             int batch_count)
{
    auto delta = batch_mean - mean;
    auto total_count = count + batch_count;

    mean.copy_(mean + delta * batch_count / total_count);
    auto m_a = variance * count;
    auto m_b = batch_var * batch_count;
    auto m2 = m_a + m_b + torch::pow(delta, 2) * count * batch_count / total_count;
    variance.copy_(m2 / total_count);
    count.copy_(total_count);
}

TEST_CASE("RunningMeanStd")
{
    SUBCASE("Calculates mean and variance correctly")
    {
        RunningMeanStd rms(5);
        auto observations = torch::rand({3, 5});
        rms->update(observations[0]);
        rms->update(observations[1]);
        rms->update(observations[2]);

        auto expected_mean = observations.mean(0);
        auto expected_variance = observations.var(0, false, false);

        auto actual_mean = rms->get_mean();
        auto actual_variance = rms->get_variance();

        for (int i = 0; i < 5; ++i)
        {
            DOCTEST_CHECK(expected_mean[i].item().toFloat() ==
                          doctest::Approx(actual_mean[i].item().toFloat())
                              .epsilon(0.001));
            DOCTEST_CHECK(expected_variance[i].item().toFloat() ==
                          doctest::Approx(actual_variance[i].item().toFloat())
                              .epsilon(0.001));
        }
    }

    SUBCASE("Loads mean and variance from constructor correctly")
    {
        RunningMeanStd rms(std::vector<float>{1, 2, 3}, std::vector<float>{4, 5, 6});

        auto mean = rms->get_mean();
        auto variance = rms->get_variance();
        DOCTEST_CHECK(mean[0].item().toFloat() == doctest::Approx(1));
        DOCTEST_CHECK(mean[1].item().toFloat() == doctest::Approx(2));
        DOCTEST_CHECK(mean[2].item().toFloat() == doctest::Approx(3));
        DOCTEST_CHECK(variance[0].item().toFloat() == doctest::Approx(4));
        DOCTEST_CHECK(variance[1].item().toFloat() == doctest::Approx(5));
        DOCTEST_CHECK(variance[2].item().toFloat() == doctest::Approx(6));
    }
}
}
