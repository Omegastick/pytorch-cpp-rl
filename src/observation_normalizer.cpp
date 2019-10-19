#include <torch/torch.h>

#include "cpprl/observation_normalizer.h"
#include "cpprl/running_mean_std.h"
#include "third_party/doctest.h"

namespace cpprl
{
ObservationNormalizerImpl::ObservationNormalizerImpl(int size, float clip)
    : clip(register_buffer("clip", torch::full({1}, clip, torch::kFloat))),
      rms(register_module("rms", RunningMeanStd(size))) {}

ObservationNormalizerImpl::ObservationNormalizerImpl(const std::vector<float> &means,
                                                     const std::vector<float> &variances,
                                                     float clip)
    : clip(register_buffer("clip", torch::full({1}, clip, torch::kFloat))),
      rms(register_module("rms", RunningMeanStd(means, variances))) {}

ObservationNormalizerImpl::ObservationNormalizerImpl(const std::vector<ObservationNormalizer> &others)
    : clip(register_buffer("clip", torch::zeros({1}, torch::kFloat))),
      rms(register_module("rms", RunningMeanStd(1)))
{
    // Calculate mean clip
    for (const auto &other : others)
    {
        clip += other->get_clip_value();
    }
    clip[0] = clip[0] / static_cast<float>(others.size());

    // Calculate mean mean
    std::vector<float> mean_means(others[0]->get_mean().size(), 0);
    for (const auto &other : others)
    {
        auto other_mean = other->get_mean();
        for (unsigned int i = 0; i < mean_means.size(); ++i)
        {
            mean_means[i] += other_mean[i];
        }
    }
    for (auto &mean : mean_means)
    {
        mean /= others.size();
    }

    // Calculate mean variances
    std::vector<float> mean_variances(others[0]->get_variance().size(), 0);
    for (const auto &other : others)
    {
        auto other_variances = other->get_variance();
        for (unsigned int i = 0; i < mean_variances.size(); ++i)
        {
            mean_variances[i] += other_variances[i];
        }
    }
    for (auto &variance : mean_variances)
    {
        variance /= others.size();
    }

    rms = RunningMeanStd(mean_means, mean_variances);

    int total_count = std::accumulate(others.begin(), others.end(), 0,
                                      [](int accumulator, const ObservationNormalizer &other) {
                                          return accumulator + other->get_step_count();
                                      });
    rms->set_count(total_count);
}

torch::Tensor ObservationNormalizerImpl::process_observation(torch::Tensor observation)
{
    auto normalized_obs = (observation - rms->get_mean()) /
                          torch::sqrt(rms->get_variance() + 1e-8);
    return torch::clamp(normalized_obs, -clip.item(), clip.item());
}

std::vector<float> ObservationNormalizerImpl::get_mean() const
{
    auto mean = rms->get_mean();
    return std::vector<float>(mean.data_ptr<float>(), mean.data_ptr<float>() + mean.numel());
}

std::vector<float> ObservationNormalizerImpl::get_variance() const
{
    auto variance = rms->get_variance();
    return std::vector<float>(variance.data_ptr<float>(), variance.data_ptr<float>() + variance.numel());
}

void ObservationNormalizerImpl::update(torch::Tensor observations)
{
    rms->update(observations);
}

TEST_CASE("ObservationNormalizer")
{
    SUBCASE("Clips values correctly")
    {
        ObservationNormalizer normalizer(7, 1);
        float observation_array[] = {-1000, -100, -10, 0, 10, 100, 1000};
        auto observation = torch::from_blob(observation_array, {7});
        auto processed_observation = normalizer->process_observation(observation);

        auto has_too_large_values = (processed_observation > 1).any().item().toBool();
        auto has_too_small_values = (processed_observation < -1).any().item().toBool();
        DOCTEST_CHECK(!has_too_large_values);
        DOCTEST_CHECK(!has_too_small_values);
    }

    SUBCASE("Normalizes values correctly")
    {
        ObservationNormalizer normalizer(5);

        float obs_1_array[] = {-10., 0., 5., 3.2, 0.};
        float obs_2_array[] = {-5., 2., 4., 3.7, -3.};
        float obs_3_array[] = {1, 2, 3, 4, 5};
        auto obs_1 = torch::from_blob(obs_1_array, {5});
        auto obs_2 = torch::from_blob(obs_2_array, {5});
        auto obs_3 = torch::from_blob(obs_3_array, {5});

        normalizer->update(obs_1);
        normalizer->update(obs_2);
        normalizer->update(obs_3);
        auto processed_observation = normalizer->process_observation(obs_3);

        DOCTEST_CHECK(processed_observation[0].item().toFloat() == doctest::Approx(1.26008659));
        DOCTEST_CHECK(processed_observation[1].item().toFloat() == doctest::Approx(0.70712887));
        DOCTEST_CHECK(processed_observation[2].item().toFloat() == doctest::Approx(-1.2240818));
        DOCTEST_CHECK(processed_observation[3].item().toFloat() == doctest::Approx(1.10914509));
        DOCTEST_CHECK(processed_observation[4].item().toFloat() == doctest::Approx(1.31322402));
    }

    SUBCASE("Loads mean and variance from constructor correctly")
    {
        ObservationNormalizer normalizer(std::vector<float>({1, 2, 3}), std::vector<float>({4, 5, 6}));

        auto mean = normalizer->get_mean();
        auto variance = normalizer->get_variance();
        DOCTEST_CHECK(mean[0] == doctest::Approx(1));
        DOCTEST_CHECK(mean[1] == doctest::Approx(2));
        DOCTEST_CHECK(mean[2] == doctest::Approx(3));
        DOCTEST_CHECK(variance[0] == doctest::Approx(4));
        DOCTEST_CHECK(variance[1] == doctest::Approx(5));
        DOCTEST_CHECK(variance[2] == doctest::Approx(6));
    }

    SUBCASE("Is constructed from other normalizers correctly")
    {
        std::vector<ObservationNormalizer> normalizers;
        for (int i = 0; i < 3; ++i)
        {
            normalizers.push_back(ObservationNormalizer(3));
            for (int j = 0; j <= i; ++j)
            {
                normalizers[i]->update(torch::rand({3}));
            }
        }

        ObservationNormalizer combined_normalizer(normalizers);

        std::vector<std::vector<float>> means;
        std::transform(normalizers.begin(), normalizers.end(), std::back_inserter(means),
                       [](const ObservationNormalizer &normalizer) { return normalizer->get_mean(); });
        std::vector<std::vector<float>> variances;
        std::transform(normalizers.begin(), normalizers.end(), std::back_inserter(variances),
                       [](const ObservationNormalizer &normalizer) { return normalizer->get_variance(); });

        std::vector<float> mean_means;
        for (int i = 0; i < 3; ++i)
        {
            mean_means.push_back((means[0][i] + means[1][i] + means[2][i]) / 3);
        }
        std::vector<float> mean_variances;
        for (int i = 0; i < 3; ++i)
        {
            mean_variances.push_back((variances[0][i] + variances[1][i] + variances[2][i]) / 3);
        }

        auto actual_mean_means = combined_normalizer->get_mean();
        auto actual_mean_variances = combined_normalizer->get_variance();

        for (int i = 0; i < 3; ++i)
        {
            DOCTEST_CHECK(actual_mean_means[i] == doctest::Approx(mean_means[i]));
            DOCTEST_CHECK(actual_mean_variances[i] == doctest::Approx(actual_mean_variances[i]));
        }
        DOCTEST_CHECK(combined_normalizer->get_step_count() == 6);
    }
}
}