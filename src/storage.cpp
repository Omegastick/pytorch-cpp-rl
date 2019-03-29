#include <torch/torch.h>

#include "cpprl/storage.h"
#include "cpprl/spaces.h"
#include "third_party/doctest.h"

namespace cpprl
{
RolloutStorage::RolloutStorage(unsigned int /*num_steps*/,
                               unsigned int /*num_processes*/,
                               torch::IntArrayRef /*obs_shape*/,
                               ActionSpace /*action_space*/,
                               unsigned int /*hidden_state_size*/) {}

void RolloutStorage::after_update() {}

void RolloutStorage::compute_returns(torch::Tensor /*next_value*/,
                                     bool /*use_gae*/,
                                     double /*gamma*/,
                                     double /*tau*/) {}

void RolloutStorage::insert(torch::Tensor /*observation*/,
                            torch::Tensor /*hidden_state*/,
                            torch::Tensor /*action*/,
                            torch::Tensor /*action_log_prob*/,
                            torch::Tensor /*value_prediction*/,
                            torch::Tensor /*reward*/,
                            torch::Tensor /*mask*/) {}

void RolloutStorage::to(torch::Device) {}

TEST_CASE("RolloutStorage")
{
    SUBCASE("RolloutStorage initializes tensors to correct sizes")
    {
        RolloutStorage storage(3, 4, {5, 2}, ActionSpace{"Discrete", {3}}, 10);

        CHECK(storage.get_observations().size(0) == 4);
        CHECK(storage.get_observations().size(1) == 3);
        CHECK(storage.get_observations().size(2) == 5);
        CHECK(storage.get_observations().size(3) == 2);

        CHECK(storage.get_hidden_states().size(0) == 4);
        CHECK(storage.get_hidden_states().size(1) == 3);
        CHECK(storage.get_hidden_states().size(2) == 10);

        CHECK(storage.get_rewards().size(0) == 3);
        CHECK(storage.get_rewards().size(1) == 3);
        CHECK(storage.get_rewards().size(2) == 1);

        CHECK(storage.get_value_predictions().size(0) == 4);
        CHECK(storage.get_value_predictions().size(1) == 3);
        CHECK(storage.get_value_predictions().size(2) == 1);

        CHECK(storage.get_returns().size(0) == 4);
        CHECK(storage.get_returns().size(1) == 3);
        CHECK(storage.get_returns().size(2) == 1);

        CHECK(storage.get_action_log_probs().size(0) == 3);
        CHECK(storage.get_action_log_probs().size(1) == 3);
        CHECK(storage.get_action_log_probs().size(2) == 1);

        CHECK(storage.get_actions().size(0) == 3);
        CHECK(storage.get_actions().size(1) == 3);
        CHECK(storage.get_actions().size(2) == 1);

        CHECK(storage.get_masks().size(0) == 4);
        CHECK(storage.get_masks().size(1) == 3);
        CHECK(storage.get_masks().size(2) == 1);
    }

    SUBCASE("to() doesn't crash")
    {
        RolloutStorage storage(3, 4, {5}, ActionSpace{"Discrete", {3}}, 10);
        storage.to(torch::kCPU);
    }

    SUBCASE("insert() inserts values")
    {
        RolloutStorage storage(3, 4, {5, 2}, ActionSpace{"Discrete", {3}}, 10);
        storage.insert(torch::rand({3, 5, 2}) + 1,
                       torch::rand({3, 10}) + 1,
                       torch::randint(1, 3, {3, 1}),
                       torch::rand({3, 1}) + 1,
                       torch::rand({3, 1}) + 1,
                       torch::rand({3, 1}) + 1,
                       torch::ones({3, 1}));

        CHECK(storage.get_observations()[0][0][0][0].item().toDouble() !=
              doctest::Approx(0));
        CHECK(storage.get_hidden_states()[0][0][0].item().toDouble() !=
              doctest::Approx(0));
        CHECK(storage.get_actions()[0][0][0].item().toDouble() !=
              doctest::Approx(0));
        CHECK(storage.get_action_log_probs()[0][0][0].item().toDouble() !=
              doctest::Approx(0));
        CHECK(storage.get_value_predictions()[0][0][0].item().toDouble() !=
              doctest::Approx(0));
        CHECK(storage.get_rewards()[0][0][0].item().toDouble() !=
              doctest::Approx(0));
        CHECK(storage.get_masks()[0][0][0].item().toDouble() !=
              doctest::Approx(0));
    }

    SUBCASE("compute_returns()")
    {
        RolloutStorage storage(3, 2, {4}, ActionSpace{"Discrete", {3}}, 5);

        std::vector<float> value_preds{0, 1};
        std::vector<float> rewards{0, 1};
        std::vector<int> masks{1, 1};
        storage.insert(torch::zeros({2, 4}),
                       torch::zeros({2, 5}),
                       torch::zeros({2, 1}),
                       torch::zeros({2, 1}),
                       torch::from_blob(&value_preds[0], {2, 1}),
                       torch::from_blob(&rewards[0], {2, 1}),
                       torch::from_blob(&masks[0], {2, 1}));
        value_preds = {1, 2};
        rewards = {1, 2};
        masks = {1, 0};
        storage.insert(torch::zeros({2, 4}),
                       torch::zeros({2, 5}),
                       torch::zeros({2, 1}),
                       torch::zeros({2, 1}),
                       torch::from_blob(&value_preds[0], {2, 1}),
                       torch::from_blob(&rewards[0], {2, 1}),
                       torch::from_blob(&masks[0], {2, 1}));
        value_preds = {2, 3};
        rewards = {2, 3};
        masks = {1, 1};
        storage.insert(torch::zeros({2, 4}),
                       torch::zeros({2, 5}),
                       torch::zeros({2, 1}),
                       torch::zeros({2, 1}),
                       torch::from_blob(&value_preds[0], {2, 1}),
                       torch::from_blob(&rewards[0], {2, 1}),
                       torch::from_blob(&masks[0], {2, 1}));

        SUBCASE("Gives correct results without GAE")
        {
            std::vector<float> next_values{0, 1};
            storage.compute_returns(torch::from_blob(&next_values[0], {2, 1}),
                                    false, 0.6, 0.6);

            CHECK(storage.get_returns()[0][0].item().toDouble() ==
                  doctest::Approx(1.32));
            CHECK(storage.get_returns()[0][1].item().toDouble() ==
                  doctest::Approx(2.2));
            CHECK(storage.get_returns()[1][0].item().toDouble() ==
                  doctest::Approx(2.2));
            CHECK(storage.get_returns()[1][1].item().toDouble() ==
                  doctest::Approx(2));
            CHECK(storage.get_returns()[2][0].item().toDouble() ==
                  doctest::Approx(2));
            CHECK(storage.get_returns()[2][1].item().toDouble() ==
                  doctest::Approx(3.6));
            CHECK(storage.get_returns()[3][0].item().toDouble() ==
                  doctest::Approx(0));
            CHECK(storage.get_returns()[3][1].item().toDouble() ==
                  doctest::Approx(1));
        }

        SUBCASE("Gives correct results with GAE")
        {
            std::vector<float> next_values{0, 1};
            storage.compute_returns(torch::from_blob(&next_values[0], {2, 1}),
                                    true, 0.6, 0.6);

            CHECK(storage.get_returns()[0][0].item().toDouble() ==
                  doctest::Approx(1.032));
            CHECK(storage.get_returns()[0][1].item().toDouble() ==
                  doctest::Approx(2.2));
            CHECK(storage.get_returns()[1][0].item().toDouble() ==
                  doctest::Approx(2.2));
            CHECK(storage.get_returns()[1][1].item().toDouble() ==
                  doctest::Approx(2));
            CHECK(storage.get_returns()[2][0].item().toDouble() ==
                  doctest::Approx(2));
            CHECK(storage.get_returns()[2][1].item().toDouble() ==
                  doctest::Approx(3.6));
            CHECK(storage.get_returns()[3][0].item().toDouble() ==
                  doctest::Approx(0));
            CHECK(storage.get_returns()[3][1].item().toDouble() ==
                  doctest::Approx(1));
        }
    }
}
}