#include <chrono>
#include <memory>

#include <torch/torch.h>

#include "cpprl/algorithms/ppo.h"
#include "cpprl/algorithms/algorithm.h"
#include "cpprl/model/mlp_base.h"
#include "cpprl/model/policy.h"
#include "cpprl/storage.h"
#include "cpprl/spaces.h"
#include "third_party/doctest.h"

namespace cpprl
{

PPO::PPO(Policy &policy,
         float clip_param,
         int epoch_count,
         int num_mini_batch,
         float value_loss_coef,
         float entropy_coef,
         float learning_rate,
         float epsilon,
         float max_grad_norm)
    : policy(policy),
      clip_param(clip_param),
      value_loss_coef(value_loss_coef),
      entropy_coef(entropy_coef),
      max_grad_norm(max_grad_norm),
      epoch_count(epoch_count),
      num_mini_batch(num_mini_batch),
      optimizer(std::make_unique<torch::optim::Adam>(
          policy->parameters(),
          torch::optim::AdamOptions(learning_rate)
              .eps(epsilon))) {}

std::vector<UpdateDatum> PPO::update(RolloutStorage & /*rollouts*/)
{
    return {};
}

TEST_CASE("PPO")
{
    torch::manual_seed(0);
    SUBCASE("update() learns basic pattern")
    {
        auto base = std::make_shared<MlpBase>(1, false, 5);
        ActionSpace space{"Discrete", {2}};
        Policy policy(space, base);
        RolloutStorage storage(20, 2, {1}, space, 5, torch::kCPU);
        PPO ppo(policy, 0.2, 3, 5, 0.5, 1e-3, 0.001);

        // The reward is the action
        auto pre_game_probs = policy->get_probs(
            torch::ones({2, 1}),
            torch::zeros({2, 5}),
            torch::ones({2, 1}));

        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 20; ++j)
            {
                auto observation = torch::randint(0, 2, {2, 1});

                std::vector<torch::Tensor> act_result;
                {
                    torch::NoGradGuard no_grad;
                    act_result = policy->act(observation,
                                             torch::Tensor(),
                                             torch::ones({2, 1}));
                }
                auto actions = act_result[1];

                auto rewards = actions;
                storage.insert(observation,
                               torch::zeros({2, 5}),
                               actions,
                               act_result[2],
                               act_result[0],
                               rewards,
                               torch::ones({2, 1}));
            }

            torch::Tensor next_value;
            {
                torch::NoGradGuard no_grad;
                next_value = policy->get_values(
                                       storage.get_observations()[-1],
                                       storage.get_hidden_states()[-1],
                                       storage.get_masks()[-1])
                                 .detach();
            }
            storage.compute_returns(next_value, false, 0., 0.9);

            ppo.update(storage);
            storage.after_update();
        }

        auto post_game_probs = policy->get_probs(
            torch::ones({2, 1}),
            torch::zeros({2, 5}),
            torch::ones({2, 1}));

        INFO("Pre-training probabilities: \n"
             << pre_game_probs << "\n");
        INFO("Post-training probabilities: \n"
             << post_game_probs << "\n");
        CHECK(post_game_probs[0][0].item().toDouble() <
              pre_game_probs[0][0].item().toDouble());
        CHECK(post_game_probs[0][1].item().toDouble() >
              pre_game_probs[0][1].item().toDouble());
    }

    SUBCASE("update() learns basic game")
    {
        auto base = std::make_shared<MlpBase>(1, false, 5);
        ActionSpace space{"Discrete", {2}};
        Policy policy(space, base);
        RolloutStorage storage(20, 2, {1}, space, 5, torch::kCPU);
        PPO ppo(policy, 0.2, 3, 5, 0.5, 1e-3, 0.001);

        // The game is: If the action matches the input, give a reward of 1, otherwise -1
        auto pre_game_probs = policy->get_probs(
            torch::ones({2, 1}),
            torch::zeros({2, 5}),
            torch::ones({2, 1}));

        auto observation = torch::randint(0, 2, {2, 1});
        storage.set_first_observation(observation);

        for (int i = 0; i < 10; ++i)
        {
            for (int j = 0; j < 20; ++j)
            {
                std::vector<torch::Tensor> act_result;
                {
                    torch::NoGradGuard no_grad;
                    act_result = policy->act(observation,
                                             torch::Tensor(),
                                             torch::ones({2, 1}));
                }
                auto actions = act_result[1];

                auto rewards = ((actions == observation.to(torch::kLong)).to(torch::kFloat) * 2) - 1;
                observation = torch::randint(0, 2, {2, 1});
                storage.insert(observation,
                               torch::zeros({2, 5}),
                               actions,
                               act_result[2],
                               act_result[0],
                               rewards,
                               torch::ones({2, 1}));
            }

            torch::Tensor next_value;
            {
                torch::NoGradGuard no_grad;
                next_value = policy->get_values(
                                       storage.get_observations()[-1],
                                       storage.get_hidden_states()[-1],
                                       storage.get_masks()[-1])
                                 .detach();
            }
            storage.compute_returns(next_value, false, 0.1, 0.9);

            ppo.update(storage);
            storage.after_update();
        }

        auto post_game_probs = policy->get_probs(
            torch::ones({2, 1}),
            torch::zeros({2, 5}),
            torch::ones({2, 1}));

        INFO("Pre-training probabilities: \n"
             << pre_game_probs << "\n");
        INFO("Post-training probabilities: \n"
             << post_game_probs << "\n");
        CHECK(post_game_probs[0][0].item().toDouble() <
              pre_game_probs[0][0].item().toDouble());
        CHECK(post_game_probs[0][1].item().toDouble() >
              pre_game_probs[0][1].item().toDouble());
    }
}
}