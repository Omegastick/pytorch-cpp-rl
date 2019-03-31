#include <memory>

#include <torch/torch.h>

#include "cpprl/algorithms/a2c.h"
#include "cpprl/algorithms/algorithm.h"
#include "cpprl/model/mlp_base.h"
#include "cpprl/model/policy.h"
#include "cpprl/storage.h"
#include "cpprl/spaces.h"
#include "third_party/doctest.h"

namespace cpprl
{
A2C::A2C(Policy &policy,
         float value_loss_coef,
         float entropy_coef,
         float learning_rate,
         float epsilon,
         float alpha,
         float max_grad_norm)
    : policy(&policy),
      value_loss_coef(value_loss_coef),
      entropy_coef(entropy_coef),
      max_grad_norm(max_grad_norm),
      optimizer(std::make_unique<torch::optim::RMSprop>(
          policy->parameters(),
          torch::optim::RMSpropOptions(learning_rate)
              .eps(epsilon)
              .alpha(alpha))) {}

std::vector<UpdateDatum> A2C::update(RolloutStorage & /*rollouts*/)
{
    return std::vector<UpdateDatum>();
}

TEST_CASE("A2C")
{
    auto base = std::make_shared<MlpBase>(2, false, 10);
    ActionSpace space{"Discrete", {2}};
    Policy policy(space, base);
    RolloutStorage storage(3, 2, {2}, space, 10);
    A2C a2c(policy, 1.0, 1e-7, 0.01, 0.000001, 0.99, 0.5);

    SUBCASE("update() learns basic game")
    {
        // The game is: If the input is {1, 0} action 0 gets a reward, and for
        // {0, 1} action 1 gets a reward.
        std::vector<float> observation_vec{1, 0};
        auto pre_game_probs = policy->get_probs(
            torch::from_blob(observation_vec.data(), {2}).expand({2, 2}),
            torch::zeros({2, 10}),
            torch::ones({2, 1}));

        for (int i = 0; i < 10; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                int target_action;
                if (torch::randint(2, {1}).item().toBool())
                {
                    observation_vec = {1, 0};
                    target_action = 0;
                }
                else
                {
                    observation_vec = {0, 1};
                    target_action = 1;
                }
                auto observation = torch::from_blob(observation_vec.data(), {2});

                auto act_result = policy->act(observation.expand({2, 2}),
                                              torch::Tensor(),
                                              torch::ones({2, 1}));
                auto actions = act_result[1];

                float rewards_array[2];
                for (int process = 0; process < actions.size(0); ++process)
                {
                    if (actions[process].item().toInt() == target_action)
                    {
                        rewards_array[process] = 1;
                    }
                    else
                    {
                        rewards_array[process] = 0;
                    }
                }
                auto rewards = torch::from_blob(rewards_array, {2, 1});
                storage.insert(observation,
                               torch::zeros({2, 10}),
                               actions,
                               act_result[2],
                               act_result[0],
                               rewards,
                               torch::ones({2, 1}));
            }

            a2c.update(storage);
            storage.after_update();
        }
        observation_vec = {1, 0};
        auto post_game_probs = policy->get_probs(
            torch::from_blob(observation_vec.data(), {2}).expand({2, 2}),
            torch::zeros({2, 10}),
            torch::ones({2, 1}));

        CHECK(post_game_probs[0][0].item().toDouble() >
              pre_game_probs[0][0].item().toDouble());
        CHECK(post_game_probs[0][1].item().toDouble() <
              pre_game_probs[0][1].item().toDouble());
    }
}
}