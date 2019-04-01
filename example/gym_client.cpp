#include <spdlog/spdlog.h>

#include <cpprl/spaces.h>
#include <cpprl/storage.h>
#include <cpprl/algorithms/a2c.h>
#include <cpprl/model/mlp_base.h>
#include <cpprl/model/policy.h>

#include "communicator.h"
#include "requests.h"

using namespace gym_client;
using namespace cpprl;

template <typename T>
std::vector<T> flatten_2d_vector(std::vector<std::vector<T>> const &input)
{
    std::vector<T> output;

    for (auto const &sub_vector : input)
    {
        output.reserve(output.size() + sub_vector.size());
        output.insert(output.end(), sub_vector.cbegin(), sub_vector.cend());
    }

    return output;
}

int main(int argc, char *argv[])
{
    spdlog::set_level(spdlog::level::debug);
    spdlog::set_pattern("%^[%T %7l] %v%$");

    spdlog::info("Connecting to gym server");
    Communicator communicator("tcp://127.0.0.1:10201");

    spdlog::info("Creating environment");
    auto make_param = std::make_shared<MakeParam>();
    make_param->env_name = "CartPole-v1";
    make_param->env_type = "classic_control";
    make_param->num_envs = 4;
    Request<MakeParam> make_request("make", make_param);
    communicator.send_request(make_request);
    spdlog::info(communicator.get_response<MakeResponse>()->result);

    spdlog::info("Resetting environment");
    auto reset_param = std::make_shared<ResetParam>();
    Request<ResetParam> reset_request("reset", reset_param);
    communicator.send_request(reset_request);
    auto observation_vec = flatten_2d_vector<float>(communicator.get_response<ResetResponse>()->observation);
    auto observation = torch::from_blob(observation_vec.data(), {4, 4});

    auto base = std::make_shared<MlpBase>(4, false, 5);
    ActionSpace space{"Discrete", {2}};
    Policy policy(space, base);
    RolloutStorage storage(100, 4, {4}, space, 5);
    A2C a2c(policy, 0.5, 1e-3, 0.0001);

    for (int i = 0; i < 1000; ++i)
    {

        for (int step = 0; step < 100; ++step)
        {
            std::vector<torch::Tensor> act_result;
            {
                torch::NoGradGuard no_grad;
                act_result = policy->act(observation,
                                         torch::Tensor(),
                                         torch::ones({2, 1}));
            }
            long *actions_array = act_result[1].data<long>();
            std::vector<std::vector<int>> actions;
            actions.resize(4);
            for (int i = 0; i < 4; ++i)
            {
                actions[i] = {static_cast<int>(actions_array[i])};
            }

            auto step_param = std::make_shared<StepParam>();
            step_param->actions = actions;
            step_param->render = false;
            Request<StepParam> step_request("step", step_param);
            communicator.send_request(step_request);
            auto step_result = communicator.get_response<StepResponse>();
            auto rewards = flatten_2d_vector<float>(step_result->reward);
            auto dones = torch::zeros({4, 1});
            for (int i = 0; i < 4; ++i)
            {
                dones[i][0] = 1 - static_cast<int>(step_result->done[i][0]);
            }

            storage.insert(observation,
                           torch::zeros({4, 5}),
                           act_result[1],
                           act_result[2],
                           act_result[0],
                           torch::from_blob(rewards.data(), {4, 1}),
                           dones);
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

        a2c.update(storage);
        storage.after_update();
    }
}