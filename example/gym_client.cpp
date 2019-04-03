#include <string.h>
#include <fstream>

#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

#include <cpprl/spaces.h>
#include <cpprl/storage.h>
#include <cpprl/algorithms/a2c.h>
#include <cpprl/model/mlp_base.h>
#include <cpprl/model/policy.h>

#include "communicator.h"
#include "requests.h"

using namespace gym_client;
using namespace cpprl;

const int reward_average_window_size = 1000;
const int num_envs = 1;
const int batch_size = 5;
const float value_loss_coef = 0.5;
const float discount_factor = 0.99;
const int hidden_size = 5;

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

    torch::set_num_threads(1);
    torch::manual_seed(0);

    spdlog::info("Connecting to gym server");
    Communicator communicator("tcp://127.0.0.1:10201");

    spdlog::info("Creating environment");
    auto make_param = std::make_shared<MakeParam>();
    make_param->env_name = "CartPole-v1";
    make_param->gamma = discount_factor;
    make_param->num_envs = num_envs;
    Request<MakeParam> make_request("make", make_param);
    communicator.send_request(make_request);
    spdlog::info(communicator.get_response<MakeResponse>()->result);

    spdlog::info("Resetting environment");
    auto reset_param = std::make_shared<ResetParam>();
    Request<ResetParam> reset_request("reset", reset_param);
    communicator.send_request(reset_request);
    auto observation_vec = flatten_2d_vector<float>(communicator.get_response<ResetResponse>()->observation);
    auto observation = torch::from_blob(observation_vec.data(), {num_envs, 4});

    auto base = std::make_shared<MlpBase>(4, false, hidden_size);
    ActionSpace space{"Discrete", {2}};
    Policy policy(space, base);
    RolloutStorage storage(batch_size, num_envs, {4}, space, hidden_size);
    A2C a2c(policy, value_loss_coef, 1e-5, 0.001);

    storage.set_first_observation(observation);

    std::ifstream weights_file{"/home/px046/prog/pytorch-cpp-rl/build/weights.json"};
    auto json = nlohmann::json::parse(weights_file);
    for (const auto &parameter : json.items())
    {
        if (base->named_parameters().contains(parameter.key()))
        {
            std::vector<int64_t> tensor_size = parameter.value()[0];
            std::vector<float> parameter_vec;
            if (parameter.key().find("bias") == std::string::npos)
            {
                std::vector<std::vector<float>> parameter_2d_vec = parameter.value()[1].get<std::vector<std::vector<float>>>();
                parameter_vec = flatten_2d_vector<float>(parameter_2d_vec);
            }
            else
            {
                parameter_vec = parameter.value()[1].get<std::vector<float>>();
            }
            auto json_weights = torch::from_blob(parameter_vec.data(), tensor_size);
            int element_count = json_weights.numel();
            memcpy(base->named_parameters()[parameter.key()].data<float>(), json_weights.data<float>(), element_count);
            spdlog::info("Wrote {}", parameter.key());
            if (parameter.key().find("bias") == std::string::npos)
            {
                spdlog::info("Json: {} - Memory: {}", parameter.value()[1][0][1], base->named_parameters()[parameter.key()][0][1].item().toFloat());
            }
        }
        else if (policy->named_modules()["output"]->named_parameters().contains(parameter.key()))
        {
            std::vector<int64_t> tensor_size = parameter.value()[0];
            std::vector<float> parameter_vec;
            if (parameter.key().find("bias") == std::string::npos)
            {
                std::vector<std::vector<float>> parameter_2d_vec = parameter.value()[1].get<std::vector<std::vector<float>>>();
                parameter_vec = flatten_2d_vector<float>(parameter_2d_vec);
            }
            else
            {
                parameter_vec = parameter.value()[1].get<std::vector<float>>();
            }
            auto json_weights = torch::from_blob(parameter_vec.data(), tensor_size);
            int element_count = json_weights.numel();
            memcpy(policy->named_modules()["output"]->named_parameters()[parameter.key()].data<float>(), json_weights.data<float>(), element_count);
            spdlog::info("Wrote {}", parameter.key());
            if (parameter.key().find("bias") == std::string::npos)
            {
                spdlog::info("Json: {} - Memory: {}",
                             parameter.value()[1][0][1],
                             policy->named_modules()["output"]->named_parameters()[parameter.key()][0][1].item().toFloat());
            }
        }
    }

    std::vector<float> running_rewards(num_envs);
    int episode_count = 0;
    std::vector<float> reward_history(reward_average_window_size);

    torch::manual_seed(0);
    for (int i = 0; i < 1; ++i)
    {
        for (int step = 0; step < batch_size; ++step)
        {
            std::vector<torch::Tensor> act_result;
            {
                torch::NoGradGuard no_grad;
                act_result = policy->act(observation,
                                         torch::Tensor(),
                                         torch::ones({2, 1}));
            }
            long *actions_array = act_result[1].data<long>();
            std::vector<std::vector<int>> actions(num_envs);
            for (int i = 0; i < num_envs; ++i)
            {
                actions[i] = {static_cast<int>(actions_array[i])};
            }

            auto step_param = std::make_shared<StepParam>();
            step_param->actions = actions;
            step_param->render = false;
            Request<StepParam> step_request("step", step_param);
            communicator.send_request(step_request);
            auto step_result = communicator.get_response<StepResponse>();
            observation_vec = flatten_2d_vector<float>(step_result->observation);
            observation = torch::from_blob(observation_vec.data(), {num_envs, 4});
            auto rewards = flatten_2d_vector<float>(step_result->reward);
            for (int i = 0; i < num_envs; ++i)
            {
                running_rewards[i] += rewards[i];
                if (step_result->done[i][0])
                {
                    reward_history[episode_count % reward_average_window_size] = running_rewards[i];
                    running_rewards[i] = 0;
                    episode_count++;
                }
            }
            auto dones = torch::zeros({num_envs, 1});
            for (int i = 0; i < num_envs; ++i)
            {
                dones[i][0] = static_cast<int>(step_result->done[i][0]);
            }

            storage.insert(observation,
                           torch::zeros({num_envs, hidden_size}),
                           act_result[1],
                           act_result[2],
                           act_result[0],
                           torch::from_blob(rewards.data(), {num_envs, 1}),
                           1 - dones);
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
        storage.compute_returns(next_value, false, discount_factor, 0.9);

        auto update_data = a2c.update(storage);
        storage.after_update();

        for (const auto &datum : update_data)
        {
            spdlog::info("{}: {}", datum.name, datum.value);
        }
        float average_reward = std::accumulate(reward_history.begin(), reward_history.end(), 0);
        average_reward /= episode_count < reward_average_window_size ? episode_count : reward_average_window_size;
        spdlog::info("Reward: {}", average_reward);
    }
}