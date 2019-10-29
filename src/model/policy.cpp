#include <torch/torch.h>

#include "cpprl/model/policy.h"
#include "cpprl/distributions/categorical.h"
#include "cpprl/model/mlp_base.h"
#include "cpprl/model/output_layers.h"
#include "cpprl/observation_normalizer.h"
#include "cpprl/spaces.h"
#include "third_party/doctest.h"

using namespace torch;

namespace cpprl
{
PolicyImpl::PolicyImpl(ActionSpace action_space,
                       std::shared_ptr<NNBase> base,
                       bool normalize_observations)
    : action_space(action_space),
      base(register_module("base", base)),
      observation_normalizer(nullptr)
{
    int num_outputs = action_space.shape[0];
    if (action_space.type == "Discrete")
    {
        output_layer = std::make_shared<CategoricalOutput>(
            base->get_output_size(), num_outputs);
    }
    else if (action_space.type == "Box")
    {
        output_layer = std::make_shared<NormalOutput>(
            base->get_output_size(), num_outputs);
    }
    else if (action_space.type == "MultiBinary")
    {
        output_layer = std::make_shared<BernoulliOutput>(
            base->get_output_size(), num_outputs);
    }
    else
    {
        throw std::runtime_error("Action space " + action_space.type + " not supported");
    }
    register_module("output", output_layer);

    if (normalize_observations)
    {
        // Normalized observations only supported for MlpBase
        auto mlp_base = static_cast<MlpBase *>(base.get());
        observation_normalizer = register_module("observation_normalizer", ObservationNormalizer(mlp_base->get_num_inputs()));
    }
}

std::vector<torch::Tensor> PolicyImpl::act(torch::Tensor inputs,
                                           torch::Tensor rnn_hxs,
                                           torch::Tensor masks) const
{
    if (observation_normalizer)
    {
        inputs = observation_normalizer->process_observation(inputs);
    }

    auto base_output = base->forward(inputs, rnn_hxs, masks);
    auto dist = output_layer->forward(base_output[1]);

    auto action = dist->sample();
    auto action_log_probs = dist->log_prob(action);

    if (action_space.type == "Discrete")
    {
        action = action.unsqueeze(-1);
        action_log_probs = action_log_probs.unsqueeze(-1);
    }
    else
    {
        action_log_probs = dist->log_prob(action).sum(-1, true);
    }

    return {base_output[0], // value
            action,
            action_log_probs,
            base_output[2]}; // rnn_hxs
}

std::vector<torch::Tensor> PolicyImpl::evaluate_actions(torch::Tensor inputs,
                                                        torch::Tensor rnn_hxs,
                                                        torch::Tensor masks,
                                                        torch::Tensor actions) const
{
    if (observation_normalizer)
    {
        inputs = observation_normalizer->process_observation(inputs);
    }

    auto base_output = base->forward(inputs, rnn_hxs, masks);
    auto dist = output_layer->forward(base_output[1]);

    torch::Tensor action_log_probs;
    if (action_space.type == "Discrete")
    {
        action_log_probs = dist->log_prob(actions.squeeze(-1))
                               .view({actions.size(0), -1})
                               .sum(-1)
                               .unsqueeze(-1);
    }
    else
    {
        action_log_probs = dist->log_prob(actions).sum(-1, true);
    }

    auto entropy = dist->entropy().mean();

    return {base_output[0], // value
            action_log_probs,
            entropy,
            base_output[2]}; // rnn_hxs
}

torch::Tensor PolicyImpl::get_probs(torch::Tensor inputs,
                                    torch::Tensor rnn_hxs,
                                    torch::Tensor masks) const
{
    if (observation_normalizer)
    {
        inputs = observation_normalizer->process_observation(inputs);
    }

    auto base_output = base->forward(inputs, rnn_hxs, masks);
    auto dist = output_layer->forward(base_output[1]);

    return static_cast<Categorical *>(dist.get())->get_probs();
}

torch::Tensor PolicyImpl::get_values(torch::Tensor inputs,
                                     torch::Tensor rnn_hxs,
                                     torch::Tensor masks) const
{
    if (observation_normalizer)
    {
        inputs = observation_normalizer->process_observation(inputs);
    }

    auto base_output = base->forward(inputs, rnn_hxs, masks);

    return base_output[0];
}

void PolicyImpl::update_observation_normalizer(torch::Tensor observations)
{
    assert(!observation_normalizer.is_empty());
    observation_normalizer->update(observations);
}

TEST_CASE("Policy")
{
    SUBCASE("Recurrent")
    {
        auto base = std::make_shared<MlpBase>(3, true, 10);
        Policy policy(ActionSpace{"Discrete", {5}}, base);

        SUBCASE("Sanity checks")
        {
            CHECK(policy->is_recurrent() == true);
            CHECK(policy->get_hidden_size() == 10);
        }

        SUBCASE("act() output tensors are correct shapes")
        {
            auto inputs = torch::rand({4, 3});
            auto rnn_hxs = torch::rand({4, 10});
            auto masks = torch::zeros({4, 1});
            auto outputs = policy->act(inputs, rnn_hxs, masks);

            REQUIRE(outputs.size() == 4);

            // Value
            INFO("Value: \n"
                 << outputs[0] << "\n");
            CHECK(outputs[0].size(0) == 4);
            CHECK(outputs[0].size(1) == 1);

            // Actions
            INFO("Actions: \n"
                 << outputs[1] << "\n");
            CHECK(outputs[1].size(0) == 4);
            CHECK(outputs[1].size(1) == 1);

            // Log probs
            INFO("Log probs: \n"
                 << outputs[2] << "\n");
            CHECK(outputs[2].size(0) == 4);
            CHECK(outputs[2].size(1) == 1);

            // Hidden states
            INFO("Hidden states: \n"
                 << outputs[3] << "\n");
            CHECK(outputs[3].size(0) == 4);
            CHECK(outputs[3].size(1) == 10);
        }

        SUBCASE("evaluate_actions() output tensors are correct shapes")
        {
            auto inputs = torch::rand({4, 3});
            auto rnn_hxs = torch::rand({4, 10});
            auto masks = torch::zeros({4, 1});
            auto actions = torch::randint(5, {4, 1});
            auto outputs = policy->evaluate_actions(inputs, rnn_hxs, masks, actions);

            REQUIRE(outputs.size() == 4);

            // Value
            INFO("Value: \n"
                 << outputs[0] << "\n");
            CHECK(outputs[0].size(0) == 4);
            CHECK(outputs[0].size(1) == 1);

            // Log probs
            INFO("Log probs: \n"
                 << outputs[1] << "\n");
            CHECK(outputs[1].size(0) == 4);
            CHECK(outputs[1].size(1) == 1);

            // Entropy
            INFO("Entropy: \n"
                 << outputs[2] << "\n");
            CHECK(outputs[2].sizes().size() == 0);

            // Hidden states
            INFO("Hidden states: \n"
                 << outputs[3] << "\n");
            CHECK(outputs[3].size(0) == 4);
            CHECK(outputs[3].size(1) == 10);
        }

        SUBCASE("get_values() output tensor is correct shapes")
        {
            auto inputs = torch::rand({4, 3});
            auto rnn_hxs = torch::rand({4, 10});
            auto masks = torch::zeros({4, 1});
            auto outputs = policy->get_values(inputs, rnn_hxs, masks);

            // Value
            CHECK(outputs.size(0) == 4);
            CHECK(outputs.size(1) == 1);
        }

        SUBCASE("get_probs() output tensor is correct shapes")
        {
            auto inputs = torch::rand({4, 3});
            auto rnn_hxs = torch::rand({4, 10});
            auto masks = torch::zeros({4, 1});
            auto outputs = policy->get_probs(inputs, rnn_hxs, masks);

            // Probabilities
            CHECK(outputs.size(0) == 4);
            CHECK(outputs.size(1) == 5);
        }
    }

    SUBCASE("Non-recurrent")
    {
        auto base = std::make_shared<MlpBase>(3, false, 10);
        Policy policy(ActionSpace{"Discrete", {5}}, base);

        SUBCASE("Sanity checks")
        {
            CHECK(policy->is_recurrent() == false);
        }

        SUBCASE("act() output tensors are correct shapes")
        {
            auto inputs = torch::rand({4, 3});
            auto rnn_hxs = torch::rand({4, 10});
            auto masks = torch::zeros({4, 1});
            auto outputs = policy->act(inputs, rnn_hxs, masks);

            REQUIRE(outputs.size() == 4);

            // Value
            INFO("Value: \n"
                 << outputs[0] << "\n");
            CHECK(outputs[0].size(0) == 4);
            CHECK(outputs[0].size(1) == 1);

            // Actions
            INFO("Actions: \n"
                 << outputs[1] << "\n");
            CHECK(outputs[1].size(0) == 4);
            CHECK(outputs[1].size(1) == 1);

            // Log probs
            INFO("Log probs: \n"
                 << outputs[2] << "\n");
            CHECK(outputs[2].size(0) == 4);
            CHECK(outputs[2].size(1) == 1);

            // Hidden states
            INFO("Hidden states: \n"
                 << outputs[3] << "\n");
            CHECK(outputs[3].size(0) == 4);
            CHECK(outputs[3].size(1) == 10);
        }

        SUBCASE("evaluate_actions() output tensors are correct shapes")
        {
            auto inputs = torch::rand({4, 3});
            auto rnn_hxs = torch::rand({4, 10});
            auto masks = torch::zeros({4, 1});
            auto actions = torch::randint(5, {4, 1});
            auto outputs = policy->evaluate_actions(inputs, rnn_hxs, masks, actions);

            REQUIRE(outputs.size() == 4);

            // Value
            INFO("Value: \n"
                 << outputs[0] << "\n");
            CHECK(outputs[0].size(0) == 4);
            CHECK(outputs[0].size(1) == 1);

            // Log probs
            INFO("Log probs: \n"
                 << outputs[1] << "\n");
            CHECK(outputs[1].size(0) == 4);
            CHECK(outputs[1].size(1) == 1);

            // Entropy
            INFO("Entropy: \n"
                 << outputs[2] << "\n");
            CHECK(outputs[2].sizes().size() == 0);

            // Hidden states
            INFO("Hidden states: \n"
                 << outputs[3] << "\n");
            CHECK(outputs[3].size(0) == 4);
            CHECK(outputs[3].size(1) == 10);
        }

        SUBCASE("get_values() output tensor is correct shapes")
        {
            auto inputs = torch::rand({4, 3});
            auto rnn_hxs = torch::rand({4, 10});
            auto masks = torch::zeros({4, 1});
            auto outputs = policy->get_values(inputs, rnn_hxs, masks);

            // Value
            CHECK(outputs.size(0) == 4);
            CHECK(outputs.size(1) == 1);
        }

        SUBCASE("get_probs() output tensor is correct shapes")
        {
            auto inputs = torch::rand({4, 3});
            auto rnn_hxs = torch::rand({4, 10});
            auto masks = torch::zeros({4, 1});
            auto outputs = policy->get_probs(inputs, rnn_hxs, masks);

            // Probabilities
            CHECK(outputs.size(0) == 4);
            CHECK(outputs.size(1) == 5);
        }
    }
}
}