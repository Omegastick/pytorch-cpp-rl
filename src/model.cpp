#include <memory>
#include <math.h>

#include <torch/torch.h>

#include "third_party/doctest.h"
#include "cpprl/model.h"
#include "cpprl/spaces.h"

using namespace torch;

namespace cpprl
{
void init_weights(torch::OrderedDict<std::string, torch::Tensor> parameters, double gain)
{
    for (const auto &parameter : parameters)
    {
        if (parameter.key().find("bias") != std::string::npos)
        {
            nn::init::constant_(parameter.value(), gain);
        }
        else if (parameter.key().find("weight") != std::string::npos)
        {
            nn::init::orthogonal_(parameter.value());
        }
    }
}

NNBase::NNBase(bool recurrent,
               unsigned int recurrent_input_size,
               unsigned int hidden_size)
    : recurrent(recurrent), hidden_size(hidden_size), gru(nullptr)
{
    // Init GRU
    if (recurrent)
    {
        gru = register_module(
            "gru", nn::GRU(nn::GRUOptions(recurrent_input_size, hidden_size)));
        // Init weights
        init_weights(gru->named_parameters(), 0);
    }
}

unsigned int NNBase::get_hidden_size() const
{
    if (recurrent)
    {
        return hidden_size;
    }
    return 1;
}

std::vector<torch::Tensor> NNBase::forward_gru(torch::Tensor x,
                                               torch::Tensor rnn_hxs,
                                               torch::Tensor masks)
{
    if (x.size(0) == rnn_hxs.size(0))
    {
        auto gru_output = gru->forward(x.unsqueeze(0),
                                       (rnn_hxs * masks).unsqueeze(0));
        return {gru_output.output.squeeze(0), gru_output.state.squeeze(0)};
    }
    else
    {
        // x is a (timesteps, agents, -1) tensor that has been flattened to
        // (timesteps * agents, -1)
        auto agents = rnn_hxs.size(0);
        auto timesteps = x.size(0) / agents;

        // Unflatten
        x = x.view({timesteps, agents, x.size(1)});

        // Same for masks
        masks = masks.view({timesteps, agents});

        // Figure out which steps in the sequence have a zero for any agent
        // We assume the first timestep has a zero in it
        auto has_zeros = (masks.index({torch::arange(1, masks.size(0), TensorOptions(ScalarType::Long))}) == 0)
                             .any(-1)
                             .nonzero()
                             .squeeze();

        // +1 to correct the masks[1:]
        has_zeros += 1;

        // Add t=0 and t=timesteps to the list
        // has_zeros = [0] + has_zeros + [timesteps]
        has_zeros = has_zeros.contiguous().to(ScalarType::Float);
        std::vector<float> has_zeros_vec(
            has_zeros.data<float>(),
            has_zeros.data<float>() + has_zeros.numel());
        has_zeros_vec.insert(has_zeros_vec.begin(), {0});
        has_zeros_vec.push_back(timesteps);

        rnn_hxs = rnn_hxs.unsqueeze(0);
        std::vector<torch::Tensor> outputs;
        for (unsigned int i = 0; i < has_zeros_vec.size() - 1; ++i)
        {
            // We can now process steps that don't have any zeros in the masks
            // together.
            // Apparently this is much faster?
            auto start_idx = has_zeros_vec[i];
            auto end_idx = has_zeros_vec[i + 1];

            auto gru_output = gru(
                x.index({torch::arange(start_idx,
                                       end_idx,
                                       TensorOptions(ScalarType::Long))}),
                rnn_hxs * masks[start_idx].view({1, -1, 1}));

            outputs.push_back(gru_output.output);
        }

        // x is a (timesteps, agents, -1) tensor
        x = torch::cat(outputs, 1).squeeze(0);
        rnn_hxs = rnn_hxs.squeeze(0);

        return {x, rnn_hxs};
    }
}

PolicyImpl::PolicyImpl(c10::IntArrayRef /*observation_shape*/,
                       ActionSpace /*action_space*/,
                       std::shared_ptr<NNBase> base) : base(base) {}

std::vector<torch::Tensor> PolicyImpl::act(torch::Tensor /*inputs*/,
                                           torch::Tensor /*rnn_hxs*/,
                                           torch::Tensor /*masks*/)
{
    return std::vector<torch::Tensor>();
}

std::vector<torch::Tensor> PolicyImpl::evaluate_actions(torch::Tensor /*inputs*/,
                                                        torch::Tensor /*rnn_hxs*/,
                                                        torch::Tensor /*masks*/,
                                                        torch::Tensor /*actions*/)
{
    return std::vector<torch::Tensor>();
}

torch::Tensor PolicyImpl::get_values(torch::Tensor /*inputs*/,
                                     torch::Tensor /*rnn_hxs*/,
                                     torch::Tensor /*masks*/)
{
    return torch::Tensor();
}

CnnBase::CnnBase(unsigned int num_inputs,
                 bool recurrent,
                 unsigned int hidden_size)
    : NNBase(recurrent, hidden_size, hidden_size),
      main(register_module(
          "main", nn::Sequential(
                      nn::Conv2d(nn::Conv2dOptions(num_inputs, 32, 8).stride(4)),
                      nn::Functional(torch::relu),
                      nn::Conv2d(nn::Conv2dOptions(32, 64, 4).stride(2)),
                      nn::Functional(torch::relu),
                      nn::Conv2d(nn::Conv2dOptions(64, 32, 3).stride(1)),
                      nn::Functional(torch::relu),
                      Flatten(),
                      nn::Linear(32 * 7 * 7, hidden_size),
                      nn::Functional(torch::relu)))),
      critic_linear(register_module("critic_linear", nn::Linear(hidden_size, 1)))
{
    // Why this is commented out: https://github.com/pytorch/pytorch/issues/18518
    // init_weights(main->named_parameters(), sqrt(2.));
    init_weights(critic_linear->named_parameters(), 1);

    train();
}

std::vector<torch::Tensor> CnnBase::forward(torch::Tensor inputs,
                                            torch::Tensor rnn_hxs,
                                            torch::Tensor masks)
{
    auto x = main->forward(inputs / 255.);

    if (is_recurrent())
    {
        auto gru_output = forward_gru(x, rnn_hxs, masks);
        x = gru_output[0];
        rnn_hxs = gru_output[1];
    }

    return {critic_linear->forward(x), x, rnn_hxs};
}

torch::Tensor FlattenImpl::forward(torch::Tensor x)
{
    return x.view({x.size(0), -1});
}

MlpBase::MlpBase(unsigned int num_inputs,
                 bool recurrent,
                 unsigned int hidden_size)
    : NNBase(recurrent, num_inputs, hidden_size),
      actor(register_module(
          "actor", nn::Sequential(
                       nn::Linear(num_inputs, hidden_size),
                       nn::Functional(torch::tanh),
                       nn::Linear(hidden_size, hidden_size),
                       nn::Functional(torch::tanh)))),
      critic(register_module(
          "critic", nn::Sequential(
                        nn::Linear(num_inputs, hidden_size),
                        nn::Functional(torch::tanh),
                        nn::Linear(hidden_size, hidden_size),
                        nn::Functional(torch::tanh)))),
      critic_linear(register_module("critic_linear", nn::Linear(hidden_size, 1)))
{
    init_weights(actor->named_parameters(), sqrt(2.));
    init_weights(critic->named_parameters(), sqrt(2.));
    init_weights(critic_linear->named_parameters(), sqrt(2.));

    train();
}

std::vector<torch::Tensor> MlpBase::forward(torch::Tensor inputs,
                                            torch::Tensor rnn_hxs,
                                            torch::Tensor masks)
{
    auto x = inputs;

    if (is_recurrent())
    {
        auto gru_output = forward_gru(x, rnn_hxs, masks);
        x = gru_output[0];
        rnn_hxs = gru_output[1];
    }

    return {critic_linear->forward(x), x, rnn_hxs};
}

TEST_CASE("init_weights()")
{
    auto module = nn::Sequential(
        nn::Linear(5, 10),
        nn::Functional(torch::relu),
        nn::Linear(10, 8));

    init_weights(module->named_parameters(), 0);

    SUBCASE("Bias weights are initialized to 0")
    {
        for (const auto &parameter : module->named_parameters())
        {
            if (parameter.key().find("bias") != std::string::npos)
            {
                CHECK(parameter.value()[0].item().toDouble() == doctest::Approx(0));
            }
        }
    }
}

TEST_CASE("NNBase")
{
    auto base = std::make_shared<NNBase>(true, 5, 10);

    SUBCASE("forward_gru() outputs correct shapes when given samples from one"
            " agent")
    {
        auto inputs = torch::rand({4, 5});
        auto rnn_hxs = torch::rand({4, 10});
        auto masks = torch::zeros({4, 1});
        auto outputs = base->forward_gru(inputs, rnn_hxs, masks);

        REQUIRE(outputs.size() == 2);

        // x
        CHECK(outputs[0].size(0) == 4);
        CHECK(outputs[0].size(1) == 10);

        // rnn_hxs
        CHECK(outputs[1].size(0) == 4);
        CHECK(outputs[1].size(1) == 10);
    }

    SUBCASE("forward_gru() outputs correct shapes when given samples from "
            "multiple agents")
    {
        auto inputs = torch::rand({12, 5});
        auto rnn_hxs = torch::rand({4, 10});
        auto masks = torch::zeros({12, 1});
        auto outputs = base->forward_gru(inputs, rnn_hxs, masks);

        REQUIRE(outputs.size() == 2);

        // x
        CHECK(outputs[0].size(0) == 12);
        CHECK(outputs[0].size(1) == 10);

        // rnn_hxs
        CHECK(outputs[1].size(0) == 4);
        CHECK(outputs[1].size(1) == 10);
    }
}

TEST_CASE("MlpBase")
{
    auto base = MlpBase(5, true, 10);

    SUBCASE("Sanity checks")
    {
        CHECK(base.is_recurrent() == true);
        CHECK(base.get_hidden_size() == 10);
    }

    SUBCASE("Output tensors are correct shapes")
    {
        auto inputs = torch::rand({4, 5});
        auto rnn_hxs = torch::rand({4, 10});
        auto masks = torch::zeros({4, 1});
        auto outputs = base.forward(inputs, rnn_hxs, masks);

        REQUIRE(outputs.size() == 3);

        // Critic
        CHECK(outputs[0].size(0) == 4);
        CHECK(outputs[0].size(1) == 1);

        // Actor
        CHECK(outputs[1].size(0) == 4);
        CHECK(outputs[1].size(1) == 10);

        // Hidden state
        CHECK(outputs[2].size(0) == 4);
        CHECK(outputs[2].size(1) == 10);
    }
}

TEST_CASE("CnnBase")
{
    auto base = std::make_shared<CnnBase>(3, true, 10);

    SUBCASE("Sanity checks")
    {
        CHECK(base->is_recurrent() == true);
        CHECK(base->get_hidden_size() == 10);
    }

    SUBCASE("Output tensors are correct shapes")
    {
        auto inputs = torch::rand({4, 3, 84, 84});
        auto rnn_hxs = torch::rand({4, 10});
        auto masks = torch::zeros({4, 1});
        auto outputs = base->forward(inputs, rnn_hxs, masks);

        REQUIRE(outputs.size() == 3);

        // Critic
        CHECK(outputs[0].size(0) == 4);
        CHECK(outputs[0].size(1) == 1);

        // Actor
        CHECK(outputs[1].size(0) == 4);
        CHECK(outputs[1].size(1) == 10);

        // Hidden state
        CHECK(outputs[2].size(0) == 4);
        CHECK(outputs[2].size(1) == 10);
    }
}

TEST_CASE("Policy")
{
    auto base = std::make_shared<MlpBase>(3, true, 10);
    auto policy = Policy(IntArrayRef{12}, ActionSpace{"Categorical", {5}}, base);

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

        REQUIRE(outputs.size() == 3);

        // Value
        CHECK(outputs[0].size(0) == 4);
        CHECK(outputs[0].size(1) == 1);

        // Actions
        CHECK(outputs[1].size(0) == 4);
        CHECK(outputs[1].size(1) == 5);

        // Log probs
        CHECK(outputs[2].size(0) == 4);
        CHECK(outputs[2].size(1) == 5);

        // Hidden states
        CHECK(outputs[3].size(0) == 4);
        CHECK(outputs[3].size(1) == 10);
    }

    SUBCASE("evaluate_actions() output tensors are correct shapes")
    {
        auto inputs = torch::rand({4, 3});
        auto rnn_hxs = torch::rand({4, 10});
        auto masks = torch::zeros({4, 1});
        auto actions = torch::rand({4, 5});
        auto outputs = policy->evaluate_actions(inputs, rnn_hxs, masks, actions);

        REQUIRE(outputs.size() == 3);

        // Value
        CHECK(outputs[0].size(0) == 4);
        CHECK(outputs[0].size(1) == 1);

        // Log probs
        CHECK(outputs[1].size(0) == 4);
        CHECK(outputs[1].size(1) == 5);

        // Entropy
        CHECK(outputs[2].size(0) == 1);

        // Hidden states
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
}

TEST_CASE("Flatten")
{
    auto flatten = Flatten();

    SUBCASE("Flatten converts 3 dimensional vector to 2 dimensional")
    {
        auto input = torch::rand({5, 5, 5});
        auto output = flatten->forward(input);

        CHECK(output.size(0) == 5);
        CHECK(output.size(1) == 25);
    }

    SUBCASE("Flatten converts 5 dimensional vector to 2 dimensional")
    {
        auto input = torch::rand({2, 2, 2, 2, 2});
        auto output = flatten->forward(input);

        CHECK(output.size(0) == 2);
        CHECK(output.size(1) == 16);
    }

    SUBCASE("Flatten converts 1 dimensional vector to 2 dimensional")
    {
        auto input = torch::rand({10});
        auto output = flatten->forward(input);

        CHECK(output.size(0) == 10);
        CHECK(output.size(1) == 1);
    }
}
}