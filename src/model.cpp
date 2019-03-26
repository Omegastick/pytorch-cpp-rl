#include <memory>

#include <torch/torch.h>

#include "third_party/doctest.h"
#include "cpprl/model.h"
#include "cpprl/spaces.h"

namespace cpprl
{
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
        for (const auto &parameter : gru->named_parameters())
        {
            if (parameter.key().find("bias") != std::string::npos)
            {
                nn::init::constant_(parameter.value(), 0);
            }
            else if (parameter.key().find("weight") != std::string::npos)
            {
                nn::init::orthogonal_(parameter.value());
            }
        }
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

std::vector<torch::Tensor> NNBase::forward_gru(torch::Tensor &x,
                                               torch::Tensor &hxs,
                                               torch::Tensor &masks)
{
    if (x.size(0) == hxs.size(0))
    {
        auto gru_output = gru->forward(x.unsqueeze(0),
                                       (hxs * masks).unsqueeze(0));
        return {gru_output.output.squeeze(0), gru_output.state.squeeze(0)};
    }
    else
    {
        // x is a (timesteps, agents, -1) tensor that has been flattened to
        // (timesteps * agents, -1)
        auto agents = hxs.size(0);
        auto timesteps = x.size(0) / agents;

        // Unflatten
        x = x.view({timesteps, agents, x.size(1)});

        // Same for masks
        masks = masks.view({timesteps, agents});

        // Figure out which steps in the sequence have a zero for any agent
        // We assume the first timestep has a zero in it
        auto has_zeros = (masks.index({torch::arange(1, masks.size(0), TensorOptions(ScalarType::Long))}) == 0)
                             .any()
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

        hxs = hxs.unsqueeze(0);
        std::vector<torch::Tensor> outputs;
        for (unsigned int i = 0; i < has_zeros_vec.size() - 1; ++i)
        {
            // We can now process steps that don't have any zeros in the masks
            // together.
            // Apparently this is much faster?
            auto start_idx = has_zeros_vec[i];
            auto end_idx = has_zeros_vec[i + 1];

            auto gru_output = gru(
                x.index(torch::arange(start_idx,
                                      end_idx,
                                      TensorOptions(ScalarType::Long))),
                hxs * masks[start_idx].view({1, -1, 1}));

            outputs.push_back(gru_output.output);
        }

        // x is a (timesteps, agents, -1) tensor
        x = torch::cat(outputs);
        hxs = hxs.squeeze(0);

        return {x, hxs};
    }
}

PolicyImpl::PolicyImpl(c10::IntArrayRef /*observation_shape*/,
                       ActionSpace /*action_space*/,
                       std::shared_ptr<NNBase> base) : base(base) {}

std::vector<torch::Tensor> PolicyImpl::act(torch::Tensor & /*inputs*/,
                                           torch::Tensor & /*rnn_hxs*/,
                                           torch::Tensor & /*masks*/) const
{
    return std::vector<torch::Tensor>();
}

std::vector<torch::Tensor> PolicyImpl::evaluate_actions(torch::Tensor & /*inputs*/,
                                                        torch::Tensor & /*rnn_hxs*/,
                                                        torch::Tensor & /*masks*/,
                                                        torch::Tensor & /*actions*/) const
{
    return std::vector<torch::Tensor>();
}

torch::Tensor PolicyImpl::get_values(torch::Tensor & /*inputs*/,
                                     torch::Tensor & /*rnn_hxs*/,
                                     torch::Tensor & /*masks*/) const
{
    return torch::Tensor();
}

CnnBase::CnnBase(unsigned int num_inputs,
                 bool recurrent,
                 unsigned int hidden_size)
    : NNBase(recurrent, num_inputs, hidden_size) {}

std::vector<torch::Tensor> CnnBase::forward(torch::Tensor & /*inputs*/,
                                            torch::Tensor & /*hxs*/,
                                            torch::Tensor & /*masks*/) const
{
    return std::vector<torch::Tensor>();
}

MlpBase::MlpBase(unsigned int num_inputs,
                 bool recurrent,
                 unsigned int hidden_size)
    : NNBase(recurrent, num_inputs, hidden_size) {}

std::vector<torch::Tensor> MlpBase::forward(torch::Tensor & /*inputs*/,
                                            torch::Tensor & /*hxs*/,
                                            torch::Tensor & /*masks*/) const
{
    return std::vector<torch::Tensor>();
}

TEST_CASE("NNBase")
{
    auto base = std::make_shared<NNBase>(true, 5, 10);

    SUBCASE("Bias weights are initialized to 0")
    {
        for (const auto &parameter : base->named_modules()["gru"]->named_parameters())
        {
            if (parameter.key().find("bias") != std::string::npos)
            {
                CHECK(parameter.value()[0].item().toDouble() == doctest::Approx(0));
            }
        }
    }

    SUBCASE("forward_gru() outputs correct shapes when given samples from one"
            " agent")
    {
        auto inputs = torch::rand({4, 5});
        auto hxs = torch::rand({4, 10});
        auto masks = torch::zeros({4, 1});
        auto outputs = base->forward_gru(inputs, hxs, masks);

        REQUIRE(outputs.size() == 2);

        // x
        CHECK(outputs[0].size(0) == 4);
        CHECK(outputs[0].size(1) == 10);

        // hxs
        CHECK(outputs[1].size(0) == 4);
        CHECK(outputs[1].size(1) == 10);
    }

    SUBCASE("forward_gru() outputs correct shapes when given samples from "
            "multiple agents")
    {
        auto inputs = torch::rand({12, 5});
        auto hxs = torch::rand({4, 10});
        auto masks = torch::zeros({12, 1});
        auto outputs = base->forward_gru(inputs, hxs, masks);

        REQUIRE(outputs.size() == 2);

        // x
        CHECK(outputs[0].size(0) == 12);
        CHECK(outputs[0].size(1) == 10);

        // hxs
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
        auto hxs = torch::rand({4, 10});
        auto masks = torch::zeros({4, 1});
        auto outputs = base.forward(inputs, hxs, masks);

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
        auto hxs = torch::rand({4, 10});
        auto masks = torch::zeros({4, 1});
        auto outputs = base->forward(inputs, hxs, masks);

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
        auto hxs = torch::rand({4, 10});
        auto masks = torch::zeros({4, 1});
        auto outputs = policy->act(inputs, hxs, masks);

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
        auto hxs = torch::rand({4, 10});
        auto masks = torch::zeros({4, 1});
        auto actions = torch::rand({4, 5});
        auto outputs = policy->evaluate_actions(inputs, hxs, masks, actions);

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
        auto hxs = torch::rand({4, 10});
        auto masks = torch::zeros({4, 1});
        auto outputs = policy->get_values(inputs, hxs, masks);

        // Value
        CHECK(outputs.size(0) == 4);
        CHECK(outputs.size(1) == 1);
    }
}
}