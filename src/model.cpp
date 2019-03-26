#include <memory>

#include <torch/torch.h>

#include "third_party/doctest.h"
#include "cpprl/model.h"
#include "cpprl/spaces.h"

namespace cpprl
{
int NNBase::recurrent_hidden_state_size() const
{
    if (recurrent)
    {
        return hidden_size;
    }
    return 1;
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
        for (const auto &pair : gru->named_parameters())
        {
            if (pair.key().find("bias") != std::string::npos)
            {
                nn::init::constant_(pair.value(), 0);
            }
            else if (pair.key().find("weight") != std::string::npos)
            {
                nn::init::orthogonal_(pair.value());
            }
        }
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

TEST_CASE("MlpBase")
{
    MlpBase base = MlpBase(5, true, 10);

    SUBCASE("Sanity checks")
    {
        CHECK(base.is_recurrent() == true);
        CHECK(base.recurrent_hidden_state_size() == 10);
    }

    SUBCASE("Output tensors are correct sizes")
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
        CHECK(base->recurrent_hidden_state_size() == 10);
    }

    SUBCASE("Output tensors are correct sizes")
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
    Policy policy = Policy(IntArrayRef{12}, ActionSpace{"Categorical", {5}}, base);

    SUBCASE("Sanity checks")
    {
        CHECK(policy->is_recurrent() == true);
        CHECK(policy->recurrent_hidden_state_size() == 10);
    }

    SUBCASE("act() output tensors are correct sizes")
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

    SUBCASE("evaluate_actions() output tensors are correct sizes")
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

    SUBCASE("get_values() output tensor is correct size")
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