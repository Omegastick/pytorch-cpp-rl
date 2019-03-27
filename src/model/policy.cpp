#include <torch/torch.h>

#include "cpprl/model/policy.h"
#include "cpprl/model/mlpbase.h"
#include "cpprl/spaces.h"
#include "third_party/doctest.h"

using namespace torch;

namespace cpprl
{
PolicyImpl::PolicyImpl(ActionSpace action_space, std::shared_ptr<NNBase> base)
    : base(base)
{
    // int num_outputs;
    if (action_space.type == "Discrete")
    {
        // num_outputs = action_space.shape[0];
        // self.dist = Categorical(self.base.output_size, num_outputs)
    }
    else if (action_space.type == "Box")
    {
        // num_outputs = action_space.shape[0];
        // self.dist = DiagGaussian(self.base.output_size, num_outputs)
    }
    else if (action_space.type == "MultiBinary")
    {
        // num_outputs = action_space.shape[0];
        // self.dist = Bernoulli(self.base.output_size, num_outputs)
    }
    else
    {
        throw std::exception();
    }
}

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

TEST_CASE("Policy")
{
    auto base = std::make_shared<MlpBase>(3, true, 10);
    auto policy = Policy(ActionSpace{"Discrete", {5}}, base);

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
}