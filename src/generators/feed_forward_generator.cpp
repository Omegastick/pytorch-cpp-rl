#include <vector>

#include <torch/torch.h>

#include "cpprl/generators/feed_forward_generator.h"
#include "cpprl/generators/generator.h"
#include "third_party/doctest.h"

namespace cpprl
{
FeedForwardGenerator::FeedForwardGenerator(int /*num_mini_batch*/,
                                           torch::Tensor /*observations*/,
                                           torch::Tensor /*hidden_states*/,
                                           torch::Tensor /*actions*/,
                                           torch::Tensor /*value_predictions*/,
                                           torch::Tensor /*returns*/,
                                           torch::Tensor /*masks*/,
                                           torch::Tensor /*old_action_log_probs*/,
                                           torch::Tensor /*advantages*/) {}

bool FeedForwardGenerator::done() const
{
    return false;
}

MiniBatch FeedForwardGenerator::next()
{
    return MiniBatch(torch::Tensor(), torch::Tensor(), torch::Tensor(),
                     torch::Tensor(), torch::Tensor(), torch::Tensor(),
                     torch::Tensor(), torch::Tensor());
}

TEST_CASE("FeedForwardGenerator")
{
    FeedForwardGenerator generator(3, torch::rand({15, 4}), torch::rand({15, 3}),
                                   torch::rand({15, 1}), torch::rand({15, 1}),
                                   torch::rand({15, 1}), torch::ones({15, 1}),
                                   torch::rand({15, 1}), torch::rand({15, 1}));

    SUBCASE("Minibatch tensors are correct sizes")
    {
        auto minibatch = generator.next();

        CHECK(minibatch.observations.sizes().vec() == std::vector<long>{5, 4});
        CHECK(minibatch.hidden_states.sizes().vec() == std::vector<long>{5, 3});
        CHECK(minibatch.actions.sizes().vec() == std::vector<long>{5, 1});
        CHECK(minibatch.value_predictions.sizes().vec() == std::vector<long>{5, 1});
        CHECK(minibatch.returns.sizes().vec() == std::vector<long>{5, 1});
        CHECK(minibatch.masks.sizes().vec() == std::vector<long>{5, 1});
        CHECK(minibatch.old_action_log_probs.sizes().vec() == std::vector<long>{5, 1});
        CHECK(minibatch.advantages.sizes().vec() == std::vector<long>{5, 1});
    }

    SUBCASE("done() indicates whether the generator has finished")
    {
        CHECK(!generator.done());
        generator.next();
        CHECK(!generator.done());
        generator.next();
        CHECK(!generator.done());
        generator.next();
        CHECK(generator.done());
    }

    SUBCASE("Calling a generator after it has finished throws an exception")
    {
        generator.next();
        generator.next();
        generator.next();
        CHECK_THROWS(generator.next());
    }
}
}