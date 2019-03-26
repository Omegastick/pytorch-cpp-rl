#include <iostream>

#include <torch/torch.h>

#include "cpprl/model.h"

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
}