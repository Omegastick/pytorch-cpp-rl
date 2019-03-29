#include <spdlog/spdlog.h>

#include "communicator.h"
#include "requests.h"

using namespace gym_client;

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
    auto observation = communicator.get_response<ResetResponse>()->observation;

    for (int i = 0; i < 1000; ++i)
    {
        auto step_param = std::make_shared<StepParam>();
        step_param->actions = {{1}, {0}, {1}, {0}};
        step_param->render = true;
        Request<StepParam> step_request("step", step_param);
        communicator.send_request(step_request);
        communicator.get_response<StepResponse>();
    }
}