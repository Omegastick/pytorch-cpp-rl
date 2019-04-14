# CppRl - PyTorch C++ Reinforcement Learning
![Linux build status](https://travis-ci.org/Omegastick/pytorch-cpp-rl.svg?branch=master) [![Windows build status](https://ci.appveyor.com/api/projects/status/ch64s3twohboxkqt?svg=true)](https://ci.appveyor.com/project/Omegastick/pytorch-cpp-rl)

![LunarLander-v2](imgs/lunar_lander.gif)
Above: results on LunarLander-v2 after 60 seconds of training on my laptop

**CppRl is a reinforcement learning framework, written using the [PyTorch C++ frontend](https://pytorch.org/cppdocs/frontend.html).**

It is *very* heavily based on [Ikostrikov's wonderful pytorch-a2c-ppo-acktr-gail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail). You could even consider this a port. The API and underlying algorithms are almost identical (with the necessary changes involved in the move to C++).

It also contains an implementation of a simple OpenAI Gym server that communicates via [ZeroMQ](http://zeromq.org/) to test the framework on Gym environments.

CppRl aims to be an extensible, reasonably optimized, production-ready framework for using reinforcement learning in projects where Python isn't viable. It should be ready to use in desktop applications on user's computers with minimal setup required on the user's side.

## Motivation
At the time of writing, there are no general-use reinforcement learning frameworks for C++. I needed one for a personal project, and the PyTorch C++ frontend had recently been released, so I figured I should make one.

## Features
- Implemented algorithms:
  - A2C
  - PPO
- Recurrent policies (GRU based)
- Cross-platform compatibility (tested on Windows 10, Ubuntu 16.04, and Ubuntu 18.04)
- Solid test coverage
- Decently optimized (always open to pull requests improving optimization though)

## Example
An example that uses the included OpenAI Gym server is provided in `example`. It can be run as follows:
Terminal 1:
```bash
./launch_gym_server.py
```
Terminal 2:
```bash
build/example/gym_server
```

It takes about 60 seconds to train an agent to 200 average reward on my laptop (i7-8550U processor).

The environment and hyperparameters can be set in `example/gym_client.cpp`.

Note: The Gym server and client aren't very well optimized, especially when it comes to environments with image observations. There are a few extra copies necessitated by using an inter-process communication system, and then `gym_client.cpp` has an extra copy or two to turn the observations into PyTorch tensors. This is why the performance isn't that good when compared with Python libraries running Gym environments.

## Building
CMake is used for the build system. 
Most dependencies are included as submodules (run `git submodule update --init --recursive` to get them).
Libtorch has to be [installed seperately](https://pytorch.org/cppdocs/installing.html).

### Linux
```bash
cd pytorch-cpp-rl
mkdir build && cd build
cmake ..
make -j4
```

### Windows
```
cd pytorch-cpp-rl
mkdir build && cd build
cmake -G "Visual Studio 15 2017 Win64" -DCMAKE_PREFIX_PATH=C:/path/to/libtorch ..
cmake --build . --config Release
```
Before running, make sure to add `libtorch/lib` to your `PATH` environment variable.

Windows performance is about 75% that of Linux's at the moment. I'm looking into how to speed things up.

## Testing
You can run the tests with `build/cpprl_tests` (`build/Release/cpprl_tests.exe` on Windows).
