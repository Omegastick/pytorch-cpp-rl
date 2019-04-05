"""
Adapted from:
github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/a2c_ppo_acktr/envs.py

Provides utility functions for making Gym environments.
"""
import gym
from gym.spaces import Box
import numpy as np

from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_normalize import (VecNormalize
                                                    as VecNormalize_)


class TransposeImage(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(TransposeImage, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


class VecNormalize(VecNormalize_):
    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs):
        if self.ob_rms:
            if self.training:
                self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean)
                          / np.sqrt(self.ob_rms.var + self.epsilon),
                          -self.clipob, self.clipob)
        return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        infos = {'reward': np.expand_dims(rews, -1)}
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon),
                           -self.cliprew,
                           self.cliprew)
        self.ret[news] = 0.
        return obs, rews, news, infos


def make_env(env_id, seed, rank):
    def _thunk():
        env = gym.make(env_id)

        env.seed(seed + rank)

        obs_shape = env.observation_space.shape

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = TransposeImage(env)

        return env

    return _thunk


def make_vec_envs(env_name, seed, num_processes, gamma):
    envs = [make_env(env_name, seed, i) for i in range(num_processes)]

    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    if len(envs.observation_space.shape) == 1:
        if gamma is None or gamma == -1:
            envs = VecNormalize(envs, ret=False)
        else:
            envs = VecNormalize(envs, gamma=gamma)

    return envs
