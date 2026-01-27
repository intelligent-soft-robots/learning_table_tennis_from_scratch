"""
Compatibility module for stable-baselines3 2.x (gymnasium) and 1.x fork (gym).
"""
from collections import OrderedDict
import stable_baselines3

_version_parts = stable_baselines3.__version__.replace('a', '.').split('.')[:2]
SB3_VERSION = tuple(map(int, _version_parts))
USE_GYMNASIUM = SB3_VERSION >= (2, 0)

if USE_GYMNASIUM:
    import gymnasium as gym
else:
    import gym


def get_observation_space(base_box):
    if USE_GYMNASIUM:
        return base_box
    else:
        return gym.spaces.Dict({"observation": base_box})


def make_obs(observation):
    if USE_GYMNASIUM:
        return observation
    else:
        return OrderedDict([("observation", observation)])


def make_obs_list(observations):
    if USE_GYMNASIUM:
        return observations
    else:
        return [OrderedDict([("observation", obs)]) for obs in observations]


def get_obs_array(obs):
    if isinstance(obs, (dict, OrderedDict)):
        return obs["observation"]
    return obs


def make_step_return(obs, reward, done, info):
    if USE_GYMNASIUM:
        return obs, reward, done, False, info
    else:
        return obs, reward, done, info


def make_reset_return(obs):
    if USE_GYMNASIUM:
        return obs, {}
    else:
        return obs
