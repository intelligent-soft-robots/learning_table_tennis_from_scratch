import sys
import os
import json


def _no_hit_reward(min_distance_ball_racket):
    return -min_distance_ball_racket


def _return_task_reward(min_distance_ball_target, c, rtt_cap):
    distance_reward_baseline = 3.0
    reward = 1.0 - ((min_distance_ball_target / c) ** 0.75)
    reward = max(reward, rtt_cap)
    return reward


def _smash_task_reward(min_distance_ball_target, max_ball_velocity, c, rtt_cap):
    distance_reward_baseline = 3.0
    reward = 1.0 - ((min_distance_ball_target / c) ** 0.75)
    reward = reward * max_ball_velocity
    reward = max(reward, rtt_cap)
    return reward


def _compute_reward(
    smash,
    min_distance_ball_racket,
    min_distance_ball_target,
    max_ball_velocity,
    c,
    rtt_cap,
):

    # i.e. the ball did not hit the racket,
    # so computing a reward based on the minimum
    # distance between the racket and the ball
    if min_distance_ball_racket is not None:
        return _no_hit_reward(min_distance_ball_racket)

    # the ball did hit the racket, so computing
    # a reward based on the ball / target

    if smash:
        return _smash_task_reward(
            min_distance_ball_target, max_ball_velocity, c, rtt_cap
        )

    else:
        return _return_task_reward(min_distance_ball_target, c, rtt_cap)


def _reward(
    min_distance_ball_racket, min_distance_ball_target, max_ball_velocity, c, rtt_cap
):
    return _compute_reward(
        False,
        min_distance_ball_racket,
        min_distance_ball_target,
        max_ball_velocity,
        c,
        rtt_cap,
    )


def _smash_reward(
    min_distance_ball_racket, min_distance_ball_target, max_ball_velocity, c, rtt_cap
):
    return _compute_reward(
        True,
        min_distance_ball_racket,
        min_distance_ball_target,
        max_ball_velocity,
        c,
        rtt_cap,
    )


class RewardConfig:
    def __init__(self, normalization_constant=3.0, rtt_cap=-0.2):
        self.normalization_constant = normalization_constant
        self.rtt_cap = rtt_cap


class Reward:
    def __init__(self, config):
        self.config = config

    def __call__(
        self, min_distance_ball_racket, min_distance_ball_target, max_ball_velocity
    ):
        return _reward(
            min_distance_ball_racket,
            min_distance_ball_target,
            max_ball_velocity,
            self.config.normalization_constant,
            self.config.rtt_cap,
        )


class SmashReward:
    def __init__(self, config):
        self.config = config

    def __call__(
        self, min_distance_ball_racket, min_distance_ball_target, max_ball_velocity
    ):
        return _smash_reward(
            min_distance_ball_racket,
            min_distance_ball_target,
            max_ball_velocity,
            self.config.normalization_constant,
            self.config.rtt_cap,
        )


class JsonReward:
    @staticmethod
    def get(jsonpath):
        if not os.path.isfile(jsonpath):
            raise FileNotFoundError(
                "failed to find reward configuration file: {}".format(jsonpath)
            )
        try:
            with open(jsonpath) as f:
                conf = json.load(f)
        except Exception as e:
            raise ValueError(
                "failed to parse reward json configuration file {}: {}".format(
                    jsonpath, e
                )
            )
        for attr in ("smash", "normalization_constant", "rtt_cap"):
            if attr not in conf:
                raise ValueError(
                    "failed to find the attribute {} "
                    "in the json reward configuration "
                    "file: {}".format(attr, jsonpath)
                )
        smash = conf["smash"]
        normalization_constant = conf["normalization_constant"]
        rtt_cap = conf["rtt_cap"]
        config = RewardConfig(normalization_constant, rtt_cap)
        if smash:
            return SmashReward(config)
        return Reward(config)

    @staticmethod
    def default_path():
        return os.path.join(
            sys.prefix,
            "learning_table_tennis_from_scratch_config",
            "reward_default.json",
        )
