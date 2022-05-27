import json
import math
import time
from typing import Dict, Union
from collections import OrderedDict
from typing import Any, Dict, Optional, Union

import gym
import numpy as np
import o80
import pam_interface

from .hysr_one_ball import HysrOneBall, HysrOneBallConfig
from .rewards import JsonReward


class _ObservationSpace:

    # the model does not support gym Dict or Tuple spaces
    # which is very inconvenient. This class implements
    # something similar to a Dict space, but which can
    # be casted to a box space.

    class Box:
        def __init__(self, low, high, size):
            self.low = low
            self.high = high
            self.size = size

        def normalize(self, value):
            return (value - self.low) / (self.high - self.low)

        def denormalize(self, value):
            return self.low + value * (self.high - self.low)

    def __init__(self):
        self._obs_boxes = OrderedDict()
        self._values = OrderedDict()

    def add_box(self, name, low, high, size):
        self._obs_boxes[name] = _ObservationSpace.Box(low, high, size)
        self._values[name] = np.zeros(size, dtype=np.float32)

    def get_gym_box(self):
        size = sum([b.size for b in self._obs_boxes.values()])
        return gym.spaces.Box(low=0.0, high=1.0, shape=(size,), dtype=np.float32)

    def get_gym_box_extra_obs(self, nb_extra_balls):
        size = sum([b.size for b in self._obs_boxes.values()])
        return gym.spaces.Box(low=0.0, high=1.0, shape=(nb_extra_balls, size), dtype=np.float32)

    def set_values(self, name, values):
        normalize = self._obs_boxes[name].normalize
        values_ = np.array(list(map(normalize, values)), dtype=np.float32)
        self._values[name] = values_

    def set_values_pressures(self, name, values, env):
        for dof in range(env._nb_dofs):
            values[2 * dof] = env._reverse_scale_pressure(dof, True, values[2 * dof])
            values[2 * dof + 1] = env._reverse_scale_pressure(
                dof, False, values[2 * dof + 1]
            )
        values_ = np.array(values, dtype=np.float32)
        self._values[name] = values_

    def set_values_non_norm(self, name, values):
        values_ = np.array(values, dtype=np.float32)
        self._values[name] = values_

    def set_values_non_array(self, name, values):
        self._values[name] = values

    def get_normalized_values(self):
        values = list(self._values.values())
        r = np.concatenate(values)
        r = np.array(r, dtype=np.float32)
        return r


class HysrManyBallEnv(gym.Env):
    def __init__(
        self,
        reward_config_file=None,
        hysr_one_ball_config_file=None,
        log_episodes=False,
        logger=None,
    ):

        super().__init__()

        self._log_episodes = log_episodes
        self._logger = logger

        hysr_one_ball_config = HysrOneBallConfig.from_json(hysr_one_ball_config_file)

        reward_function = JsonReward.get(reward_config_file)

        self._config = pam_interface.JsonConfiguration(
            str(hysr_one_ball_config.pam_config_file)
        )
        self._nb_dofs = len(self._config.max_pressures_ago)
        self._algo_time_step = hysr_one_ball_config.algo_time_step
        self._pressure_change_range = hysr_one_ball_config.pressure_change_range
        self._accelerated_time = hysr_one_ball_config.accelerated_time

        self._hysr = HysrOneBall(hysr_one_ball_config, reward_function)

        self.action_space = gym.spaces.Box(
            low=-1.0, high=+1.0, shape=(self._nb_dofs * 2,), dtype=np.float32
        )

        self._obs_boxes = _ObservationSpace()
        self._hs_boxes = _ObservationSpace()

        self._obs_boxes.add_box("robot_position", -math.pi, +math.pi, self._nb_dofs)
        self._obs_boxes.add_box("robot_velocity", 0.0, 10.0, self._nb_dofs)
        self._obs_boxes.add_box(
            "robot_pressure",
            self._config.min_pressure(),
            self._config.max_pressure(),
            self._nb_dofs * 2,
        )

        self._obs_boxes.add_box(
            "ball_position",
            min(hysr_one_ball_config.world_boundaries["min"]),
            max(hysr_one_ball_config.world_boundaries["max"]),
            3,
        )
        self._obs_boxes.add_box("ball_velocity", -10.0, +10.0, 3)

        self.observation_space = gym.spaces.Dict(
            {
            "observation": self._obs_boxes.get_gym_box(),
            }
        )

        if not self._accelerated_time:
            self._frequency_manager = o80.FrequencyManager(
                1.0 / hysr_one_ball_config.algo_time_step
            )

        self.n_eps = 0
        self.init_episode()

    def init_episode(self):
        self.n_steps = 0
        self.data_buffer = []
        self.extra_data_buffer = [[] for _ in range(self._hysr._hysr_config.extra_balls_per_set)]

        # initialize initial action (for action diffs)
        self.last_action = np.zeros(self._nb_dofs * 2, dtype=np.float32)
        starting_pressures = self._hysr.get_starting_pressures()
        for dof in range(self._nb_dofs):
            self.last_action[2 * dof] = self._reverse_scale_pressure(
                dof, True, starting_pressures[dof][0]
            )
            self.last_action[2 * dof + 1] = self._reverse_scale_pressure(
                dof, False, starting_pressures[dof][1]
            )

    def _bound_pressure(self, dof, ago, value):
        if ago:
            return int(
                max(
                    min(value, self._config.max_pressures_ago[dof]),
                    self._config.min_pressures_ago[dof],
                )
            )
        else:
            return int(
                max(
                    min(value, self._config.max_pressures_antago[dof]),
                    self._config.min_pressures_antago[dof],
                )
            )

    def _scale_pressure(self, dof, ago, value):
        if ago:
            return (
                value
                * (
                    self._config.max_pressures_ago[dof]
                    - self._config.min_pressures_ago[dof]
                )
                + self._config.min_pressures_ago[dof]
            )
        else:
            return (
                value
                * (
                    self._config.max_pressures_antago[dof]
                    - self._config.min_pressures_antago[dof]
                )
                + self._config.min_pressures_antago[dof]
            )

    def _reverse_scale_pressure(self, dof, ago, value):
        if ago:
            return (value - self._config.min_pressures_ago[dof]) / (
                self._config.max_pressures_ago[dof]
                - self._config.min_pressures_ago[dof]
            )
        else:
            return (value - self._config.min_pressures_antago[dof]) / (
                self._config.max_pressures_antago[dof]
                - self._config.min_pressures_antago[dof]
            )

    def _convert_observation(self, observation):
        self._obs_boxes.set_values_non_norm(
            "robot_position", observation.joint_positions
        )
        self._obs_boxes.set_values_non_norm(
            "robot_velocity", observation.joint_velocities
        )
        self._obs_boxes.set_values_pressures(
            "robot_pressure", observation.pressures, self
        )
        self._obs_boxes.set_values_non_norm("ball_position", observation.ball_position)
        self._obs_boxes.set_values_non_norm("ball_velocity", observation.ball_velocity)
        return self._obs_boxes.get_normalized_values()

    def _get_obs(self, state) -> Dict[str, Union[int, np.ndarray]]:
            """
            Helper to create the observation.

            :return: The current observation.
            """
            return OrderedDict(
                [
                    ("observation", self._convert_observation(state)),
                ]
            )

    def _get_extra_obs(self, extra_states) -> Dict[str, Union[int, np.ndarray]]:
            """
            Helper to create the observation.

            :return: The current observation.
            """
            return [OrderedDict(
                [
                    ("observation", self._convert_observation(extra_state) ),
                ]
            )
                for extra_state in extra_states
            ]


    # remove transitions between the ball hitting the racket and the ball hitting the table as well as transitions after the end of the episode
    def get_reduced_episodes(self):
        self.extra_data_buffer.append(self.data_buffer)     # put main ball in extra ball buffer (to avoid code repetition)
        all_trans = [[] for _ in range(self._hysr._hysr_config.extra_balls_per_set + 1)]
        idx = 0
        for data_buffer in self.extra_data_buffer:
            obs_before_racket_hit = None
            action_before_racket_hit = None
            for obs, action, reward, episode_over, previous_obs, min_distance_ball_racket in data_buffer:
                if not episode_over and min_distance_ball_racket:   # normal transition
                    all_trans[idx].append((previous_obs, obs, action, reward, episode_over, [{}]))
                    obs_after_racket_hit = obs
                elif not episode_over and not self._hysr._ball_status.min_distance_ball_racket:   # ball hit racket, but didn't cross table plane (episode_over is FALSE) -> do not add
                    if not obs_after_racket_hit:
                        obs_before_racket_hit = previous_obs
                        action_before_racket_hit = action
                elif episode_over and obs_before_racket_hit:  # episode over and ball hit
                    all_trans[idx].append((obs_before_racket_hit, obs, action_before_racket_hit, reward, episode_over, [{}]))
                    break
                elif episode_over and not obs_before_racket_hit:  # episode over and ball not hit
                    all_trans[idx].append((previous_obs, obs, action, reward, episode_over, [{}]))
                    break
            idx += 1

        trans = all_trans[-1]     # main ball
        extra_trans = all_trans[:-1]    # extra balls

        return trans, extra_trans


    def step(self, action):

        if not self._accelerated_time and self._frequency_manager is None:
            self._frequency_manager = o80.FrequencyManager(1.0 / self._algo_time_step)

        action_orig = action.copy()

        # casting similar to old code
        action_diffs_factor = self._pressure_change_range / 18000
        action = action * action_diffs_factor
        action_sigmoid = [1 / (1 + np.exp(-a)) - 0.5 for a in action]
        action = [
            np.clip(a1 + a2, 0, 1) for a1, a2 in zip(self.last_action, action_sigmoid)
        ]
        self.last_action = action.copy()
        action_casted = action.copy()

        # put pressure in range as defined in parameters file
        for dof in range(self._nb_dofs):
            p_plus = 0
            p_minus = 0
            action[2 * dof] = self._scale_pressure(dof, True, action[2 * dof]) + p_plus
            action[2 * dof + 1] = (
                self._scale_pressure(dof, False, action[2 * dof + 1]) + p_minus
            )

        # final target pressure (make sure that it is within bounds)
        for dof in range(self._nb_dofs):
            action[2 * dof] = self._bound_pressure(dof, True, action[2 * dof])
            action[2 * dof + 1] = self._bound_pressure(dof, False, action[2 * dof + 1])

        # hysr takes a list of int, not float, as input
        action = [int(a) for a in action]

        # performing a step
        observation, reward, episode_over, extra_observations, extra_rewards, extra_dones = self._hysr.step(list(action))

        # imposing frequency to learning agent
        if not self._accelerated_time:
            self._frequency_manager.wait()

        # Ignore steps after hitting/missing all balls
        idx_ball_still_active = -1
        if not episode_over and not self._hysr._ball_status.min_distance_ball_racket:
            idx = 0
            for episode_over, min_distance_ball_racket in zip(extra_dones, self._hysr.extra_min_distance_ball_racket):
                if not episode_over and min_distance_ball_racket:
                    idx_ball_still_active = idx
                    break
                idx += 1
            
            if idx_ball_still_active == -1: # non of the balls still can be hit anymore 
                return self.step(action_orig)

        obs = self._get_obs(observation)
        extra_obs = self._get_extra_obs(extra_observations)

        # logging
        self.n_steps += 1
        self.data_buffer.append(
            (
                obs,
                action_orig,
                reward,
                episode_over,
                self.previous_obs,
                self._hysr._ball_status.min_distance_ball_racket
            )
        )

        # add extra transitions
        idx = 0
        for extra_ob, extra_reward, extra_episode_over, extra_previous_obs, extra_min_distance_ball_racket in \
            zip(extra_obs, extra_rewards, extra_dones, self.previous_extra_obs, self._hysr.extra_min_distance_ball_racket):
            self.extra_data_buffer[idx].append(
                (extra_ob,
                action_orig,
                extra_reward,
                extra_episode_over,
                extra_previous_obs,
                extra_min_distance_ball_racket)
            )
            idx += 1

        infos = {}

        self.previous_extra_obs = extra_obs
        self.previous_obs = obs           

        all_episodes_over = episode_over and all(extra_dones)

        if all_episodes_over:
            infos["trajectory"], infos["hsm_trajectories"] = self.get_reduced_episodes()
            if self._log_episodes:
                self.dump_data(self.data_buffer)
            if self._logger:
                self._logger.record("eprew", reward)
                self._logger.dump()
            self.n_eps += 1

        # use different ball for observation if main ball not active anymore
        if idx_ball_still_active!=-1:
            obs = extra_obs[idx_ball_still_active]
            reward = extra_rewards[idx_ball_still_active]

        if not all_episodes_over:
            reward = 0

        return obs, reward, all_episodes_over, infos

    def reset(self):
        self.init_episode()
        observation, extra_observations = self._hysr.reset()
        obs = self._get_obs(observation)
        extra_obs = self._get_extra_obs(extra_observations)
        if not self._accelerated_time:
            self._frequency_manager = None
        self.previous_extra_obs = extra_obs
        self.previous_obs = obs
        return obs

    def dump_data(self, data_buffer):
        filename = "/tmp/ep_" + time.strftime("%Y%m%d-%H%M%S")
        dict_data = dict()
        with open(filename, "w") as json_data:
            dict_data["ob"] = [x[0].tolist() for x in data_buffer]
            dict_data["action_orig"] = [x[1].tolist() for x in data_buffer]
            dict_data["reward"] = [x[2] for x in data_buffer]
            dict_data["episode_over"] = [x[3] for x in data_buffer]
            json.dump(dict_data, json_data)
