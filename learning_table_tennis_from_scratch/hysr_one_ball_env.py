import o80
import pam_interface
import math
import gym
import time
import json
import numpy as np
from .hysr_one_ball import HysrOneBall, HysrOneBallConfig
from .rewards import JsonReward
from collections import OrderedDict
from baselines import logger


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
        self._values[name] = np.zeros(size)

    def get_gym_box(self):
        size = sum([b.size for b in self._obs_boxes.values()])
        return gym.spaces.Box(low=0.0, high=1.0, shape=(size,), dtype=np.float)

    def set_values(self, name, values):
        normalize = self._obs_boxes[name].normalize
        values_ = np.array(list(map(normalize, values)))
        self._values[name] = values_

    def set_values_pressures(self, name, values, env):
        for dof in range(env._nb_dofs):
            values[2 * dof] = env._reverse_scale_pressure(dof, True, values[2 * dof])
            values[2 * dof + 1] = env._reverse_scale_pressure(
                dof, False, values[2 * dof + 1]
            )
        values_ = np.array(values)
        self._values[name] = values_

    def set_values_non_norm(self, name, values):
        values_ = np.array(values)
        self._values[name] = values_

    def get_normalized_values(self):
        values = list(self._values.values())
        r = np.concatenate(values)
        return r


class HysrOneBallEnv(gym.Env):
    def __init__(
        self,
        reward_config_file=None,
        hysr_one_ball_config_file=None,
        log_episodes=False,
        log_tensorboard=False,
    ):

        super().__init__()

        self._log_episodes = log_episodes
        self._log_tensorboard = log_tensorboard

        hysr_one_ball_config = HysrOneBallConfig.from_json(hysr_one_ball_config_file)

        reward_function = JsonReward.get(reward_config_file)

        self._config = pam_interface.JsonConfiguration(
            hysr_one_ball_config.pam_config_file
        )
        self._nb_dofs = len(self._config.max_pressures_ago)
        self._algo_time_step = hysr_one_ball_config.algo_time_step
        self._pressure_change_range = hysr_one_ball_config.pressure_change_range
        self._accelerated_time = hysr_one_ball_config.accelerated_time

        self._hysr = HysrOneBall(hysr_one_ball_config, reward_function)

        self.action_space = gym.spaces.Box(
            low=-1.0, high=+1.0, shape=(self._nb_dofs * 2,), dtype=np.float
        )

        self._obs_boxes = _ObservationSpace()

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

        self.observation_space = self._obs_boxes.get_gym_box()

        if not self._accelerated_time:
            self._frequency_manager = o80.FrequencyManager(
                1.0 / hysr_one_ball_config.algo_time_step
            )

        self.n_eps = 0
        self.init_episode()

    def init_episode(self):
        self.n_steps = 0
        if self._log_episodes:
            self.data_buffer = []

        # initialize initial action (for action diffs)
        self.last_action = np.zeros(self._nb_dofs * 2)
        for dof in range(self._nb_dofs):
            self.last_action[2 * dof] = self._reverse_scale_pressure(dof, True, self._hysr._reference_posture[dof][0])
            self.last_action[2 * dof + 1] = self._reverse_scale_pressure(dof, False, self._hysr._reference_posture[dof][1])

            

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

    def step(self, action):

        if not self._accelerated_time and self._frequency_manager is None:
            self._frequency_manager = o80.FrequencyManager(1.0 / self._algo_time_step)

        action_orig = action.copy()

        # casting similar to old code
        action_diffs_factor = self._pressure_change_range/18000
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
        observation, reward, episode_over = self._hysr.step(list(action))

        # formatting observation in a format suitable for gym
        observation = self._convert_observation(observation)

        # imposing frequency to learning agent
        if not self._accelerated_time:
            self._frequency_manager.wait()

        # Ignore steps after hitting the ball
        if not episode_over and not self._hysr._ball_status.min_distance_ball_racket:
            return self.step(action_orig)

        # logging
        self.n_steps += 1
        if self._log_episodes:
            self.data_buffer.append(
                (
                    observation.copy(),
                    action_orig,
                    action_casted,
                    action.copy(),
                    reward,
                    episode_over,
                )
            )
        if episode_over:
            if self._log_episodes:
                self.dump_data(self.data_buffer)
            self.n_eps += 1
            if self._log_tensorboard:
                logger.logkv("eprew", reward)
                logger.dumpkvs()

        return observation, reward, episode_over, {}

    def reset(self):
        self.init_episode()
        observation = self._hysr.reset()
        observation = self._convert_observation(observation)
        if not self._accelerated_time:
            self._frequency_manager = None
        return observation

    def dump_data(self, data_buffer):
        filename = "/tmp/ep_" + time.strftime("%Y%m%d-%H%M%S")
        dict_data = dict()
        with open(filename, "w") as json_data:
            dict_data["ob"] = [x[0].tolist() for x in data_buffer]
            dict_data["action_orig"] = [x[1].tolist() for x in data_buffer]
            dict_data["action_casted"] = [x[2] for x in data_buffer]
            dict_data["prdes"] = [x[3] for x in data_buffer]
            dict_data["reward"] = [x[4] for x in data_buffer]
            dict_data["episode_over"] = [x[5] for x in data_buffer]
            json.dump(dict_data, json_data)
