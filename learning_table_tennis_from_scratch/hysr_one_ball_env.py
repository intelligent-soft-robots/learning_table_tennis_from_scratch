import json
import math
import time
from collections import OrderedDict

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

    def get_normalized_values(self):
        values = list(self._values.values())
        r = np.concatenate(values)
        r = np.array(r, dtype=np.float32)
        return r


class HysrOneBallEnv(gym.Env):
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
            min(hysr_one_ball_config.world_boundaries.min),
            max(hysr_one_ball_config.world_boundaries.max),
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
        self.last_action = np.zeros(self._nb_dofs * 2, dtype=np.float32)
        starting_pressures = self._hysr.get_starting_pressures()
        for dof in range(self._nb_dofs):
            self.last_action[2 * dof] = self._reverse_scale_pressure(
                dof, True, starting_pressures[dof][0]
            )
            self.last_action[2 * dof + 1] = self._reverse_scale_pressure(
                dof, False, starting_pressures[dof][1]
            )
        self.joint_penalty = 0.0
        self.accelarated_towards_limits = False

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

        # action[0] = action[0]*32
        # action[1] = action[1]*32

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
        observation, reward, episode_over = self._hysr.step(list(action))

        # give negative reward 0.5 if joint limits are reached (limit joint 1: 100, 2 and 3: 89, 4: no limit)
        if abs(observation.joint_positions[0]) > 100 * np.pi/180 or abs(observation.joint_positions[1]) > 89 * np.pi/180 or abs(observation.joint_positions[2]) > 89 * np.pi/180:
            if self.joint_penalty < -0.1:
                print("joint limits reached", observation.joint_positions)
            self.joint_penalty = -0.25

        # imposing frequency to learning agent
        if not self._accelerated_time:
            self._frequency_manager.wait()

        # Ignore steps after hitting the ball
        if (not episode_over and not self._hysr._ball_status.min_distance_ball_racket) or (not episode_over and observation.ball_position[2] < -1.39 + 0.15 and observation.ball_position[1]<-3.38):

            if not self._hysr._ball_status.min_distance_ball_racket and not self.print_ball_hit:
                print("ball hit")
                self.print_ball_hit = True
            if observation.ball_position[2] < -1.39 + 0.15 and not self.print_ball_below_racket:
                print("ball below racket")
                self.print_ball_below_racket = True

            #  # stop second joint if necessary
            # if (observation.joint_positions[1] > 80 * np.pi/180  and observation.joint_velocities[1]>0.0):
            #     action_orig[2] = 2.0
            #     action_orig[3] = -2.0

            # # stop first joint if necessary
            # if (observation.joint_positions[0] > 60 * np.pi/180  and observation.joint_velocities[0]>0) or observation.joint_velocities[0]>10.0:
            #     self.accelarated_towards_limits = True
            #     if observation.joint_velocities[0]>10.0:
            #         print("pos:", observation.joint_positions[0], "vel:", observation.joint_velocities[0], " -- 6.0")
            #         action_orig[0] = 6.0
            #         action_orig[1] = -6.0
            #     else:
            #         print("pos:", observation.joint_positions[0], "vel:", observation.joint_velocities[0], " -- 3.0")
            #         action_orig[0] = 3.0
            #         action_orig[1] = -3.0
            # elif (observation.joint_positions[0] > 10 * np.pi/180  and observation.joint_velocities[0]>6.0):
            #     print("pos:", observation.joint_positions[0], "vel:", observation.joint_velocities[0], " -- 3.0")
            #     action_orig[0] = 1.0
            #     action_orig[1] = -1.0
            #     print("pos:", observation.joint_positions[0], "vel:", observation.joint_velocities[0], " -- 1.0")
            # elif observation.joint_positions[0] < -60 * np.pi/180 and self.accelarated_towards_limits:
            #     print("pos:", observation.joint_positions[0], "vel:", observation.joint_velocities[0], " -- -2.0")
            #     action_orig[0] = -2.0
            #     action_orig[1] = 2.0
            # elif self.accelarated_towards_limits and observation.joint_velocities[0]<-2.0:
            #     print("pos:", observation.joint_positions[0], "vel:", observation.joint_velocities[0]," -- -2.5")
            #     action_orig[0] = -2.5
            #     action_orig[1] = 2.5
            # elif self.accelarated_towards_limits and observation.joint_velocities[0]>6.0:
            #     print("pos:", observation.joint_positions[0], "vel:", observation.joint_velocities[0]," -- 1.5")
            #     action_orig[0] = 1.5
            #     action_orig[1] = -1.5
            # else:
            #     action_orig[0] = 0.0
            #     action_orig[1] = 0.0
            #     if self.accelarated_towards_limits:
            #         print("pos:", observation.joint_positions[0], "vel:", observation.joint_velocities[0], " -- 0.0")

                        # stop first joint if necessary
            

            self.last_action = np.ones(self._nb_dofs * 2, dtype=np.float32) * 0.29  # 0.29 = 1.5 bar, 0.5 = 2.0 bar
            action_orig = np.zeros(self._nb_dofs * 2, dtype=np.float32)

            if (observation.joint_positions[0] > 60 * np.pi/180  and observation.joint_velocities[0]>0) or observation.joint_velocities[0]>10.0:
                self.accelarated_towards_limits = True
                if observation.joint_velocities[0]>10.0:
                    print("pos:", observation.joint_positions[0], "vel:", observation.joint_velocities[0], " -- 6.0")
                    self.last_action[0] = 0.29
                    self.last_action[1] = 0
                else:
                    print("pos:", observation.joint_positions[0], "vel:", observation.joint_velocities[0], " -- 3.0")
                    self.last_action[0] = 0.29
                    self.last_action[1] = 0.1
            elif (observation.joint_positions[0] > 10 * np.pi/180  and observation.joint_velocities[0]>6.0):
                print("pos:", observation.joint_positions[0], "vel:", observation.joint_velocities[0], " -- 3.0")
                self.last_action[0] = 0.29
                self.last_action[1] = 0.2
                print("pos:", observation.joint_positions[0], "vel:", observation.joint_velocities[0], " -- 1.0")
            elif observation.joint_positions[0] < -60 * np.pi/180 and self.accelarated_towards_limits:
                print("pos:", observation.joint_positions[0], "vel:", observation.joint_velocities[0], " -- -2.0")
                self.last_action[0] = 0.2
                self.last_action[1] = 0.29
            elif self.accelarated_towards_limits and observation.joint_velocities[0]<-2.0:
                print("pos:", observation.joint_positions[0], "vel:", observation.joint_velocities[0]," -- -2.5")
                self.last_action[0] = 0.15
                self.last_action[1] = 0.29
            elif self.accelarated_towards_limits and observation.joint_velocities[0]>6.0:
                print("pos:", observation.joint_positions[0], "vel:", observation.joint_velocities[0]," -- 1.5")
                self.last_action[0] = 0.29
                self.last_action[1] = 0.15
            else:
                self.last_action[0] = 0.29
                self.last_action[1] = 0.29
                if self.accelarated_towards_limits:
                    print("pos:", observation.joint_positions[0], "vel:", observation.joint_velocities[0], " -- 0.0")




            # # stop first joint if necessary
            # if (observation.joint_positions[0] > 60 * np.pi/180  and observation.joint_velocities[0]>0) or observation.joint_velocities[0]>10.0:
            #     self.accelarated_towards_limits = True
            #     self.last_action = np.ones(self._nb_dofs * 2, dtype=np.float32) * 0.5  # 0.29 = 1.5 bar, 0.5 = 2.0 bar
            #     # self.last_action[0] = 0.29
            #     action_orig = np.zeros(self._nb_dofs * 2, dtype=np.float32)

            return self.step(action_orig)
        
        # if not episode_over and observation.ball_position[2] < -1.39 + 0.15:
        #     self.last_action = np.ones(self._nb_dofs * 2, dtype=np.float32) * 0.5
        #     action_orig = np.zeros(self._nb_dofs * 2, dtype=np.float32)
        #     return self.step(action_orig)

        
         # formatting observation in a format suitable for gym
        observation = self._convert_observation(observation)

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
            reward += self.joint_penalty
            if self._log_episodes:
                self.dump_data(self.data_buffer)
            self.n_eps += 1
            if self._logger:
                print("log", "rew:", reward, "n_eps:", self.n_eps, "n_steps:", self.n_steps)
                self._logger.record("eprew", reward - self.joint_penalty)
                self._logger.record("joint_penalty", self.joint_penalty)
                self._logger.record("min_discante_ball_racket", self._hysr._ball_status.min_distance_ball_racket or 0)
                self._logger.record("min_distance_ball_target_capped",
                    min(
                        self._hysr._ball_status.min_distance_ball_target or self._hysr._reward_function.config.normalization_constant,
                        self._hysr._reward_function.config.normalization_constant))
                self._logger.record("max_ball_velocity", self._hysr._ball_status.max_ball_velocity)
                self._logger.dump()

        return observation, reward, episode_over, {}

    def reset(self):
        self.init_episode()
        observation = self._hysr.reset()
        observation = self._convert_observation(observation)
        if not self._accelerated_time:
            self._frequency_manager = None
        self.print_ball_hit = False
        self.print_ball_below_racket = False
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
