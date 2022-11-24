import json
import math
import time
from collections import OrderedDict

import gymnasium as gym
import numpy as np
import o80
import pam_interface

from .hysr_one_ball import HysrOneBall, HysrOneBallConfig
from .rewards import JsonReward

def sat(x,lmin,lmax):
        y=min(max(x, lmin), lmax)
        return y

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
        self._action_in_state = hysr_one_ball_config.action_in_state
        self._action_repeat_counter = hysr_one_ball_config.action_repeat_counter
        self._pd_control = hysr_one_ball_config.pd_control
        self._pd_control_T = hysr_one_ball_config.pd_control_T
        self._pd_control_K_p = hysr_one_ball_config.pd_control_K_p
        self._pd_control_K_d = hysr_one_ball_config.pd_control_K_d

        self._hysr = HysrOneBall(hysr_one_ball_config, reward_function)

        self.delta_p = hysr_one_ball_config.delta_p
        self.delta_p_p0_is_action = hysr_one_ball_config.delta_p_p0_is_action
        self.delta_p_p0_value = hysr_one_ball_config.delta_p_p0_value
        self.delta_u_init = hysr_one_ball_config.delta_u_init

        self._obs_boxes = _ObservationSpace()

        if (self.delta_p and not self.delta_p_p0_is_action) or self._pd_control:
            self.action_space = gym.spaces.Box(
                low=-1.0, high=+1.0, shape=(self._nb_dofs,), dtype=np.float32
            )
            if self._action_in_state:
                self._obs_boxes.add_box("action_copy", -1, +1, self._nb_dofs)
        else:
            self.action_space = gym.spaces.Box(
                low=-1.0, high=+1.0, shape=(self._nb_dofs * 2,), dtype=np.float32
            )
            if self._action_in_state:
                self._obs_boxes.add_box("action_copy", -1, +1, self._nb_dofs * 2)

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
        self.last_action = self.get_init_action()
        if self._pd_control:
            self.q_target_list = []

    def get_init_action(self):
        init_action = np.zeros(self._nb_dofs * 2, dtype=np.float32)
        starting_pressures = self._hysr.get_starting_pressures()
        for dof in range(self._nb_dofs):
            if self.delta_p:
                if self.delta_p_p0_is_action:
                    init_action[2 * dof] = self.delta_u_init[dof] #+ self.n_eps/1000 * (dof == 1)
                else:
                    init_action[dof] = self.delta_u_init[dof] #+ self.n_eps/1000 * (dof == 1)

            else:
                init_action[2 * dof] = self._reverse_scale_pressure(
                    dof, True, starting_pressures[dof][0]
                )
                init_action[2 * dof + 1] = self._reverse_scale_pressure(
                    dof, False, starting_pressures[dof][1]
                )
        return init_action
        

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

    def _scale_pressure_delta_p(self, dof, ago, u, p0):
        incorr = True
        if ago:
            pmin = self._config.min_pressures_ago[dof]
            pmax = self._config.max_pressures_ago[dof]
        else:
            pmin = self._config.min_pressures_antago[dof]
            pmax = self._config.max_pressures_antago[dof]
        m=pmax-pmin
        if incorr:
            ddp=.5-sat(abs(p0-.5),0,.5)
        else:
            ddp=0
        if ago:
            p=sat(m*(p0+(1-ddp)*sat(u,-1,1))+pmin,pmin,pmax)
        else:
            p=sat(m*(p0-(1-ddp)*sat(u,-1,1))+pmin,pmin,pmax)
        return p


    def _convert_observation(self, observation, action_casted):
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
        if self._action_in_state:
            self._obs_boxes.set_values_non_norm("action_copy", action_casted)
        self.last_observation = self._obs_boxes.get_normalized_values()
        return self.last_observation.copy()

    def step(self, action):
        if not self._accelerated_time and self._frequency_manager is None:
            self._frequency_manager = o80.FrequencyManager(1.0 / self._algo_time_step)

        # pad action with zeros in case of delta_p approach to keep dimension of action
        if self.delta_p and not self.delta_p_p0_is_action:
            action_orig_delta_p = action
            action = np.concatenate([action, np.zeros(np.shape(action))])

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

        if self._pd_control:
            q = self.last_observation[0:4]
            dq = self.last_observation[4:8]
            q_target_from_action_casted = np.array(action_casted[0:4]) * 2 * np.pi - np.pi
            self.q_target_list.append(q_target_from_action_casted)
            if self.n_steps>=self._pd_control_T:
                q_target = self.q_target_list[self.n_steps-self._pd_control_T]
            else:
                q_target = q
            u = self._pd_control_K_p * (q - q_target) + self._pd_control_K_d * dq
            u = np.clip(u, -1, 1)
            action_casted = u
            action = np.concatenate([np.zeros(np.shape(action)), np.zeros(np.shape(action))])

        # put pressure in range as defined in parameters file
        if not self.delta_p and not self._pd_control:
            for dof in range(self._nb_dofs):
                action[2 * dof] = self._scale_pressure(dof, True, action_casted[2 * dof])
                action[2 * dof + 1] = (
                    self._scale_pressure(dof, False, action_casted[2 * dof + 1])
                )
        else:
            for dof in range(self._nb_dofs):
                if self.delta_p_p0_is_action:
                    p0 = action_casted[2*dof+1]
                    value = action_casted[2*dof] * 2 - 1
                else:
                    p0 = self.delta_p_p0_value[dof]
                    if self.delta_p:
                        value = action_casted[dof] * 2 - 1
                    else:
                        value = action_casted[dof]
                action[2 * dof] = self._scale_pressure_delta_p(dof, True, value, p0)
                action[2 * dof+1] = self._scale_pressure_delta_p(dof, False, value, p0)

        # final target pressure (make sure that it is within bounds)
        for dof in range(self._nb_dofs):
            action[2 * dof] = self._bound_pressure(dof, True, action[2 * dof])
            action[2 * dof + 1] = self._bound_pressure(dof, False, action[2 * dof + 1])

        # hysr takes a list of int, not float, as input
        action = [int(a) for a in action]

        # performing a step
        for _ in range(self._action_repeat_counter):
            observation, reward, episode_over, *extrainfo = self._hysr.step(list(action))
            if episode_over:
                break

        # formatting observation in a format suitable for gym
        observation = self._convert_observation(observation, action_casted)

        # imposing frequency to learning agent
        if not self._accelerated_time:
            self._frequency_manager.wait()

        # Ignore steps after hitting the ball
        if not episode_over and not self._hysr._ball_status.min_distance_ball_racket:
            if self.delta_p and not self.delta_p_p0_is_action:
                return self.step(action_orig_delta_p)
            else:
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
            print("ep:", self.n_eps, " steps:", self.n_steps, " rew:", reward)
            if self._logger:
                self._logger.record("eprew", reward)
                self._logger.record("min_discante_ball_racket", self._hysr._ball_status.min_distance_ball_racket or 0)
                self._logger.record("min_distance_ball_target_capped",
                    min(
                        self._hysr._ball_status.min_distance_ball_target or self._hysr._reward_function.config.normalization_constant,
                        self._hysr._reward_function.config.normalization_constant))
                self._logger.record("max_ball_velocity", self._hysr._ball_status.max_ball_velocity)
                # self._logger.dump()

        return observation, reward, episode_over, False, {}

    def reset(self, *, seed=None, options=None):
        np.random.seed(seed)

        self.init_episode()
        observation, *extrainfo = self._hysr.reset()
        observation = self._convert_observation(observation, self.last_action)
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

    def close(self):
        self._hysr.close()


                

