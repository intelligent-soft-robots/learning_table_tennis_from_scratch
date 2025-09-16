import json
import math
import time
import os
from collections import OrderedDict
from typing import Dict, Union

import gymnasium as gym
import numpy as np
import o80
import pam_interface

from .hysr_one_ball import HysrOneBall, HysrOneBallConfig
from .rewards import JsonReward

from scipy.interpolate import make_interp_spline

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
    
    def get_start_index(self, name):
        start_idx = 0
        for key, box in self._obs_boxes.items():
            if key == name:
                return start_idx
            start_idx += box.size
        raise KeyError(f"Box '{name}' not found")


class HysrManyBallEnv(gym.Env):
    def __init__(
        self,
        reward_config_file=None,
        hysr_one_ball_config_file=None,
        log_episodes=False,
        logger=None,
        stop_new_actions_after_main_ball_hit=True,
    ):
        super().__init__()

        
        self._logger = logger
        self._stop_new_actions_after_main_ball_hit = stop_new_actions_after_main_ball_hit

        hysr_one_ball_config = HysrOneBallConfig.from_json(hysr_one_ball_config_file)

        self._save_folder_traj = hysr_one_ball_config.save_folder_traj
        if self._save_folder_traj and not os.path.exists(self._save_folder_traj):
            os.makedirs(self._save_folder_traj)

        self._log_episodes = log_episodes and self._save_folder_traj!=""

        reward_function = JsonReward.get(reward_config_file)

        # check if reward function has config object
        if hasattr(reward_function, "config"):
            self.normalization_constant = reward_function.config.normalization_constant
        else:
            self.normalization_constant = reward_function.normalization_constant

        self._config = pam_interface.JsonConfiguration(
            str(hysr_one_ball_config.pam_config_file)
        )
        self._nb_dofs = len(self._config.max_pressures_ago)
        self._algo_time_step = hysr_one_ball_config.algo_time_step
        self._pressure_change_range = hysr_one_ball_config.pressure_change_range
        self._accelerated_time = hysr_one_ball_config.accelerated_time
        self._goal_in_state = (hysr_one_ball_config.target_position_sampling_radius != 0)
        self._action_repeat_counter = hysr_one_ball_config.action_repeat_counter

        self._hysr = HysrOneBall(hysr_one_ball_config, reward_function)
        self._unsuccessful_episode_counter = 0  # Counter for unsuccessful episodes
        self._hit_ball_indices = set()  # Track all hit ball indices that reached other side
        self._hit_indices_file = os.path.join(self._save_folder_traj, "hit_ball_indices.json") if self._save_folder_traj else None
        self._load_hit_indices()

        self._obs_boxes = _ObservationSpace()
        
        self.action_space = gym.spaces.Box(
            low=-1.0, high=+1.0, shape=(self._nb_dofs * 2,), dtype=np.float32
        )

        self._obs_boxes.add_box("robot_position", -math.pi, +math.pi, self._nb_dofs)
        self._obs_boxes.add_box("robot_velocity", -10.0, 10.0, self._nb_dofs)
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

        if self._goal_in_state:
            self._obs_boxes.add_box("goal", -10.0, +10.0, 3)

        

        self.observation_space = self._obs_boxes.get_gym_box()

        if not self._accelerated_time:
            self._frequency_manager = o80.FrequencyManager(
                1.0 / hysr_one_ball_config.algo_time_step
            )

        self.n_eps = 0
        self.n_steps_on_policy = 0
        self.init_episode()

    def _load_hit_indices(self):
        """Load previously hit ball indices from file."""
        if self._hit_indices_file and os.path.exists(self._hit_indices_file):
            try:
                with open(self._hit_indices_file, 'r') as f:
                    data = json.load(f)
                    self._hit_ball_indices = set(data.get('hit_indices', []))
                    print(f"Loaded {len(self._hit_ball_indices)} previously hit ball indices")
            except Exception as e:
                print(f"Error loading hit indices: {e}")
                self._hit_ball_indices = set()
        else:
            self._hit_ball_indices = set()

    def _save_hit_indices(self):
        """Save hit ball indices to file."""
        if self._hit_indices_file:
            try:
                with open(self._hit_indices_file, 'w') as f:
                    json.dump({
                        'hit_indices': sorted(list(self._hit_ball_indices)),
                        'total_count': len(self._hit_ball_indices),
                        'last_updated': time.strftime("%Y-%m-%d_%H-%M-%S")
                    }, f, indent=2)
            except Exception as e:
                print(f"Error saving hit indices: {e}")

    def init_episode(self):
        self.n_steps = 0

        self.data_buffer = []
        self.ball_hit = False
        self.extra_data_buffer = [[] for _ in range(self._hysr._hysr_config.extra_balls_per_set)]
        self.extra_ball_hit = [False for _ in range(self._hysr._hysr_config.extra_balls_per_set)]

        if self.n_eps == 0:
            print("---HysrManyBallEnv with {} extra balls---".format(self._hysr._hysr_config.extra_balls_per_set))

        # initialize initial action (for action diffs)
        self.last_action = self.get_init_action()
        self._ball_hit = False

    def get_init_action(self):
        init_action = np.zeros(self._nb_dofs * 2, dtype=np.float32)
        starting_pressures = self._hysr.get_starting_pressures()
        for dof in range(self._nb_dofs):
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
        if self._goal_in_state:
            self._obs_boxes.set_values_non_norm("goal", self._hysr._ball_status.target_position)
        self.last_observation = self._obs_boxes.get_normalized_values()
        return self.last_observation.copy()





    def set_ball_id(self, ball_id, extra_balls=False):
        # print("set ball id env", ball_id)
        self._hysr.set_ball_id(ball_id, extra_balls)

    def set_ball_random_trajectory_translation(self, translation, extra_balls=False):
        self._hysr.set_ball_random_trajectory_translation(translation, extra_balls)

    def set_goal(self, goal):
        self._hysr.set_goal(goal)

    def _get_obs(self, state) -> Dict[str, Union[int, np.ndarray]]:
            """
            Helper to create the observation.

            :return: The current observation.
            """
            # return OrderedDict(
            #     [
            #         ("observation", self._convert_observation(state)),
            #     ]
            # )
            return self._convert_observation(state)

    def _get_extra_obs(self, extra_states) -> Dict[str, Union[int, np.ndarray]]:
            """
            Helper to create the observation.

            :return: The current observation.
            """
            # return [OrderedDict(
            #     [
            #         ("observation", self._convert_observation(extra_state) ),
            #     ]
            # )
            #     for extra_state in extra_states
            # ]
            return [self._convert_observation(extra_state) for extra_state in extra_states]


    # remove transitions between the ball hitting the racket and the ball hitting the table as well as transitions after the end of the episode
    def get_reduced_episodes(self):
        self.extra_data_buffer.append(self.data_buffer)     # put main ball in extra ball buffer (to avoid code repetition)
        all_trans = [[] for _ in range(self._hysr._hysr_config.extra_balls_per_set + 1)]
        idx = 0
        for data_buffer in self.extra_data_buffer:
            obs_before_racket_hit = None
            action_before_racket_hit = None
            for obs, action, _, _, reward, episode_over, _, previous_obs, min_distance_ball_racket in data_buffer:
                if not episode_over and min_distance_ball_racket:   # normal transition
                    all_trans[idx].append((previous_obs, obs, action, reward, episode_over, [{}]))
                elif not episode_over and not self._hysr._ball_status.min_distance_ball_racket:   # ball hit racket, but didn't cross table plane (episode_over is FALSE) -> do not add
                    if obs_before_racket_hit is None:
                        obs_before_racket_hit = previous_obs
                        action_before_racket_hit = action
                elif episode_over and obs_before_racket_hit is not None:  # episode over and ball hit
                    all_trans[idx].append((obs_before_racket_hit, obs, action_before_racket_hit, reward, episode_over, [{}]))
                    break
                elif episode_over and obs_before_racket_hit is None:  # episode over and ball not hit
                    all_trans[idx].append((previous_obs, obs, action, reward, episode_over, [{}]))
                    break
            idx += 1

        assert len(self.extra_data_buffer) == len(all_trans)

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

        # increase actions in 1. dof further
        action[0] *= 4
        action[1] *= 4

        action_sigmoid = [1 / (1 + np.exp(-a)) - 0.5 for a in action]
        action = [
            np.clip(a1 + a2, 0, 1) for a1, a2 in zip(self.last_action, action_sigmoid)
        ]
        self.last_action = action.copy()
        action_casted = action.copy()

        # put pressure in range as defined in parameters file
        for dof in range(self._nb_dofs):
            action[2 * dof] = self._scale_pressure(dof, True, action_casted[2 * dof])
            action[2 * dof + 1] = (
                self._scale_pressure(dof, False, action_casted[2 * dof + 1])
            )

        # final target pressure (make sure that it is within bounds)
        for dof in range(self._nb_dofs):
            action[2 * dof] = self._bound_pressure(dof, True, action[2 * dof])
            action[2 * dof + 1] = self._bound_pressure(dof, False, action[2 * dof + 1])

        # hysr takes a list of int, not float, as input
        action = [int(a) for a in action]

        infos = {}
        idx_ball_still_active = -1

        obs = None
        extra_obs = None

        # performing a step
        for _ in range(self._action_repeat_counter):
            observation, reward, episode_over, extra_observations, extra_rewards, extra_dones = self._hysr.step(list(action))
            all_episodes_over = episode_over and all(extra_dones)

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
                
                if idx_ball_still_active == -1: # non of the balls can still be hit anymore 
                    if not episode_over and not self._hysr._ball_status.min_distance_ball_racket:
                        return self.step(action_orig)

            obs = self._get_obs(observation)
            extra_obs = self._get_extra_obs(extra_observations)

            # logging
            self.n_steps += 1
            if self._log_episodes:
                # Prepare data to log
                fk = self._hysr._mirrorings[0].get_fk()
                rob_pos = fk[0]
                rob_vel = fk[1]
                racket_pos = fk[2]
                racket_vel = fk[3]
                racket_ori = fk[4]
                timestamp = fk[5]
                data_entry = (
                    self.previous_obs.copy(),
                    action_orig,
                    action_casted,
                    action.copy(),
                    reward,
                    episode_over,
                    (rob_pos, rob_vel, racket_pos, racket_vel, racket_ori, timestamp),
                    obs.copy(),
                    self._hysr._ball_status.min_distance_ball_racket
                )

                skip_entry = False
                if not episode_over and not self._hysr._ball_status.min_distance_ball_racket:
                    if not self.ball_hit:
                        self.ball_hit = True
                    else:
                        skip_entry = True

                if len(self.data_buffer) > 0 and self.data_buffer[-1][5]:  # episode over
                    skip_entry = True

                if not skip_entry:
                    if not episode_over or not self.ball_hit:
                        self.data_buffer.append(data_entry)
                    else:
                        # replace last entry with new one
                        self.data_buffer[-1] = (
                            self.data_buffer[-1][0],
                            self.data_buffer[-1][1],
                            self.data_buffer[-1][2],
                            self.data_buffer[-1][3],
                            reward,
                            episode_over,
                            self.data_buffer[-1][6],
                            obs.copy(),
                            self.data_buffer[-1][8]
                        )

                # add extra transitions
                idx = -1
                for extra_ob, extra_reward, extra_episode_over, extra_previous_obs, extra_min_distance_ball_racket in \
                    zip(extra_obs, extra_rewards, extra_dones, self.previous_extra_obs, self._hysr.extra_min_distance_ball_racket):
                    idx += 1
                    if not extra_episode_over and not extra_min_distance_ball_racket:
                        if not self.extra_ball_hit[idx]:
                            self.extra_ball_hit[idx] = True
                        else:
                            continue
                    if len(self.extra_data_buffer[idx]) > 0 and self.extra_data_buffer[idx][-1][5]:  # episode over
                        continue

                    if not extra_episode_over or not self.extra_ball_hit[idx]:
                        self.extra_data_buffer[idx].append(
                            (
                                extra_previous_obs.copy(),
                                action_orig,
                                None,
                                None,
                                extra_reward,
                                extra_episode_over,
                                (rob_pos, rob_vel, racket_pos, racket_vel, racket_ori, timestamp),
                                extra_ob.copy(),
                                extra_min_distance_ball_racket
                            )
                        )
                    else:
                        # replace last entry with new one
                        self.extra_data_buffer[idx][-1] = (
                            self.extra_data_buffer[idx][-1][0],
                            self.extra_data_buffer[idx][-1][1],
                            None,
                            None,
                            extra_reward,
                            extra_episode_over,
                            self.extra_data_buffer[idx][-1][6],
                            extra_ob.copy(),
                            extra_min_distance_ball_racket
                        )

                infos = {}

            self.previous_extra_obs = extra_obs.copy()
            self.previous_obs = obs = obs.copy()           

            all_episodes_over = episode_over and all(extra_dones)

            # temporary fix for ppo: also use old action if main ball was hit
            if not self._hysr._ball_status.min_distance_ball_racket and not episode_over and self._stop_new_actions_after_main_ball_hit:
                return self.step(action_orig)

            if all_episodes_over:
                break


        if all_episodes_over:
            print("ep:", self.n_eps, " rew:", reward)
            infos["trajectory"], infos["hsm_trajectories"] = self.get_reduced_episodes()
            if self._log_episodes:
                self.dump_data(self.data_buffer)
                for idx, extra_data_buffer in enumerate(self.extra_data_buffer):
                    # ignore last buffer (main ball)
                    if idx == len(self.extra_data_buffer) - 1:
                        continue
                    self.dump_data(extra_data_buffer, idx)
            if self._logger:
                self._logger.record("eprew", reward)
                self._logger.record("n_steps_on_policy", self.n_steps_on_policy)
                self._logger.record("min_discante_ball_racket", self._hysr._ball_status.min_distance_ball_racket or 0)
                self._logger.record("min_distance_ball_target_capped",
                    min(
                        self._hysr._ball_status.min_distance_ball_target or self.normalization_constant,
                        self.normalization_constant))
                self._logger.record("max_ball_velocity", self._hysr._ball_status.max_ball_velocity)
                # self._logger.dump()
            self.n_eps += 1

        # use different ball for observation if main ball not active anymore
        if idx_ball_still_active!=-1:
            obs = extra_obs[idx_ball_still_active]
            # reward = 0 #extra_rewards[idx_ball_still_active]
        else:
            self.n_steps_on_policy += 1

        if not all_episodes_over:
            reward = 0

        return obs, reward, all_episodes_over, False, infos

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.init_episode()
        observation, extra_observations = self._hysr.reset()
        obs = self._get_obs(observation)
        extra_obs = self._get_extra_obs(extra_observations)
        if not self._accelerated_time:
            self._frequency_manager = None
        self.previous_extra_obs = extra_obs.copy()
        self.previous_obs = obs.copy()
        return obs, {}

    def dump_data(self, data_buffer, index=None):
        if len(data_buffer) > 0:
            final_observation = data_buffer[-1][7]  # next_ob is at index 7
            ball_pos_start = self._obs_boxes.get_start_index("ball_position")
            final_ball_y = final_observation[ball_pos_start + 1]
            table_center_y = self._hysr._hysr_config.table_position[1]
            ball_reached_other_side = final_ball_y > table_center_y
            
            # Track successful hits - more specific check for landing on table
            if ball_reached_other_side and self.ball_hit:
                final_ball_x = final_observation[ball_pos_start]
                table_center_x = self._hysr._hysr_config.table_position[0]
                
                # Standard table dimensions
                table_half_width = 0.7625  # Half of 1.525m width
                table_half_length = 1.37   # Half of 2.74m length
                
                # Check if ball is within table bounds
                ball_within_x_bounds = (table_center_x - table_half_width) <= final_ball_x <= (table_center_x + table_half_width)
                ball_within_y_bounds = table_center_y < final_ball_y <= (table_center_y + table_half_length)
                
                if ball_within_x_bounds and ball_within_y_bounds:
                    current_traj_index = self._hysr._ball_behavior._random_traj_index
                    if current_traj_index not in self._hit_ball_indices:
                        self._hit_ball_indices.add(current_traj_index)
                        self._save_hit_indices()
                        print(f"New successful hit! Ball index {current_traj_index} landed on table at ({final_ball_x:.3f}, {final_ball_y:.3f}). Total unique hits: {len(self._hit_ball_indices)}")
            
            else:
                print("x", end="", flush=True)

            if not ball_reached_other_side:
                self._unsuccessful_episode_counter += 1
                if self._unsuccessful_episode_counter % 100 != 0:
                    return
        
        filename = self._save_folder_traj + "ppo" + time.strftime("%Y%m%d-%H%M%S")
        if index is not None:
            filename += "_" + str(index)
        filename += "_" + str(np.random.randint(10000)) + ".json"
        dict_data_full = dict()
        with open(filename, "w") as json_data:
            dict_data_full["ob"] = [x[0].tolist() for x in data_buffer]
            dict_data_full["next_ob"] = [x[7].tolist() for x in data_buffer]
            dict_data_full["action_orig"] = [x[1].tolist() for x in data_buffer]
            dict_data_full["action_casted"] = [x[2] for x in data_buffer]
            dict_data_full["prdes"] = [x[3] for x in data_buffer]
            dict_data_full["reward"] = [x[4] for x in data_buffer]
            dict_data_full["episode_over"] = [x[5] for x in data_buffer]
            dict_data_full["fk"] = [x[6] for x in data_buffer]
            dict_data_full["random_traj_index"] = self._hysr._ball_behavior._random_traj_index
            json.dump(dict_data_full, json_data)

    def get_hit_indices_stats(self):
        """Get statistics about hit ball indices."""
        return {
            'total_unique_hits': len(self._hit_ball_indices),
            'hit_indices': sorted(list(self._hit_ball_indices)),
            'file_path': self._hit_indices_file
        }

    def close(self):
        self._hysr.close()
