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

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.optimize import minimize
from scipy.stats import norm
from noise import pnoise1

from scipy.interpolate import make_interp_spline


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
        continue_after_hit=True,
        continue_after_hit_with_smooth_approximation=True,
    ):
        super().__init__()

        self._log_episodes = log_episodes
        self._logger = logger
        self._continue_after_hit = continue_after_hit
        self._continue_after_hit_with_smooth_approximation = continue_after_hit_with_smooth_approximation

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

        self._hysr = HysrOneBall(hysr_one_ball_config, reward_function)


        self._obs_boxes = _ObservationSpace()

        
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

        self.opt_data = []
        self.last_sample_plus_noise = 0

        self.n_eps = 0
        self.init_episode()
        self.noise_facor = 0.02

        self.motion_gen = self.MotionGenerator()

        self.running_mean_rew = 0


    def init_episode(self):
        self.n_steps = 0
        if self._log_episodes:
            self.data_buffer = []
            self.data_buffer_short = []
        # initialize initial action (for action diffs)
        self.last_action = self.get_init_action()
        self._ball_hit = False  # To track if the ball has been hit
        self.first_step_after_hit = True

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





    def set_ball_id(self, ball_id):
        self._hysr.set_ball_id(ball_id)

    def set_goal(self, goal):
        self._hysr.set_goal(goal)


    def get_action_motion1(self):
        # motion 1: fixed pressure for dof 2,3,4 and changing pressure for dof 1

        if self.n_steps == 0:
            self.n_steps_motion_change = np.random.randint(15, 50)
            # self.n_steps_motion_change = int(self.n_steps_motion_change)

            self.action_dof_1 = np.random.uniform(0.05, 0.075)

            self.action_dof3 = np.random.uniform(-0.2, 0.2)

            self.action_dof4 = np.random.uniform(-0.2, 0.2)
            # self.action_dof_1 = float(self.action_dof_1)

        if self.n_steps < self.n_steps_motion_change:
            action = np.array([0.2, -0.2, -0.05, 0.05, self.action_dof3, -self.action_dof3,  self.action_dof4,  -self.action_dof4])
        else:
            action = np.array([-self.action_dof_1 , self.action_dof_1 , 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        print(".", end="")

        # time.sleep(0.02)

        return action


    def get_action_motion2(self):

        if self.n_steps == 0:
            self.n_steps_motion_change = np.random.randint(40, 80)

            self.n_steps_hit_motion_steps = np.random.randint(10, 30)

            self.action_dof_1 = np.random.uniform(0.04, 0.10)

            self.action_dof3 = np.random.uniform(-0.3, 0.3)

            self.action_dof4 = np.random.uniform(-0.2, 0.2)

            print("n_steps_motion_change: ", self.n_steps_motion_change, "n_steps_hit_motion_steps: ", self.n_steps_hit_motion_steps, "action_dof_1: ", self.action_dof_1, "action_dof3: ", self.action_dof3, "action_dof4: ", self.action_dof4)

        change_rate = 1/self.n_steps_motion_change
        w1 = change_rate * self.n_steps

        w2 = 1 / self.n_steps_motion_change * np.clip(self.n_steps - self.n_steps_motion_change, 0, self.n_steps_hit_motion_steps)
        
        action = \
            np.clip(1-w1, 0, 1) * 2 * np.array([0.2, -0.2, -0.07, 0.07, self.action_dof3, -self.action_dof3,  self.action_dof4,  -self.action_dof4]) + \
            np.clip(w1-w2, 0, 1) * 2 * np.array([-self.action_dof_1 , self.action_dof_1 , 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) + \
            np.clip(w2-w1, 0, 1) * 0.5 * np.array([self.action_dof_1 , -self.action_dof_1 , 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        action = action + self.motion_gen.get_noise()

        # print(".", end="")

        # time.sleep(0.02)

        return action
    
    def get_action_test(self):
        if self.n_steps == 0:
            self.n_steps_motion_change = 66

            self.n_steps_hit_motion_steps = 10

            self.action_dof_1 = 0.06

            self.action_dof3 = -0.1

            self.action_dof4 = -0.05

        change_rate = 1/self.n_steps_motion_change
        w1 = change_rate * self.n_steps

        w2 = 1 / self.n_steps_motion_change * np.clip(self.n_steps - self.n_steps_motion_change, 0, self.n_steps_hit_motion_steps)
        
        action = \
            np.clip(1-w1, 0, 1) * 2 * np.array([0.2, -0.2, -0.07, 0.07, self.action_dof3, -self.action_dof3,  self.action_dof4,  -self.action_dof4]) + \
            np.clip(w1-w2, 0, 1) * 2 * np.array([-self.action_dof_1 , self.action_dof_1 , 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) + \
            np.clip(w2-w1, 0, 1) * 0.5 * np.array([self.action_dof_1 , -self.action_dof_1 , 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        return action

    # def acquisition_function(params, gp=gp, xi=0.01, kappa=0.01):
    #     params = np.array(params).reshape(1, -1)
    #     mean, std = gp.predict(params, return_std=True)
        
    #     current_best = np.max(y)

    #     # Calculate Expected Improvement
    #     ei = mean - current_best - xi
    #     ei[ei < 0] = 0
    #     ei = ei * norm.cdf((mean - current_best - xi) / (std + 1e-9))

    #     # Calculate Entropy (using standard deviation as a proxy for uncertainty)
    #     entropy = std

    #     # Combine EI and entropy
    #     combined_acquisition = ei + kappa * entropy # Expected Improvement + Entropy

    #     return combined_acquisition


    def perform_bayesian_optimization(self):
        # Reshape and prepare data for Gaussian Process
        
        # first two entries are X, third entry is y
        X = np.array([np.concatenate([Amp, phi]) for Amp, phi, _ in self.opt_data])
        y = np.array([rew for _, _, rew in self.opt_data])

        # Define and train Gaussian Process
        kernel = Matern(nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel)
        gp.fit(X, y)

        def acquisition_function(params):
            params = np.array(params).reshape(1, -1)
            mean, std = gp.predict(params, return_std=True)
            return mean + 0.6 * std  # Expected Improvement (std weighted less)

        # Optimize acquisition function
        result = minimize(lambda params: -acquisition_function(params), 
                          x0=np.random.rand(16),  # Initial guess, 16 = 8 Amp + 8 phi
                          bounds=[(-1, 1)] * 16) 

        # Extract optimized Amp and phi
        optimized_params = result.x
        Amp_optimized = optimized_params[:8]
        phi_optimized = optimized_params[8:]

        # print("noise: ", self.noise_facor)
        # self.noise = np.random.uniform(-self.noise_facor, self.noise_facor, 8)

        # Amp_optimized += self.noise
        # phi_optimized += self.noise

        return Amp_optimized, phi_optimized


    def get_action_motion3(self):

        if self.n_steps == 0:
            if len(self.opt_data) < 50:
                self.Amp, self.phi = np.random.uniform(-1, 1, 8), np.random.uniform(-1, 1, 8)
                self.opt_data.append([self.Amp, self.phi])
            else:
                if np.random.random() < 0.5:
                    self.Amp, self.phi = self.perform_bayesian_optimization()
                    
                    self.add_noise = np.zeros(8)
                    self.opt_data.append([self.Amp, self.phi])
                    
                else:
                    # sample from 5% best data
                    best_data = sorted(self.opt_data, key=lambda x: x[2], reverse=True)
                    random_index = np.random.randint(0, int(len(best_data)*0.05))
                    self.Amp, self.phi = best_data[random_index][0], best_data[random_index][1]
                    
                    print("best data: ", best_data[random_index][2], "Amp: ", self.Amp, "phi: ", self.phi)
                    # self.add_noise = np.random.uniform(-self.noise_facor, self.noise_facor, 8)

                    # if self.last_sample_plus_noise > 0.5:
                    #     self.noise_facor*=1.2
                    # else:
                    #     self.noise_facor/=1.1
                    #     self.noise_facor = max(self.noise_facor, 0.001)

                    # print("noise: ", self.noise_facor)
                    

        action = np.sin(np.array([self.n_steps * 0.02 + self.phi[i] for i in range(8)])) * self.Amp

        action = action + self.motion_gen.get_noise()

        # if len(self.opt_data) > 100:
        #     action += self.add_noise
        
        # action += self.noise

        return action



        
    class MotionGenerator:
        def __init__(self, scale=0.5, octaves=1, persistence=0.5, lacunarity=2.0):
            self.scale = scale
            self.octaves = octaves
            self.persistence = persistence
            self.lacunarity = lacunarity
            self.steps = np.random.rand(8) * 10000  # Independent step for each DOF

        def get_noise(self):
            action = np.zeros(8)
            for i in range(8):
                self.steps[i] += 1  # Increment step for each DOF independently

                # Generate independent Perlin noise for amplitude and phase
                amp = pnoise1(self.steps[i] * self.scale, octaves=self.octaves, persistence=self.persistence, lacunarity=self.lacunarity)
                phi = pnoise1((self.steps[i] + 5000) * self.scale, octaves=self.octaves, persistence=self.persistence, lacunarity=self.lacunarity)

                # Scale amp and phi
                amp = np.interp(amp, [-1, 1], [-1, 1])
                phi = np.interp(phi, [-1, 1], [0, 2 * np.pi])  # Scale phi to [0, 2π] for full sine wave cycle

                # Generate action
                action[i] = np.sin(self.steps[i] * 0.02 + phi) * amp
            # print("a", action)
            return action


    def get_continued_action(self, hit_window=10, future_steps=20, method='exp_decay'):
        if self.n_steps <= hit_window:
            return None
            
        # Extract previous actions
        previous_actions = np.array([x[1] for x in self.data_buffer[-hit_window:]])
        
        # Calculate action differences
        action_diffs = np.diff(previous_actions, axis=0)
        last_action = previous_actions[-1]
        
        if method == 'exp_decay':
            # Exponentially decay the action differences
            last_diffs = action_diffs[-3:].mean(axis=0)  # Average of last 3 differences
            decay_rate = 0.85  # Adjust this to control decay speed
            
            # Predict next action with decaying differences
            decay_factor = decay_rate ** (self.n_steps - len(self.data_buffer))
            new_action_diff = last_diffs * decay_factor
            
            # Bound the differences to prevent explosions
            max_diff = np.abs(action_diffs).max(axis=0)
            new_action_diff = np.clip(new_action_diff, -max_diff, max_diff)
            
            new_action = last_action + new_action_diff

        elif method == 'spline':
            # Fit a smooth spline to previous actions
            t_prev = np.arange(hit_window)
            t_future = np.arange(hit_window + future_steps)
            
            # Fit separate splines for each dimension
            new_action = np.zeros_like(last_action)
            for dim in range(last_action.shape[0]):
                spline = make_interp_spline(t_prev, previous_actions[:, dim], k=3)
                
                # Get the continuation and apply dampening
                future_values = spline(t_future)
                dampening = np.exp(-0.1 * (t_future[-1] - t_future[hit_window]))
                future_values[hit_window:] *= dampening
                
                new_action[dim] = future_values[hit_window]

        elif method == 'damped':
            # Use a damped oscillator model
            velocity = action_diffs[-1]  # Current velocity (last action difference)
            damping = 0.9  # Damping coefficient
            spring = 0.1   # Spring coefficient
            
            # Update velocity and position using damped oscillator equations
            new_velocity = velocity * damping - spring * (last_action - previous_actions[-2])
            new_action = last_action + new_velocity
            
        # Bound the final actions to the historical range
        action_min = np.min(previous_actions, axis=0)
        action_max = np.max(previous_actions, axis=0)
        margin = 0.2 * (action_max - action_min)  # Allow 20% outside historical range
        new_action = np.clip(new_action, 
                            action_min - margin,
                            action_max + margin)
        
        return new_action



    def step(self, action):

        # action = self.get_action_motion2()
        # action = self.get_action_test()

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

        # performing a step
        for _ in range(self._action_repeat_counter):
            observation, reward, episode_over = self._hysr.step(list(action))
            if episode_over:
                break

        # formatting observation in a format suitable for gym
        observation = self._convert_observation(observation, action_casted)

        # imposing frequency to learning agent
        if not self._accelerated_time:
            self._frequency_manager.wait()

        # Update ball hit status
        if not self._ball_hit and self._hysr._ball_status.min_distance_ball_racket:
            self._ball_hit = True

        # Skip steps after hitting the ball if continue_after_hit is False
        if not self._continue_after_hit and not episode_over and not self._hysr._ball_status.min_distance_ball_racket:
            return self.step(action_orig)

        # logging
        self.n_steps += 1
        if self._log_episodes:
            # Prepare data to log
            data_entry = (
                self.previous_observation.copy(),
                action_orig,
                action_casted,
                action.copy(),
                reward,
                episode_over,
                # (rob_pos, rob_vel, racket_pos, racket_vel, racket_ori, timestamp),
                observation.copy(),
            )
            # Append to full trajectory
            self.data_buffer.append(data_entry)
            # Append to short trajectory if before hit, or first step after hit
            if self._hysr._ball_status.min_distance_ball_racket or self.first_step_after_hit:
                if not self._hysr._ball_status.min_distance_ball_racket:
                    self.first_step_after_hit = False
                self.data_buffer_short.append(data_entry)

            # in final transition, keep the last observation and action, but replace reward, next observation and episode_over
            if episode_over:
                self.data_buffer_short[-1] = (
                    self.data_buffer_short[-1][0],
                    self.data_buffer_short[-1][1],
                    self.data_buffer_short[-1][2],
                    self.data_buffer_short[-1][3],
                    reward,
                    episode_over,
                    observation.copy(),
                )

        if self._continue_after_hit_with_smooth_approximation and not episode_over and not self._hysr._ball_status.min_distance_ball_racket:
            new_action_orig = self.get_continued_action(
                hit_window=10,
                method='exp_decay'
            )
            if new_action_orig is not None:
                return self.step(new_action_orig)


        if episode_over:
            self.running_mean_rew = 0.95 * self.running_mean_rew + 0.05 * reward
            print("running mean reward: ", self.running_mean_rew)
            # if len(self.opt_data[-1]) < 3:
            #    self.opt_data[-1].append(reward)
            #else:
            #    self.last_sample_plus_noise = reward
            # print("phi", self.phi, "Amp", self.Amp, "rew", reward)
            if self._log_episodes: # and self.running_mean_rew > 0.65:
                self.dump_data(self.data_buffer, self.data_buffer_short)
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
                self._logger.dump()
            if reward > 0:
                print("SUCCESS")

        self.previous_observation = observation.copy()

        return observation, reward, episode_over, {}

    def reset(self):
        self.init_episode()
        observation = self._hysr.reset()
        observation = self._convert_observation(observation, self.last_action)
        if not self._accelerated_time:
            self._frequency_manager = None

        self.previous_observation = observation.copy()
        return observation

    def dump_data(self, data_buffer, data_buffer_short=None):
        # Dump full trajectory
        filename_full = "/tmp/ep_full_test_" + time.strftime("%Y%m%d-%H%M%S")
        dict_data_full = dict()
        with open(filename_full, "w") as json_data_full:
            dict_data_full["ob"] = [x[0].tolist() for x in data_buffer]
            dict_data_full["next_ob"] = [x[-1].tolist() for x in data_buffer]
            dict_data_full["action_orig"] = [x[1].tolist() for x in data_buffer]
            dict_data_full["action_casted"] = [x[2] for x in data_buffer]
            dict_data_full["prdes"] = [x[3] for x in data_buffer]
            dict_data_full["reward"] = [x[4] for x in data_buffer]
            dict_data_full["episode_over"] = [x[5] for x in data_buffer]
            # dict_data_full["fk"] = [x[6] for x in data_buffer]
            dict_data_full["random_traj_index"] = self._hysr._ball_behavior._random_traj_index
            json.dump(dict_data_full, json_data_full)

        # Dump shorter trajectory if available
        if data_buffer_short is not None:
            filename_short = "/tmp/ep_short_test_" + time.strftime("%Y%m%d-%H%M%S")
            dict_data_short = dict()
            with open(filename_short, "w") as json_data_short:
                dict_data_short["ob"] = [x[0].tolist() for x in data_buffer_short]
                dict_data_short["next_ob"] = [x[-1].tolist() for x in data_buffer_short]
                dict_data_short["action_orig"] = [x[1].tolist() for x in data_buffer_short]
                dict_data_short["action_casted"] = [x[2] for x in data_buffer_short]
                dict_data_short["prdes"] = [x[3] for x in data_buffer_short]
                dict_data_short["reward"] = [x[4] for x in data_buffer_short]
                dict_data_short["episode_over"] = [x[5] for x in data_buffer_short]
                # dict_data_short["fk"] = [x[6] for x in data_buffer_short]
                dict_data_short["random_traj_index"] = self._hysr._ball_behavior._random_traj_index
                json.dump(dict_data_short, json_data_short)

    def close(self):
        self._hysr.close()
