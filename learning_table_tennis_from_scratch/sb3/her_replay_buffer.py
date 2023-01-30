from http.client import NOT_IMPLEMENTED
import warnings
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch as th
import random

from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.preprocessing import get_obs_shape
from stable_baselines3.common.type_aliases import DictReplayBufferSamples
from stable_baselines3.common.vec_env import VecEnv, VecNormalize
from stable_baselines3.her.goal_selection_strategy import KEY_TO_GOAL_STRATEGY, GoalSelectionStrategy
from stable_baselines3.her.hindsight_state_selection_strategy import KEY_TO_HINDSIGHT_STATE_STRATEGY, KEY_TO_HINDSIGHT_STATE_STRATEGY_HORIZON, HindsightStateSelectionStrategy, HindsightStateSelectionStrategyHorizon



def get_time_limit(env: VecEnv, current_max_episode_length: Optional[int]) -> int:
    """
    Get time limit from environment.

    :param env: Environment from which we want to get the time limit.
    :param current_max_episode_length: Current value for max_episode_length.
    :return: max episode length
    """
    # try to get the attribute from environment
    if current_max_episode_length is None:
        try:
            current_max_episode_length = env.get_attr("spec")[0].max_episode_steps
            # Raise the error because the attribute is present but is None
            if current_max_episode_length is None:
                raise AttributeError
        # if not available check if a valid value was passed as an argument
        except AttributeError:
            raise ValueError(
                "The max episode length could not be inferred.\n"
                "You must specify a `max_episode_steps` when registering the environment,\n"
                "use a `gym.wrappers.TimeLimit` wrapper "
                "or pass `max_episode_length` to the model constructor"
            )
    return current_max_episode_length


class HerReplayBuffer(DictReplayBuffer):
    """
    Hindsight Experience Replay (HER) buffer.
    Paper: https://arxiv.org/abs/1707.01495

    .. warning::

      For performance reasons, the maximum number of steps per episodes must be specified.
      In most cases, it will be inferred if you specify ``max_episode_steps`` when registering the environment
      or if you use a ``gym.wrappers.TimeLimit`` (and ``env.spec`` is not None).
      Otherwise, you can directly pass ``max_episode_length`` to the replay buffer constructor.


    Replay buffer for sampling HER (Hindsight Experience Replay) transitions.
    In the online sampling case, these new transitions will not be saved in the replay buffer
    and will only be created at sampling time.

    :param env: The training environment
    :param buffer_size: The size of the buffer measured in transitions.
    :param max_episode_length: The maximum length of an episode. If not specified,
        it will be automatically inferred if the environment uses a ``gym.wrappers.TimeLimit`` wrapper.
    :param goal_selection_strategy: Strategy for sampling goals for replay.
        One of ['episode', 'final', 'future']
     :param hindsight_state_selection_strategy: Strategy for sampling hindsight states for replay.
        One of ['random', 'reward', 'adavantage']
    :param device: PyTorch device
    :param n_sampled_goal: Number of virtual transitions to create per real transition,
        by sampling new goals.
    :param k_best
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    def __init__(
        self,
        env: VecEnv,
        buffer_size: int,
        device: Union[th.device, str] = "cpu",
        replay_buffer: Optional[DictReplayBuffer] = None,
        max_episode_length: Optional[int] = None,
        n_sampled_goal: int = 4,
        n_sampled_hindsight_states: int = 3,
        goal_selection_strategy: Union[GoalSelectionStrategy, str] = "future",
        hindsight_state_selection_strategy: Union[HindsightStateSelectionStrategy, str] = "random",
        hindsight_state_selection_strategy_horizon: Union[HindsightStateSelectionStrategy, str] = "future",
        online_sampling: bool = True,
        handle_timeout_termination: bool = True,
        apply_HER: bool = True,
        apply_HSM: bool = False,
        HSM_goal_env: bool = False,
        HSM_shape: int = -1,
        HSM_gamma = 0.999,
        HSM_n_traj_freq = 100,
        HSM_critic = None,
        HSM_critic_target = None,
        HSM_policy = None,
        HSM_min_criterion = -1000,
        n_sampled_hindsight_states_change_per_step = 0,
        HSM_criterion_change_per_step = 0,
        HSM_use_likelihood_ratio = False,
        HSM_likelihood_ratio_cutoff = 1/np.e,
        prioritized_replay_baseline = False,
        logger = None
    ):

        super(HerReplayBuffer, self).__init__(buffer_size, env.observation_space, env.action_space, device, env.num_envs)

        if apply_HER:
            # convert goal_selection_strategy into GoalSelectionStrategy if string
            if isinstance(goal_selection_strategy, str):
                self.goal_selection_strategy = KEY_TO_GOAL_STRATEGY[goal_selection_strategy.lower()]
            else:
                self.goal_selection_strategy = goal_selection_strategy

            # check if goal_selection_strategy is valid
            assert isinstance(
                self.goal_selection_strategy, GoalSelectionStrategy
            ), f"Invalid goal selection strategy, please use one of {list(GoalSelectionStrategy)}"
        if apply_HSM:
            # convert hindsight_state_selection_strategy into HindsightSelectionStrategy if string
            if isinstance(hindsight_state_selection_strategy, str):
                self.hindsight_state_selection_strategy = KEY_TO_HINDSIGHT_STATE_STRATEGY[hindsight_state_selection_strategy.lower()]
            else:
                self.hindsight_state_selection_strategy = hindsight_state_selection_strategy

            # check if hindsight_state_selection_strategy is valid
            assert isinstance(
                self.hindsight_state_selection_strategy, HindsightStateSelectionStrategy
            ), f"Invalid hindsight state selection strategy, please use one of {list(HindsightStateSelectionStrategy)}"

            # convert hindsight_state_selection_strategy_horizon into HindsightSelectionStrategyHorizon if string
            if isinstance(hindsight_state_selection_strategy_horizon, str):
                self.hindsight_state_selection_strategy_horizon = KEY_TO_HINDSIGHT_STATE_STRATEGY_HORIZON[hindsight_state_selection_strategy_horizon.lower()]
            else:
                self.hindsight_state_selection_strategy_horizon = hindsight_state_selection_strategy_horizon

            # check if hindsight_state_selection_strategy_horizon is valid
            assert isinstance(
                self.hindsight_state_selection_strategy_horizon, HindsightStateSelectionStrategyHorizon
            ), f"Invalid hindsight state selection strategy, please use one of {list(HindsightStateSelectionStrategyHorizon)}"


        self.n_sampled_goal = n_sampled_goal

        self.n_sampled_hindsight_states = n_sampled_hindsight_states

        # if we sample her transitions online use custom replay buffer
        self.online_sampling = online_sampling
        # compute ratio between HER replays and regular replays in percent for online HER sampling
        self.her_ratio = 1 - (1.0 / (self.n_sampled_goal + 1))
        # maximum steps in episode
        self.max_episode_length = get_time_limit(env, max_episode_length)
        # storage for transitions of current episode for offline sampling
        # for online sampling, it replaces the "classic" replay buffer completely
        her_buffer_size = buffer_size if online_sampling else self.max_episode_length

        self.env = env
        self.buffer_size = her_buffer_size

        if online_sampling:
            replay_buffer = None
            if apply_HSM and not apply_HER:
                raise ValueError(f"Online sampling for sampling hindsight states is not supported!")
        self.replay_buffer = replay_buffer
        self.online_sampling = online_sampling

        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination

        self.apply_HER = apply_HER

        self.apply_HSM = apply_HSM
        self.HSM_goal_env = HSM_goal_env
        self.HSM_shape = HSM_shape
        self.HSM_critic = HSM_critic
        self.HSM_critic_target = HSM_critic_target
        self.HSM_policy = HSM_policy
        self.HSM_gamma = HSM_gamma
        self.HSM_n_traj_freq = HSM_n_traj_freq
        self.HSM_min_criterion = HSM_min_criterion

        self.n_total_steps = 0
        self.n_sampled_hindsight_states_change_per_step = n_sampled_hindsight_states_change_per_step
        self.HSM_criterion_change_per_step = HSM_criterion_change_per_step
        self.HSM_use_likelihood_ratio = HSM_use_likelihood_ratio
        self.HSM_likelihood_ratio_cutoff = HSM_likelihood_ratio_cutoff

        self.prioritized_replay_baseline = prioritized_replay_baseline

        self.HSM_logging = False

        self.hsm_traj_buffer = []
        self.idx_eps_hsm = 0
        self.current_trajectory = []
        self.current_hsm_trajectories = [[]  for _ in range(HSM_shape)]
        self.extra_reset = True


        # buffer with episodes
        # number of episodes which can be stored until buffer size is reached
        self.max_episode_stored = self.buffer_size // self.max_episode_length
        self.current_idx = 0
        # Counter to prevent overflow
        self.episode_steps = 0

        # Get shape of observation and goal (usually the same)
        self.obs_shape = get_obs_shape(self.env.observation_space.spaces["observation"])
        if self.apply_HER or self.HSM_goal_env:
            self.goal_shape = get_obs_shape(self.env.observation_space.spaces["achieved_goal"])

        # input dimensions for buffer initialization
        if self.apply_HER or self.HSM_goal_env:
            input_shape = {
                "observation": (self.env.num_envs,) + self.obs_shape,
                "achieved_goal": (self.env.num_envs,) + self.goal_shape,
                "desired_goal": (self.env.num_envs,) + self.goal_shape,
                "action": (self.action_dim,),
                "reward": (1,),
                "next_obs": (self.env.num_envs,) + self.obs_shape,
                "next_achieved_goal": (self.env.num_envs,) + self.goal_shape,
                "next_desired_goal": (self.env.num_envs,) + self.goal_shape,
                "done": (1,),
            }
            self._observation_keys = ["observation", "achieved_goal", "desired_goal"]
        else:
            input_shape = {
                "observation": (self.env.num_envs,) + self.obs_shape,
                "action": (self.action_dim,),
                "reward": (1,),
                "next_obs": (self.env.num_envs,) + self.obs_shape,
                "done": (1,),
            }
            self._observation_keys = ["observation"]



        self._buffer = {
            key: np.zeros((self.max_episode_stored, self.max_episode_length, *dim), dtype=np.float32)
            for key, dim in input_shape.items()
        }
        # Store info dicts are it can be used to compute the reward (e.g. continuity cost)
        self.info_buffer = [deque(maxlen=self.max_episode_length) for _ in range(self.max_episode_stored)]
        # episode length storage, needed for episodes which has less steps than the maximum length
        self.episode_lengths = np.zeros(self.max_episode_stored, dtype=np.int64)

        # logging for gym robotic enfs
        self.gym_robotics_logger = logger
        self.n_hsm_traj = 0
        self.n_success_hsm_traj = 0
        self.n_dx_hsm_traj = 0
        self.n_success_dx_hsm_traj = 0
        self.n_traj = 0
        self.n_success_traj = 0
        self.n_dx_traj = 0
        self.n_success_dx_traj = 0


    def __getstate__(self) -> Dict[str, Any]:
        """
        Gets state for pickling.

        Excludes self.env, as in general Env's may not be pickleable.
        Note: when using offline sampling, this will also save the offline replay buffer.
        """
        state = self.__dict__.copy()
        # these attributes are not pickleable
        del state["env"]
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """
        Restores pickled state.

        User must call ``set_env()`` after unpickling before using.

        :param state:
        """
        self.__dict__.update(state)
        assert "env" not in state
        self.env = None

    def set_env(self, env: VecEnv) -> None:
        """
        Sets the environment.

        :param env:
        """
        if self.env is not None:
            raise ValueError("Trying to set env of already initialized environment.")

        self.env = env

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> DictReplayBufferSamples:
        """
        Abstract method from base class.
        """
        raise NotImplementedError()

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> DictReplayBufferSamples:
        """
        Sample function for online sampling of HER transition,
        this replaces the "regular" replay buffer ``sample()``
        method in the ``train()`` function.

        :param batch_size: Number of element to sample
        :param env: Associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return: Samples.
        """
        if self.replay_buffer is not None:
            return self.replay_buffer.sample(batch_size, env)
        return self._sample_transitions(batch_size, maybe_vec_env=env, online_sampling=True)  # pytype: disable=bad-return-type

    def _sample_offline(
        self,
        n_sampled_goal: Optional[int] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """
        Sample function for offline sampling of HER transition,
        in that case, only one episode is used and transitions
        are added to the regular replay buffer.

        :param n_sampled_goal: Number of sampled goals for replay
        :return: at most(n_sampled_goal * episode_length) HER transitions.
        """
        # `maybe_vec_env=None` as we should store unnormalized transitions,
        # they will be normalized at sampling time
        return self._sample_transitions(
            batch_size=None,
            maybe_vec_env=None,
            online_sampling=False,
            n_sampled_goal=n_sampled_goal,
        )

    def sample_goals(
        self,
        episode_indices: np.ndarray,
        her_indices: np.ndarray,
        transitions_indices: np.ndarray,
    ) -> np.ndarray:
        """
        Sample goals based on goal_selection_strategy.
        This is a vectorized (fast) version.

        :param episode_indices: Episode indices to use.
        :param her_indices: HER indices.
        :param transitions_indices: Transition indices to use.
        :return: Return sampled goals.
        """
        her_episode_indices = episode_indices[her_indices]

        if self.goal_selection_strategy == GoalSelectionStrategy.FINAL:
            # replay with final state of current episode
            transitions_indices = self.episode_lengths[her_episode_indices] - 1

        elif self.goal_selection_strategy == GoalSelectionStrategy.FUTURE:
            # replay with random state which comes from the same episode and was observed after current transition
            transitions_indices = np.random.randint(
                transitions_indices[her_indices], self.episode_lengths[her_episode_indices]
            )

        elif self.goal_selection_strategy == GoalSelectionStrategy.EPISODE:
            # replay with random state which comes from the same episode as current transition
            transitions_indices = np.random.randint(self.episode_lengths[her_episode_indices])

        else:
            raise ValueError(f"Strategy {self.goal_selection_strategy} for sampling goals not supported!")

        return self._buffer["next_achieved_goal"][her_episode_indices, transitions_indices]



    def add_full_trajectories_HSM(
            self,
            trajectory,
            hsm_trajectories
            ):

        self.n_traj += 1
        if np.linalg.norm(trajectory[0][0]["achieved_goal"] - trajectory[-1][1]["achieved_goal"])>0.02:
            self.n_dx_traj += 1
        if sum([trajectory[idx][3]*1.0**idx for idx in range(0, len(trajectory))])>-49:
            self.n_success_traj += 1
        if (np.linalg.norm(trajectory[0][0]["achieved_goal"] - trajectory[-1][1]["achieved_goal"])>0.02 and
                sum([trajectory[idx][3]*1.0**idx for idx in range(0, len(trajectory))])>-49):
            self.n_success_dx_traj += 1

        # store hsm trajectories in buffer
        if self.prioritized_replay_baseline:
            # use 'normal' trajectories
            for hsm_trajectory in hsm_trajectories:
                self.hsm_traj_buffer.append((trajectory, 0, self.idx_eps_hsm))
        else:
            # use extra trajectories
            idx_env = 0
            for hsm_trajectory in hsm_trajectories:
                self.hsm_traj_buffer.append((hsm_trajectory, idx_env, self.idx_eps_hsm))
                idx_env += 1
        self.idx_eps_hsm += 1
        # calculate criterion for every trajectory in hsm trajectories buffer
        if len(self.hsm_traj_buffer)>=self.HSM_n_traj_freq:
            hsm_trajectories_criterion = []
            for hsm_trajectory, idx_env, idx_eps in self.hsm_traj_buffer:
                if self.hindsight_state_selection_strategy == HindsightStateSelectionStrategy.RANDOM and self.hindsight_state_selection_strategy_horizon == HindsightStateSelectionStrategyHorizon.EPISODE:
                    criterion = -np.random.random()
                elif self.hindsight_state_selection_strategy == HindsightStateSelectionStrategy.REWARD and self.hindsight_state_selection_strategy_horizon == HindsightStateSelectionStrategyHorizon.EPISODE:
                    criterion = -sum([hsm_trajectory[idx][3]*1.0**idx for idx in range(0, len(hsm_trajectory))])
                elif self.hindsight_state_selection_strategy == HindsightStateSelectionStrategy.ACHIEVED_GOAL and self.hindsight_state_selection_strategy_horizon == HindsightStateSelectionStrategyHorizon.EPISODE:
                    criterion = -float(np.linalg.norm(hsm_trajectory[0][0]["achieved_goal"] - hsm_trajectory[-1][1]["achieved_goal"])>self.HSM_min_criterion)
                elif self.hindsight_state_selection_strategy == HindsightStateSelectionStrategy.REWARD_ACHIEVED_GOAL_002 and self.hindsight_state_selection_strategy_horizon == HindsightStateSelectionStrategyHorizon.EPISODE:
                    if np.linalg.norm(hsm_trajectory[0][0]["achieved_goal"] - hsm_trajectory[-1][1]["achieved_goal"])>0.02:
                        criterion = -sum([hsm_trajectory[idx][3]*1.0**idx for idx in range(0, len(hsm_trajectory))])
                    else:
                        criterion = 9999   # do not use
                else:
                    raise ValueError(f"Strategy {self.hindsight_state_selection_strategy} - {self.hindsight_state_selection_strategy_horizon} for sampling hindsight states is not supported!")
                hsm_trajectories_criterion.append((criterion, hsm_trajectory, idx_eps, idx_env))

            n_to_select = self.n_sampled_hindsight_states * self.HSM_n_traj_freq // self.HSM_shape
            hsm_criteria = np.array([t[0] for t in hsm_trajectories_criterion])
            print("crit:", hsm_criteria, "n:", n_to_select)
            indices = np.argpartition(hsm_criteria, n_to_select)[:n_to_select]
            print("ind:", indices)

            HSM_min_criterion_eff = self.HSM_min_criterion + self.HSM_criterion_change_per_step * self.n_total_steps

            for idx in indices:
                criterion, trajectory, _, _ = hsm_trajectories_criterion[idx]
                if -1.0*criterion>=HSM_min_criterion_eff:
                    self.n_hsm_traj += 1
                    if np.linalg.norm(trajectory[0][0]["achieved_goal"] - trajectory[-1][1]["achieved_goal"])>0.02:
                        self.n_dx_hsm_traj += 1
                    if sum([trajectory[idx][3]*1.0**idx for idx in range(0, len(trajectory))])>-49:
                        self.n_success_hsm_traj += 1
                    if (np.linalg.norm(trajectory[0][0]["achieved_goal"] - trajectory[-1][1]["achieved_goal"])>0.02 and
                         sum([trajectory[idx][3]*1.0**idx for idx in range(0, len(trajectory))])>-49):
                        self.n_success_dx_hsm_traj += 1
                    for obs, next_obs, action, reward, done, infos in trajectory:
                        self.add(obs, next_obs, action, reward, done, infos)

            self.hsm_traj_buffer = []

        if self.gym_robotics_logger:
            self.gym_robotics_logger.record("buffer/n_hsm_traj", self.n_hsm_traj)
            self.gym_robotics_logger.record("buffer/n_success_hsm_traj", self.n_success_hsm_traj)
            self.gym_robotics_logger.record("buffer/n_dx_hsm_traj", self.n_dx_hsm_traj)
            self.gym_robotics_logger.record("buffer/n_success_dx_hsm_traj", self.n_success_dx_hsm_traj)
            self.gym_robotics_logger.record("buffer/n_traj", self.n_traj)
            self.gym_robotics_logger.record("buffer/n_success_traj", self.n_success_traj)
            self.gym_robotics_logger.record("buffer/n_dx_traj", self.n_dx_traj)
            self.gym_robotics_logger.record("buffer/n_success_dx_traj", self.n_success_dx_traj)


    def add_trajectories_HSM(
            self,
            trajectory,
            hsm_trajectories
            ):

        if self.online_sampling:
            self.add_full_trajectories_HSM(trajectory, hsm_trajectories)
        else:
            self.add_partial_trajectories_HSM(trajectory, hsm_trajectories)


    def add_partial_trajectories_HSM(
            self,
            trajectory,
            hsm_trajectories
            ):

        # store hsm trajectories in buffer
        if self.prioritized_replay_baseline:
            # use 'normal' trajectories
            for hsm_trajectory in hsm_trajectories:
                self.hsm_traj_buffer.append((trajectory, trajectory, 0, self.idx_eps_hsm))
        else:
            # use extra trajectories
            idx_env = 0
            for hsm_trajectory in hsm_trajectories:
                self.hsm_traj_buffer.append((hsm_trajectory, trajectory, idx_env, self.idx_eps_hsm))
                idx_env += 1
        self.idx_eps_hsm += 1

        # calculate criterion for every transition in hsm trajectories buffer
        if len(self.hsm_traj_buffer)>=self.HSM_n_traj_freq:
            hsm_transitions = []
            for hsm_trajectory, trajectory, idx_env, idx_eps in self.hsm_traj_buffer:
                for idx_trans in range(len(hsm_trajectory)):
                    transition = hsm_trajectory[idx_trans]
                    if self.hindsight_state_selection_strategy == HindsightStateSelectionStrategy.RANDOM:
                        criterion = -np.random.random()
                    elif self.hindsight_state_selection_strategy == HindsightStateSelectionStrategy.REWARD:
                        if self.hindsight_state_selection_strategy_horizon == HindsightStateSelectionStrategyHorizon.STEP:
                            criterion = -transition[3]
                        elif self.hindsight_state_selection_strategy_horizon == HindsightStateSelectionStrategyHorizon.FUTURE:
                            criterion = -sum([hsm_trajectory[idx][3]*self.HSM_gamma**(idx-idx_trans) for idx in range(idx_trans, len(hsm_trajectory))])
                        elif self.hindsight_state_selection_strategy_horizon == HindsightStateSelectionStrategyHorizon.EPISODE:
                            criterion = -sum([hsm_trajectory[idx][3]*1.0**idx for idx in range(0, len(hsm_trajectory))])
                        else:
                            raise ValueError(f"Strategy {self.hindsight_state_selection_strategy} - {self.hindsight_state_selection_strategy_horizon} for sampling hindsight states is not supported!")
                    elif self.hindsight_state_selection_strategy == HindsightStateSelectionStrategy.ADVANTAGE:
                        ob, next_ob, action, reward, done, _ = transition
                        if self.hindsight_state_selection_strategy_horizon == HindsightStateSelectionStrategyHorizon.STEP:
                            ob_1 = ob
                            ob_2 = next_ob
                            reward_sum = reward
                        elif self.hindsight_state_selection_strategy_horizon == HindsightStateSelectionStrategyHorizon.FUTURE:
                            ob_1 = ob
                            reward_sum = sum([hsm_trajectory[idx][3]*self.HSM_gamma**(idx-idx_trans) for idx in range(idx_trans, len(hsm_trajectory))])
                        elif self.hindsight_state_selection_strategy_horizon == HindsightStateSelectionStrategyHorizon.EPISODE:
                            ob_1 = hsm_trajectory[0][0]
                            reward_sum = sum([hsm_trajectory[idx][3]*self.HSM_gamma**idx for idx in range(0, len(hsm_trajectory))])
                        else:
                            raise ValueError(f"Strategy {self.hindsight_state_selection_strategy} - {self.hindsight_state_selection_strategy_horizon} for sampling hindsight states is not supported!")

                        action = self.to_torch([action])
                        ob_norm_1 = self._normalize_obs(ob_1)
                        ob_norm_1 = {key: self.to_torch([ob_norm_1[key]]) for key in self._observation_keys}
                        if self.hindsight_state_selection_strategy_horizon == HindsightStateSelectionStrategyHorizon.STEP:
                            ob_norm_2 = self._normalize_obs(ob_2)
                            ob_norm_2 = {key: self.to_torch([ob_norm_2[key]]) for key in self._observation_keys}
                            action_pi_2, _ = self.HSM_policy.actor.action_log_prob(ob_norm_2)
                            q_value_pi_2 = th.cat(self.HSM_critic(ob_norm_2, action_pi_2), dim=1)
                            min_qf_pi_2, _ = th.min(q_value_pi_2, dim=1, keepdim=True)
                            min_qf_pi_2 = min_qf_pi_2[0][0].item()
                        else:
                            min_qf_pi_2 = 0
                        q_value = th.cat(self.HSM_critic(ob_norm_1, action), dim=1)
                        min_q_value, _ = th.min(q_value, dim=1, keepdim=True)
                        min_q_value = min_q_value[0][0].item()
                        advantage = reward_sum + self.HSM_gamma * min_qf_pi_2 *(1 - done) -  min_q_value
                        criterion = -abs(advantage)
                    else:
                        raise ValueError(f"Strategy {self.hindsight_state_selection_strategy} - {self.hindsight_state_selection_strategy_horizon} for sampling hindsight states is not supported!")

                    if self.HSM_use_likelihood_ratio and idx_trans<len(trajectory):


                        ob_norm = self._normalize_obs(ob)
                        ob_norm = {key: self.to_torch([ob_norm[key]]) for key in self._observation_keys}
                        mean_actions, log_std, _ = self.HSM_policy.actor.get_action_dist_params(ob_norm)
                        self.HSM_policy.actor.action_dist.proba_distribution(mean_actions, log_std)
                        log_likelihood = self.HSM_policy.actor.action_dist.log_prob(action)

                        ob_real = trajectory[idx_trans][0]
                        ob_real_norm = self._normalize_obs(ob_real)
                        ob_real_norm = {key: self.to_torch([ob_real_norm[key]]) for key in self._observation_keys}
                        mean_actions, log_std, _ = self.HSM_policy.actor.get_action_dist_params(ob_real_norm)
                        self.HSM_policy.actor.action_dist.proba_distribution(mean_actions, log_std)
                        log_likelihood_real = self.HSM_policy.actor.action_dist.log_prob(action)

                        log_likelihood = log_likelihood[0].item()
                        log_likelihood_real = log_likelihood_real[0].item()

                        is_factor = np.exp(log_likelihood - log_likelihood_real)
                        is_factor = min(self.HSM_likelihood_ratio_cutoff, is_factor)
                        is_factor = max(is_factor, self.HSM_likelihood_ratio_cutoff)

                        if idx_trans==0:
                            is_factor_running_avg = is_factor
                        else:
                            is_factor_running_avg = is_factor_running_avg* 0.9 + is_factor*0.1

                        criterion_new = is_factor * criterion
                        criterion = criterion_new

                    elif self.HSM_use_likelihood_ratio and (not idx_trans<len(trajectory)):
                        criterion_new = criterion * is_factor_running_avg
                        criterion = criterion_new

                    hsm_transitions.append((criterion, transition, idx_trans, idx_eps, idx_env))

            #store in log file
            if self.HSM_logging:
                with open ("/tmp/log_all.txt", "a+") as f:
                    f.write(repr(hsm_transitions) + "\n\n")
                    print("------log------")


            # sort by criterion and add to replay buffer

            n_sampled_hindsight_states_eff = max([0, self.n_sampled_hindsight_states + self.n_sampled_hindsight_states_change_per_step * self.n_total_steps])
            HSM_min_criterion_eff = self.HSM_min_criterion + self.HSM_criterion_change_per_step * self.n_total_steps

            n_to_select = int(len(hsm_transitions) * n_sampled_hindsight_states_eff / self.HSM_shape)
            random.shuffle(hsm_transitions)
            hsm_criteria = np.array([t[0] for t in hsm_transitions])
            indices = np.argpartition(hsm_criteria, n_to_select)[:n_to_select]

            hsm_transitions_selected = []
            for idx in indices:
                criterion, transition, _, _, _ = hsm_transitions[idx]
                if -1.0*criterion>=HSM_min_criterion_eff:
                    self.replay_buffer.add(*transition)

                    # update current pointer
                    self.current_idx += 1
                    self.episode_steps += 1

                    if self.episode_steps >= self.max_episode_length or idx==indices[-1]:
                        self.store_episode()
                        if not self.online_sampling:
                            # clear storage for current episode
                            self.reset()
                        self.episode_steps = 0

                    hsm_transitions_selected.append(hsm_transitions[idx])

            if self.HSM_logging:
                with open ("/tmp/log_select.txt", "a+") as f:
                    f.write(repr(hsm_transitions_selected) + "\n\n")

            self.hsm_traj_buffer = []

        # add all 'normally' collected transitions
        # ball_hit = trajectory[-1][3]>0.01

        # for transition in trajectory:
        #     self.replay_buffer.add(*transition)

        return


    def _sample_transitions(
        self,
        batch_size: Optional[int],
        maybe_vec_env: Optional[VecNormalize],
        online_sampling: bool,
        n_sampled_goal: Optional[int] = None,
    ) -> Union[DictReplayBufferSamples, Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, np.ndarray]]:
        """
        :param batch_size: Number of element to sample (only used for online sampling)
        :param env: associated gym VecEnv to normalize the observations/rewards
            Only valid when using online sampling
        :param online_sampling: Using online_sampling for HER or not.
        :param n_sampled_goal: Number of sampled goals for replay. (offline sampling)
        :return: Samples.
        """
        # Select which episodes to use
        if online_sampling:
            assert batch_size is not None, "No batch_size specified for online sampling of HER transitions"
            # Do not sample the episode with index `self.pos` as the episode is invalid
            if self.full:
                episode_indices = (
                    np.random.randint(1, self.n_episodes_stored, batch_size) + self.pos
                ) % self.n_episodes_stored
            else:
                episode_indices = np.random.randint(0, self.n_episodes_stored, batch_size)
            # A subset of the transitions will be relabeled using HER algorithm
            her_indices = np.arange(batch_size)[: int(self.her_ratio * batch_size)]
        else:
            assert maybe_vec_env is None, "Transitions must be stored unnormalized in the replay buffer"
            assert n_sampled_goal is not None, "No n_sampled_goal specified for offline sampling of HER transitions"
            # Offline sampling: there is only one episode stored
            episode_length = self.episode_lengths[0]
            # we sample n_sampled_goal per timestep in the episode (only one is stored).
            episode_indices = np.tile(0, (episode_length * n_sampled_goal))
            # we only sample virtual transitions
            # as real transitions are already stored in the replay buffer
            her_indices = np.arange(len(episode_indices))

        ep_lengths = self.episode_lengths[episode_indices]

        if online_sampling:
            # Select which transitions to use
            transitions_indices = np.random.randint(ep_lengths)
        else:
            if her_indices.size == 0:
                # Episode of one timestep, not enough for using the "future" strategy
                # no virtual transitions are created in that case
                return {}, {}, np.zeros(0), np.zeros(0)
            else:
                # Repeat every transition index n_sampled_goals times
                # to sample n_sampled_goal per timestep in the episode (only one is stored).
                # Now with the corrected episode length when using "future" strategy
                transitions_indices = np.tile(np.arange(ep_lengths[0]), n_sampled_goal)
                episode_indices = episode_indices[transitions_indices]
                her_indices = np.arange(len(episode_indices))

        # get selected transitions
        transitions = {key: self._buffer[key][episode_indices, transitions_indices].copy() for key in self._buffer.keys()}

        if self.apply_HER:
            # sample new desired goals and relabel the transitions
            new_goals = self.sample_goals(episode_indices, her_indices, transitions_indices)
            transitions["desired_goal"][her_indices] = new_goals




        # Convert info buffer to numpy array
        transitions["info"] = np.array(
            [
                self.info_buffer[episode_idx][transition_idx]
                for episode_idx, transition_idx in zip(episode_indices, transitions_indices)
            ]
        )

        # Edge case: episode of one timesteps with the future strategy
        # no virtual transition can be created
        if len(her_indices) > 0:
            # Vectorized computation of the new reward
            transitions["reward"][her_indices, 0] = self.env.env_method(
                "compute_reward",
                # the new state depends on the previous state and action
                # s_{t+1} = f(s_t, a_t)
                # so the next_achieved_goal depends also on the previous state and action
                # because we are in a GoalEnv:
                # r_t = reward(s_t, a_t) = reward(next_achieved_goal, desired_goal)
                # therefore we have to use "next_achieved_goal" and not "achieved_goal"
                transitions["next_achieved_goal"][her_indices, 0],
                # here we use the new desired goal
                transitions["desired_goal"][her_indices, 0],
                transitions["info"][her_indices, 0],
            )

        # concatenate observation with (desired) goal
        observations = self._normalize_obs(transitions, maybe_vec_env)

        # HACK to make normalize obs and `add()` work with the next observation
        next_observations = {
            "observation": transitions["next_obs"],
            "achieved_goal": transitions["next_achieved_goal"],
            # The desired goal for the next observation must be the same as the previous one
            "desired_goal": transitions["desired_goal"],
        }
        next_observations = self._normalize_obs(next_observations, maybe_vec_env)

        if online_sampling:
            next_obs = {key: self.to_torch(next_observations[key][:, 0, :]) for key in self._observation_keys}

            normalized_obs = {key: self.to_torch(observations[key][:, 0, :]) for key in self._observation_keys}

            return DictReplayBufferSamples(
                observations=normalized_obs,
                actions=self.to_torch(transitions["action"]),
                next_observations=next_obs,
                dones=self.to_torch(transitions["done"]),
                rewards=self.to_torch(self._normalize_reward(transitions["reward"], maybe_vec_env)),
            )
        else:
            return observations, next_observations, transitions["action"], transitions["reward"]

    def add(
        self,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:

        if self.current_idx == 0 and self.full:
            # Clear info buffer
            self.info_buffer[self.pos] = deque(maxlen=self.max_episode_length)

        # Remove termination signals due to timeout
        if self.handle_timeout_termination:
            done_ = done * (1 - np.array([info.get("TimeLimit.truncated", False) for info in infos]))
        else:
            done_ = done

        if not hasattr(reward, "__len__"):
            reward = [reward]

        self._buffer["observation"][self.pos][self.current_idx] = obs["observation"]
        self._buffer["action"][self.pos][self.current_idx] = action
        self._buffer["done"][self.pos][self.current_idx] = done_
        self._buffer["reward"][self.pos][self.current_idx] = reward
        self._buffer["next_obs"][self.pos][self.current_idx] = next_obs["observation"]
        if self.apply_HER or self.HSM_goal_env:
            self._buffer["achieved_goal"][self.pos][self.current_idx] = obs["achieved_goal"]
            self._buffer["desired_goal"][self.pos][self.current_idx] = obs["desired_goal"]
            self._buffer["next_achieved_goal"][self.pos][self.current_idx] = next_obs["achieved_goal"]
            self._buffer["next_desired_goal"][self.pos][self.current_idx] = next_obs["desired_goal"]

        # When doing offline sampling
        # Add real transition to normal replay buffer
        if self.replay_buffer is not None:
            self.replay_buffer.add(
                obs,
                next_obs,
                action,
                reward,
                done,
                infos,
            )
            self.n_total_steps += 1


        # remove extra data from info for info buffer
        infos_without_extras = []
        for info in infos:
            info_without_extras = info.copy()
            for key in ("extra_obs",
                    "extra_rewards",
                    "extra_terminated",
                    "extra_truncated",
                    "extra_is_success",
                    "initial_extra_obs",
                    "trajectory",
                    "hsm_trajectories"):
                if key in info_without_extras:
                    del info_without_extras[key]
            infos_without_extras.append(info_without_extras)
        self.info_buffer[self.pos].append(infos_without_extras)

        # update current pointer
        self.current_idx += 1

        self.episode_steps += 1

        if done or self.episode_steps >= self.max_episode_length:
            self.store_episode()
            if not self.online_sampling:
                if self.apply_HER:
                    # sample virtual transitions and store them in replay buffer
                    self._sample_her_transitions()
                # clear storage for current episode
                self.reset()

            self.episode_steps = 0


        #add transitions when using hsm
        if self.apply_HSM:
            for info in infos:
                if "trajectory" in info:
                    self.add_trajectories_HSM(
                        info["trajectory"], info["hsm_trajectories"]
                    )
                if "extra_obs" in info:
                    extra_obs = info["extra_obs"]
                    extra_rewards = info["extra_rewards"]
                    extra_dones = info["extra_terminated"]
                    info_without_extras = info.copy()
                    del info_without_extras["extra_obs"]
                    del info_without_extras["extra_rewards"]
                    del info_without_extras["extra_terminated"]
                    del info_without_extras["extra_truncated"]
                    del info_without_extras["extra_is_success"]
                    del info_without_extras["initial_extra_obs"]
                    info_for_extras = {}
                    if info.get("TimeLimit.truncated", False):
                        info_for_extras = {'TimeLimit.truncated': True}
                        extra_dones = [True for _ in extra_dones]
                    if self.extra_reset:
                        self.extra_current_obs = info["initial_extra_obs"]
                        self.extra_reset = False
                    self.current_trajectory.append((
                        obs,
                        next_obs,
                        action,
                        reward,
                        done,
                        [info_without_extras]))
                    for idx in range(len(extra_obs)):
                        for key, value in self.extra_current_obs[idx].items():
                            if value.ndim==1:
                                self.extra_current_obs[idx][key] = np.array([value])
                        for key, value in  extra_obs[idx].items():
                            if value.ndim==1:
                                extra_obs[idx][key] = np.array([value])
                        self.current_hsm_trajectories[idx].append((
                            self.extra_current_obs[idx],
                            extra_obs[idx],
                            action,
                            extra_rewards[idx],
                            extra_dones[idx],
                            [info_for_extras]))



                    self.extra_current_obs = extra_obs.copy()

                    ep_over = done_ or done or np.any(np.array([info.get("TimeLimit.truncated", False) for info in infos]))
                    if ep_over:
                        self.add_trajectories_HSM(self.current_trajectory, self.current_hsm_trajectories)
                        self.current_trajectory = []
                        self.current_hsm_trajectories = [[]  for _ in range(len(extra_obs))]
                        self.extra_reset = True



       

              







    def store_episode(self) -> None:
        """
        Increment episode counter
        and reset transition pointer.
        """
        # add episode length to length storage
        self.episode_lengths[self.pos] = self.current_idx

        # update current episode pointer
        # Note: in the OpenAI implementation
        # when the buffer is full, the episode replaced
        # is randomly chosen
        self.pos += 1
        if self.pos == self.max_episode_stored:
            self.full = True
            self.pos = 0
        # reset transition pointer
        self.current_idx = 0

    def _sample_her_transitions(self) -> None:
        """
        Sample additional goals and store new transitions in replay buffer
        when using offline sampling.
        """

        # Sample goals to create virtual transitions for the last episode.
        observations, next_observations, actions, rewards = self._sample_offline(n_sampled_goal=self.n_sampled_goal)

        # Store virtual transitions in the replay buffer, if available
        if len(observations) > 0:
            for i in range(len(observations["observation"])):
                self.replay_buffer.add(
                    {key: obs[i] for key, obs in observations.items()},
                    {key: next_obs[i] for key, next_obs in next_observations.items()},
                    actions[i],
                    rewards[i],
                    # We consider the transition as non-terminal
                    done=[False],
                    infos=[{}],
                )

    @property
    def n_episodes_stored(self) -> int:
        if self.full:
            return self.max_episode_stored
        return self.pos

    def size(self) -> int:
        """
        :return: The current number of transitions in the buffer.
        """
        return int(np.sum(self.episode_lengths))

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.current_idx = 0
        self.full = False
        self.episode_lengths = np.zeros(self.max_episode_stored, dtype=np.int64)

    def truncate_last_trajectory(self) -> None:
        """
        Only for online sampling, called when loading the replay buffer.
        If called, we assume that the last trajectory in the replay buffer was finished
        (and truncate it).
        If not called, we assume that we continue the same trajectory (same episode).
        """
        # If we are at the start of an episode, no need to truncate
        current_idx = self.current_idx

        # truncate interrupted episode
        if current_idx > 0:
            warnings.warn(
                "The last trajectory in the replay buffer will be truncated.\n"
                "If you are in the same episode as when the replay buffer was saved,\n"
                "you should use `truncate_last_trajectory=False` to avoid that issue."
            )
            # get current episode and transition index
            pos = self.pos
            # set episode length for current episode
            self.episode_lengths[pos] = current_idx
            # set done = True for current episode
            # current_idx was already incremented
            self._buffer["done"][pos][current_idx - 1] = np.array([True], dtype=np.float32)
            # reset current transition index
            self.current_idx = 0
            # increment episode counter
            self.pos = (self.pos + 1) % self.max_episode_stored
            # update "full" indicator
            self.full = self.full or self.pos == 0
