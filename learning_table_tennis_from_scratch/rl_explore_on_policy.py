# source: https://github.com/RLE-Foundation/RLeXplore/blob/main/2%20rlexplore_with_sb3.ipynb

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.base_class import BaseAlgorithm

import torch as th

class RLeXploreWithOnPolicyRL(BaseCallback):
    """
    A custom callback for combining RLeXplore and on-policy algorithms from SB3.
    """
    def __init__(self, irs, verbose=0):
        super(RLeXploreWithOnPolicyRL, self).__init__(verbose)
        self.irs = irs
        self.buffer = None

    def init_callback(self, model: BaseAlgorithm) -> None:
        super().init_callback(model)
        self.buffer = self.model.rollout_buffer

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        observations = self.locals["obs_tensor"]
        device = observations.device
        actions = th.as_tensor(self.locals["actions"], device=device)
        rewards = th.as_tensor(self.locals["rewards"], device=device)
        dones = th.as_tensor(self.locals["dones"], device=device)
        next_observations = th.as_tensor(self.locals["new_obs"], device=device)

        # ===================== watch the interaction ===================== #
        self.irs.watch(observations, actions, rewards, dones, dones, next_observations)
        # ===================== watch the interaction ===================== #
        return True

    def _on_rollout_end(self) -> None:
        # prepare the data samples
        obs = th.as_tensor(self.buffer.observations)          # (n_steps, n_envs, obs_dim)
        new_obs = obs.clone()
        new_obs[:-1] = obs[1:]
        new_obs[-1] = th.as_tensor(self.locals["new_obs"])    # last next_obs

        actions = th.as_tensor(self.buffer.actions)           # (n_steps, n_envs, act_dim)
        rewards = th.as_tensor(self.buffer.rewards)           # (n_steps, n_envs)
        dones = th.as_tensor(self.buffer.episode_starts)      # (n_steps, n_envs)
        
        intrinsic_rewards = self.irs.compute(
            samples=dict(
                observations=obs,
                actions=actions,
                rewards=rewards,
                terminateds=dones,
                truncateds=dones,
                next_observations=new_obs,
            ),
            sync=True,
        )

        # Ensure shape matches rollout buffer exactly
        if intrinsic_rewards.shape != self.buffer.advantages.shape:
            print("Shape mismatch:",
                "intrinsic_rewards", intrinsic_rewards.shape,
                "buffer", self.buffer.advantages.shape)
            intrinsic_rewards = intrinsic_rewards.view_as(self.buffer.advantages)

        # Add intrinsic rewards
        ir_np = intrinsic_rewards.cpu().numpy()
        self.buffer.advantages += ir_np
        self.buffer.returns += ir_np