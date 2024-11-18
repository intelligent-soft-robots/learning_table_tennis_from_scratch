import argparse
import itertools
import pickle
import re
import traceback
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import trange

from learning_table_tennis_from_scratch.hysr_one_ball_env import HysrOneBallEnv


def get_outpath(outdir: Path, job_id: str, intervention: int) -> Path:
    job_str = f"_job{job_id}" if job_id != "" else ""
    return outdir / f"data{job_str}_intervention{intervention}.pkl"


def make_env(
    env: str,
    job_id: str = "",
    seed: Optional[int] = None,
    tabletennis_config_dir: Optional[Path] = None,
    **gym_kwargs,
):
    if env == "tabletennis":
        dict_config = {
            "reward_config_file": tabletennis_config_dir / "reward_config.json",
            "hysr_one_ball_config_file": tabletennis_config_dir / "hysr_config.json",
            "log_episodes": False,
            "logger": None,
            "job_id": job_id,
        }
        return make_vec_env(HysrOneBallEnv, env_kwargs=dict_config, seed=seed)
    else:
        vec_env = DummyVecEnv([lambda: gym.make(env, **gym_kwargs)])
        if seed is not None:
            vec_env.seed(seed)
        return vec_env


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_logs", type=Path)
    parser.add_argument("env", type=str)
    parser.add_argument("intervention_std", type=float)
    parser.add_argument("--env-name", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num-interventions", type=int, default=100)
    parser.add_argument("--num-episodes-per-intervention", type=int, default=20)
    parser.add_argument("--max-episode-length", type=int)
    parser.add_argument(
        "--outdir", type=Path, default=Path(__file__).parents[1] / "out" / "tcr_dataset"
    )
    parser.add_argument("--job-id", type=str)
    args = parser.parse_args()

    if not args.outdir.exists():
        args.outdir.mkdir(parents=True)
    start_intervention = 0
    if len(list(args.outdir.iterdir())) != 0:
        print(f"Directory {args.outdir} is not empty. Continuing data collection.")
        for start_intervention in range(args.num_interventions):
            if not get_outpath(args.outdir, args.job_id, start_intervention).exists():
                break

    if args.seed is not None:
        set_random_seed(args.seed)

    vec_env = make_env(args.env, args.job_id, args.seed, args.train_logs / "config")
    checkpoints_dir = args.train_logs / "checkpoints"
    checkpoint_path = sorted(
        [p for p in checkpoints_dir.iterdir()],
        key=lambda p: int(re.search("_\d+_", p.name).group()[1:-1]),
    )[-1]

    # NOTE: It's important to set the seed when loading the model.  Otherwise, the
    # RNG state will be restored from the loaded model, resulting in all runs to
    # behave the same.
    agent = PPO.load(checkpoint_path, vec_env, seed=args.seed, device="cpu")

    env = vec_env.envs[0]
    print("Starting data collection")
    avg_reward = 0.0
    for i in trange(
        start_intervention,
        args.num_interventions,
        initial=start_intervention,
        total=args.num_interventions,
    ):
        max_episode_length = (
            args.max_episode_length
            if args.max_episode_length is not None
            else env.get_wrapper_attr("_max_episode_steps")
        )
        intervention_trajectory = np.random.normal(
            scale=args.intervention_std, size=(max_episode_length, 8)
        )
        observation_trajectories = []
        action_trajectories = []
        reward_trajectories = []
        # Make sure that a different ball trajectory is used in every episode (otherwise multiple episodes would just be
        # the same...)
        if isinstance(env.unwrapped, HysrOneBallEnv):
            num_ball_trajectories = (
                env.unwrapped._hysr._ball_behavior._trajectory_reader.size()
            )
            ball_trajectory_indices = np.random.choice(
                np.arange(num_ball_trajectories),
                size=args.num_episodes_per_intervention,
                replace=False,
            )
        j = 0
        while j < args.num_episodes_per_intervention:
            observations = []
            actions = []
            rewards = []
            if isinstance(env.unwrapped, HysrOneBallEnv):
                ball_traj_idx = ball_trajectory_indices[j]
                env.unwrapped._hysr._ball_behavior.type = (
                    env.unwrapped._hysr._ball_behavior.INDEX
                )
                env.unwrapped._hysr._ball_behavior.value = ball_traj_idx
            try:
                obs, _ = env.reset()
                observations.append(obs)
                for k in itertools.count():
                    action, _ = agent.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = env.step(
                        action + intervention_trajectory[k]
                    )
                    done = terminated or truncated
                    observations.append(obs)
                    actions.append(action)
                    rewards.append(reward)
                    if done:
                        break
                observations_np = np.array(observations)
                actions_np = np.array(actions)
                rewards_np = np.array(rewards)
                if (
                    not np.any(np.isnan(observations_np))
                    and not np.any(np.isnan(actions_np))
                    and not np.any(np.isnan(rewards_np))
                ):
                    observation_trajectories.append(observations_np)
                    action_trajectories.append(actions_np)
                    reward_trajectories.append(rewards_np)
                    j += 1
                else:
                    nan_value_field_names = [
                        field_name
                        for field_name, field_values in zip(
                            ["observations", "actions", "rewards"],
                            [observations_np, actions_np, rewards_np],
                        )
                        if np.any(np.isnan(field_values))
                    ]
                    print(f"Encountered NaN values for index {j} ({', '.join(nan_value_field_names)}). Retrying...")
            except:
                print(
                    f"Encountered exception for index {j}: \n{traceback.format_exc()}. Retrying..."
                )

        num_rec_int = i - start_intervention
        avg_reward = avg_reward * num_rec_int / (num_rec_int + 1) + np.mean(
            [np.sum(r) for r in reward_trajectories]
        ) / (num_rec_int + 1)
        print(f"Average reward: {avg_reward}")
        outpath = get_outpath(args.outdir, args.job_id, i)
        with outpath.open("wb") as outfile:
            episode_data = {
                "observations": observation_trajectories,
                "actions": action_trajectories,
                "rewards": reward_trajectories,
            }
            pickle.dump((intervention_trajectory, episode_data), outfile)
