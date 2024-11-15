import argparse
import itertools
import json
import logging
import pickle
import re
from pathlib import Path
import traceback

import numpy as np
import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

from tcr_data_collection import make_env

logger = logging.getLogger(__name__)

from learning_table_tennis_from_scratch.hysr_one_ball_env import HysrOneBallEnv
from learning_table_tennis_from_scratch.rl_config import RLConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_logs", type=Path)
    parser.add_argument("--env-config", type=Path)
    parser.add_argument("--env-name", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num-interventions", type=int, default=50)
    parser.add_argument("--max-intervention-std", type=float, default=1.0)
    parser.add_argument("--num-episodes-per-intervention", type=int, default=100)
    parser.add_argument("--max-episode-length", type=int, default=200)
    parser.add_argument(
        "--outfile", type=Path, default=Path(__file__).parents[1] / "out" / "tcr_intervention_scales.pkl"
    )
    args = parser.parse_args()

    if args.env_config is not None:
        env_config = args.env_config
    elif args.env_name is not None:
        env_config = args.env_name
    else:
        raise ValueError("Either --env-config or --env-name must be specified")


    if args.seed is not None:
        set_random_seed(args.seed)

    vec_env = make_env(env_config)

    checkpoints_dir = args.train_logs / "checkpoints"
    checkpoint_path = sorted(
        [p for p in checkpoints_dir.iterdir()],
        key=lambda p: int(re.search("_\d+_", p.name).group()[1:-1]),
    )[-1]

    # NOTE: It's important to set the seed when loading the model. Otherwise, the
    # RNG state will be restored from the loaded model, resulting in all runs to
    # behave the same.
    agent = PPO.load(checkpoint_path, vec_env, seed=args.seed, device="cpu")

    env = vec_env.envs[0]
    print("Starting data collection")
    rewards_int_stds = {}
    dist_target_int_stds = {}
    intervention_stds = np.linspace(0.0, args.max_intervention_std, args.num_interventions + 1)
    for intervention_std in tqdm.tqdm(intervention_stds):
        rewards_int_stds[intervention_std] = []
        dist_target_int_stds[intervention_std] = []
        if isinstance(env.unwrapped, HysrOneBallEnv):
            num_ball_trajectories = env.unwrapped._hysr._ball_behavior._trajectory_reader.size()
            ball_trajectory_indices = np.random.choice(
                np.arange(num_ball_trajectories),
                size=args.num_episodes_per_intervention,
                replace=False,
            )
        for i in range(args.num_episodes_per_intervention):
            if isinstance(env.unwrapped, HysrOneBallEnv):
                env.unwrapped._hysr._ball_behavior.type = (
                    env.unwrapped._hysr._ball_behavior.INDEX
                )
                env.unwrapped._hysr._ball_behavior.value = ball_trajectory_indices[i]

            intervention_trajectory = np.random.normal(
                scale=intervention_std, size=(args.max_episode_length, 8)
            )
            try:
                obs, _ = env.reset()
                rewards = []
                for k in itertools.count():
                    action, _ = agent.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = env.step(action + intervention_trajectory[k])
                    done = terminated or truncated
                    rewards.append(reward)
                    if done:
                        break
                rewards_int_stds[intervention_std].append(np.sum(rewards))
                if isinstance(env.unwrapped, HysrOneBallEnv):
                    dist_target_int_stds[intervention_std].append(env.unwrapped._hysr._ball_status.min_distance_ball_target)
            except Exception:
                print(f"Encountered exception for intervention std: {intervention_std}: \n{traceback.format_exc()}")

        with args.outfile.open("wb") as outfile:
            pickle.dump({"rewards": rewards_int_stds, "distances_to_target": dist_target_int_stds}, outfile)
