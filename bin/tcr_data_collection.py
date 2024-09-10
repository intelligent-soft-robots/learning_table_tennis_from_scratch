import argparse
import itertools
import json
import pickle
import re
import traceback
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from tqdm import trange

from learning_table_tennis_from_scratch.hysr_one_ball_env import HysrOneBallEnv
from learning_table_tennis_from_scratch.rl_config import RLConfig

def get_outpath(outdir: Path, job_id: str, intervention: int) -> Path:
    job_str = f"_job{job_id}" if job_id != "" else ""
    return outdir / f"data{job_str}_intervention{intervention}.pkl"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_logs", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parents[1] / "example" / "config.json",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num-interventions", type=int, default=100)
    parser.add_argument("--num-episodes-per-intervention", type=int, default=20)
    parser.add_argument("--intervention-std", type=float, default=0.01)
    parser.add_argument("--max-episode-length", type=int, default=200)
    parser.add_argument(
        "--outdir", type=Path, default=Path(__file__).parents[1] / "out" / "tcr_dataset"
    )
    parser.add_argument("--job-id", type=str, default="")
    args = parser.parse_args()

    if not args.outdir.exists():
        args.outdir.mkdir(parents=True)
    start_intervention = 0
    if len(list(args.outdir.iterdir())) != 0:
        print(f"Directory {args.outdir} is not empty. Continuing data collection.")
        for start_intervention in range(args.num_interventions):
            if not get_outpath(args.outdir, args.job_id, start_intervention).exists():
                break

    with args.config.open() as f:
        config = json.load(f)

    algorithm = "ppo"

    if args.seed is not None:
        set_random_seed(args.seed)

    rl_config = RLConfig.from_json(config["rl_config"], algorithm)

    env_config = {
        "reward_config_file": config["reward_config"],
        "hysr_one_ball_config_file": config["hysr_config"],
        "log_episodes": False,
        "logger": None,
        "job_id": args.job_id,
    }
    vec_env = make_vec_env(HysrOneBallEnv, env_kwargs=env_config, seed=args.seed)
    checkpoints_dir = args.train_logs / "checkpoints"
    checkpoint_path = sorted(
        [p for p in checkpoints_dir.iterdir()],
        key=lambda p: int(re.search("_\d+_", p.name).group()[1:-1]),
    )[-1]

    # NOTE: It's important to set the seed when loading the model.  Otherwise the
    # RNG state will be restored from the loaded model, resulting in all runs to
    # behave the same.
    agent = PPO.load(checkpoint_path, vec_env, seed=args.seed, device="cpu")

    env = vec_env.envs[0]
    num_ball_trajectories = env.unwrapped._hysr._ball_behavior._trajectory_reader.size()
    print("Starting data collection")
    avg_reward = 0.0
    for i in trange(start_intervention, args.num_interventions, initial=start_intervention, total=args.num_interventions):
        intervention_trajectory = np.random.normal(
            scale=args.intervention_std, size=(args.max_episode_length, 8)
        )
        observation_trajectories = []
        action_trajectories = []
        reward_trajectories = []
        # Make sure that a different ball trajectory is used in every episode (otherwise multiple episodes would just be
        # the same...)
        ball_trajectory_indices = np.random.choice(
            np.arange(num_ball_trajectories),
            size=args.num_episodes_per_intervention,
            replace=False,
        )
        j = 0
        while j < len(ball_trajectory_indices):
            ball_traj_idx = ball_trajectory_indices[j]
            try:
                env.unwrapped._hysr._ball_behavior.type = (
                    env.unwrapped._hysr._ball_behavior.INDEX
                )
                env.unwrapped._hysr._ball_behavior.value = ball_traj_idx
                obs = env.reset()
                observations = [obs]
                actions = []
                rewards = []
                for k in itertools.count():
                    action, _ = agent.predict(obs, deterministic=True)
                    obs, reward, done, _ = env.step(action + intervention_trajectory[k])
                    observations.append(obs)
                    actions.append(action)
                    rewards.append(reward)
                    if done:
                        break
                observation_trajectories.append(np.stack(observations, axis=0))
                action_trajectories.append(np.stack(actions, axis=0))
                reward_trajectories.append(np.stack(rewards, axis=0))
                j += 1
            except:
                print(f"Encountered exception for ball trajectory index: {ball_traj_idx}: \n{traceback.format_exc()}. "
                      f"Retrying...")
        num_rec_int = i - start_intervention
        avg_reward = avg_reward * num_rec_int / (num_rec_int + 1) + np.mean([np.sum(r) for r in reward_trajectories]) / (num_rec_int + 1)
        print(f"Average reward: {avg_reward}")
        outpath = get_outpath(args.outdir, args.job_id, i)
        with outpath.open("wb") as outfile:
            episode_data = {
                "observations": observation_trajectories,
                "actions": action_trajectories,
                "rewards": reward_trajectories,
            }
            pickle.dump((intervention_trajectory, episode_data), outfile)
