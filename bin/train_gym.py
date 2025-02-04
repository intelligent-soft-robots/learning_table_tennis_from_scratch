import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default="Pendulum-v1")
args = parser.parse_args()

if __name__ == '__main__':
    outdir = Path(__file__).parents[1] / "logs" / f"{args.env.lower().replace('-', '_')}"
    outdir.mkdir()
    agent = PPO("MlpPolicy", gym.make(args.env), tensorboard_log=outdir)
    checkpoint_cb = CheckpointCallback(10000, outdir)
    agent.learn(total_timesteps=1000000, callback=checkpoint_cb, progress_bar=True)
