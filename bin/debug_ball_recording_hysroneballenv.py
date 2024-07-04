import time

import numpy as np

from learning_table_tennis_from_scratch.hysr_one_ball_env import HysrOneBallEnv
from stable_baselines3.common.env_util import make_vec_env

from learning_table_tennis_from_scratch.hysr_one_ball import HysrOneBallConfig, HysrOneBall

from learning_table_tennis_from_scratch.rewards import JsonReward

hysr_one_ball_config = HysrOneBallConfig.from_json("../config/hysr_config.json")
reward_function = JsonReward.get("../config/reward_config.json")

env_config = {
    "reward_config_file": "../config/reward_config.json",
    "hysr_one_ball_config_file": "../config/hysr_config.json",
    "log_episodes": False,
    "logger": None,
}
env = make_vec_env(HysrOneBallEnv, env_kwargs=env_config, seed=42)

observations = []
env.reset()
done = False
while not done:
    observation, _, done, _ = env.step(np.array([[0, 0, 0, 0, 0, 0, 0, 0]]))
    done = done[0]
    observations.append(observation)
    time.sleep(0.01)
# for _ in range(200):
#     observation, _, _ = env.envs[0].unwrapped._hysr.step([0, 0, 0, 0, 0, 0, 0, 0])
#     observations.append(observation)
#     time.sleep(0.01)
pass