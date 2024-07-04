import time

from learning_table_tennis_from_scratch.hysr_one_ball import HysrOneBallConfig, HysrOneBall

from learning_table_tennis_from_scratch.rewards import JsonReward

hysr_one_ball_config = HysrOneBallConfig.from_json("../config/hysr_config.json")
reward_function = JsonReward.get("../config/reward_config.json")

hysr = HysrOneBall(hysr_one_ball_config, reward_function)
observations = []
hysr.reset()
for _ in range(1000):
    observation, _, _ = hysr.step([0, 0, 0, 0, 0, 0, 0, 0])
    observations.append(observation)
    time.sleep(0.01)
pass