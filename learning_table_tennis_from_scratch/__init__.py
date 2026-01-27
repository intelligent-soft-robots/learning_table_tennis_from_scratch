from learning_table_tennis_from_scratch.compat import gym

gym.envs.registration.register(
    id="hysroneball-v0",
    entry_point="learning_table_tennis_from_scratch.hysr_one_ball_env:HysrOneBallEnv",
)
