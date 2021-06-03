from learning_table_tennis_from_scratch.hysr_one_ball_env import HysrOneBallEnv
from learning_table_tennis_from_scratch.ppo_config import PPOConfig


def run_stable_baselines(
    pam_config_file,
    reward_config_file,
    hysr_one_ball_config_file,
    ppo_config_file,
    tensorboard_log="/tmp/ppo2",
):

    import gym
    from stable_baselines.common.policies import MlpPolicy
    from stable_baselines.common import make_vec_env
    from stable_baselines import PPO2
    from stable_baselines.common.env_checker import check_env

    env_config = {
        "pam_config_file": pam_config_file,
        "reward_config_file": reward_config_file,
        "hysr_one_ball_config_file": hysr_one_ball_config_file,
    }
    env = make_vec_env(HysrOneBallEnv, env_kwargs=env_config)

    ppo_config = PPOConfig.from_json(ppo_config_file)
    model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log="/tmp/ppo2", **ppo_config)
    model.learn(total_timesteps=1000000)
    model.save("ppo2_hysr_one_ball")
