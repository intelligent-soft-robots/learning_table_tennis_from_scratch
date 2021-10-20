from learning_table_tennis_from_scratch.hysr_one_ball_env import HysrOneBallEnv
from learning_table_tennis_from_scratch.ppo_config import PPOConfig
from learning_table_tennis_from_scratch.ppo_config import OpenAIPPOConfig


def run_stable_baselines(
    env,
    pam_config_file,
    reward_config_file,
    hysr_one_ball_config_file,
    ppo_config_file,
    log_episodes=False,
    log_tensorboard=False,
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
        "log_episodes": log_episodes,
        "log_tensorboard": log_tensorboard,
    }
    env = make_vec_env(HysrOneBallEnv, env_kwargs=env_config)

    ppo_config = PPOConfig.from_json(ppo_config_file)
    if log_tensorboard:
        model = PPO2(
            MlpPolicy, env, verbose=1, log_tensorboard=log_tensorboard, **ppo_config
        )
    else:
        model = PPO2(MlpPolicy, env, verbose=1, **ppo_config)
    model.learn(total_timesteps=1000000)
    model.save("ppo2_hysr_one_ball")


def run_openai_baselines(
    env,
    pam_config_file,
    reward_config_file,
    hysr_one_ball_config_file,
    ppo_config_file,
    log_episodes=False,
    log_tensorboard=False,
):

    import baselines
    from baselines.common.vec_env import VecFrameStack, VecNormalize, VecEnv
    from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    from baselines.common.cmd_util import (
        common_arg_parser,
        parse_unknown_args,
        make_env,
    )
    from baselines.common.tf_util import get_session
    from baselines import logger
    import tensorflow as tf

    from stable_baselines.common import make_vec_env

    env_config = {
        "pam_config_file": pam_config_file,
        "reward_config_file": reward_config_file,
        "hysr_one_ball_config_file": hysr_one_ball_config_file,
        "log_episodes": log_episodes,
        "log_tensorboard": log_tensorboard,
    }
    env = make_vec_env(env, env_kwargs=env_config)

    ppo_config = OpenAIPPOConfig.from_json(ppo_config_file)
    total_timesteps = ppo_config["num_timesteps"]
    del ppo_config["num_timesteps"]

    if ppo_config["activation"] == "tf.tanh":
        ppo_config["activation"] = tf.tanh

    alg = "ppo2"
    learn = get_alg_module_openai_baselines(alg).learn
    # seed = 123
    print("total timesteps:", total_timesteps)
    model = learn(
        env=env,
        # seed=seed,
        total_timesteps=total_timesteps,
        **ppo_config
    )
    model.save("ppo2_openai_baselines_hysr_one_ball")
    return model, env


def get_alg_module_openai_baselines(alg, submodule=None):
    from importlib import import_module

    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        alg_module = import_module(".".join(["baselines", alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module(".".join(["rl_" + "algs", alg, submodule]))
    return alg_module
