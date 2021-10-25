from learning_table_tennis_from_scratch.hysr_one_ball_env import HysrOneBallEnv
from learning_table_tennis_from_scratch.ppo_config import PPOConfig
from learning_table_tennis_from_scratch.ppo_config import OpenAIPPOConfig


def run_stable_baselines(
    reward_config_file,
    hysr_one_ball_config_file,
    ppo_config_file,
    log_episodes=False,
    log_tensorboard=False,
):

    from stable_baselines.common.policies import MlpPolicy
    from stable_baselines.common import make_vec_env
    from stable_baselines import PPO2

    env_config = {
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
    reward_config_file,
    hysr_one_ball_config_file,
    ppo_config_file,
    log_episodes=False,
    log_tensorboard=False,
    model_file_path=None,
):
    import tensorflow as tf
    from stable_baselines.common import make_vec_env

    env_config = {
        "reward_config_file": reward_config_file,
        "hysr_one_ball_config_file": hysr_one_ball_config_file,
        "log_episodes": log_episodes,
        "log_tensorboard": log_tensorboard,
    }
    env = make_vec_env(HysrOneBallEnv, env_kwargs=env_config)

    ppo_config = OpenAIPPOConfig.from_json(ppo_config_file)
    total_timesteps = ppo_config["num_timesteps"]
    del ppo_config["num_timesteps"]
    save_path = ppo_config["save_path"]
    del ppo_config["save_path"]

    if ppo_config["activation"] == "tf.tanh":
        ppo_config["activation"] = tf.tanh

    alg = "ppo2"
    learn = get_alg_module_openai_baselines(alg).learn

    # seed = 123
    if model_file_path is None:
        print("total timesteps:", total_timesteps)
        model = learn(
            env=env,
            # seed=seed,
            total_timesteps=total_timesteps,
            **ppo_config
        )
        model.save("ppo2_openai_baselines_hysr_one_ball")

    else:
        ppo_config["load_path"] = model_file_path
        model = learn(
            env=env,
            # seed=seed,
            total_timesteps=0,
            **ppo_config
        )

    if save_path:
        model.save(save_path)
        print("model saved to", save_path)

    return model, env


def replay_openai_baselines(
    model_file_path,
    nb_episodes,
    reward_config_file,
    hysr_one_ball_config_file,
    ppo_config_file,
    log_episodes=False,
    log_tensorboard=False,
):

    model, env = run_openai_baselines(
        reward_config_file,
        hysr_one_ball_config_file,
        ppo_config_file,
        log_episodes=False,
        log_tensorboard=False,
        model_file_path=model_file_path,
    )

    observation = env.reset()

    for episode in range(nb_episodes):
        done = False
        while not done:
            actions = model.step(observation)[0][0]
            observation, _, done, __ = env.step([actions])

    env.close()


def get_alg_module_openai_baselines(alg, submodule=None):
    from importlib import import_module

    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        # (note: we used a modified version of baselines ppo2 which
        #        allows an update at each episode)
        import learning_table_tennis_from_scratch.modified_baselines_ppo2 as alg_module
    except ImportError:
        # then from rl_algs
        alg_module = import_module(".".join(["rl_" + "algs", alg, submodule]))
    return alg_module
