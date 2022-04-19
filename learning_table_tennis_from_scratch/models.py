import pathlib

from learning_table_tennis_from_scratch.hysr_one_ball_env import HysrOneBallEnv
from learning_table_tennis_from_scratch.ppo_config import PPOConfig
from learning_table_tennis_from_scratch.ppo_config import OpenAIPPOConfig


def run_stable_baselines(
    reward_config_file,
    hysr_one_ball_config_file,
    ppo_config_file,
    log_episodes=False,
    seed=None,
):
    from stable_baselines3 import PPO
    from stable_baselines3.common import logger
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import CheckpointCallback

    ppo_config = PPOConfig.from_json(ppo_config_file)

    tensorboard_logger = None
    checkpoint_callback = None
    if ppo_config.log_path:
        tensorboard_logger = logger.configure(
            ppo_config.log_path, ["stdout", "csv", "tensorboard"]
        )
        tensorboard_logger.set_level(logger.INFO)

        # Save a checkpoint every n_steps steps
        checkpoint_callback = CheckpointCallback(
            save_freq=ppo_config.n_steps,
            save_path=pathlib.Path(ppo_config.log_path) / "checkpoints",
        )

    env_config = {
        "reward_config_file": reward_config_file,
        "hysr_one_ball_config_file": hysr_one_ball_config_file,
        "log_episodes": log_episodes,
        "logger": tensorboard_logger,
    }
    env = make_vec_env(HysrOneBallEnv, env_kwargs=env_config)

    model = PPO(
        "MlpPolicy",
        env,
        seed=seed,
        policy_kwargs={
            "net_arch": [ppo_config.num_hidden] * ppo_config.num_layers,
        },
        **ppo_config.get_ppo_params(),
    )
    # set custom logger, so we also get CSV output
    model.set_logger(tensorboard_logger)

    model.learn(total_timesteps=ppo_config.num_timesteps, callback=checkpoint_callback)

    if ppo_config.save_path:
        model.save(ppo_config.save_path)


def run_openai_baselines(
    reward_config_file,
    hysr_one_ball_config_file,
    ppo_config_file,
    log_episodes=False,
    model_file_path=None,
    seed=None,
):
    import warnings

    warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

    import tensorflow as tf

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    from baselines import logger
    from stable_baselines.common import make_vec_env

    class OpenaiLoggerWrapper:
        """Wrapper for baselines.logger so it has same methods as stable_baselines3."""

        def __init__(self, logger):
            self.logger = logger

        def record(self, key, value):
            self.logger.logkv(key, value)

        def dump(self):
            self.logger.dumpkvs()

    ppo_config = OpenAIPPOConfig.from_json(ppo_config_file)

    if ppo_config["log_tensorboard"]:
        tensorboard_logger = OpenaiLoggerWrapper(logger)
    else:
        tensorboard_logger = None
    del ppo_config["log_tensorboard"]

    env_config = {
        "reward_config_file": reward_config_file,
        "hysr_one_ball_config_file": hysr_one_ball_config_file,
        "log_episodes": log_episodes,
        "logger": tensorboard_logger,
    }
    env = make_vec_env(HysrOneBallEnv, env_kwargs=env_config)

    total_timesteps = ppo_config["num_timesteps"]
    del ppo_config["num_timesteps"]
    save_path = ppo_config["save_path"]
    del ppo_config["save_path"]

    if ppo_config["activation"] == "tf.tanh":
        ppo_config["activation"] = tf.tanh

    alg = "ppo2"
    learn = get_alg_module_openai_baselines(alg).learn

    if model_file_path is None:
        print("total timesteps:", total_timesteps)
        model = learn(env=env, seed=seed, total_timesteps=total_timesteps, **ppo_config)
        model.save("ppo2_openai_baselines_hysr_one_ball")

    else:
        ppo_config["load_path"] = model_file_path
        model = learn(env=env, seed=seed, total_timesteps=0, **ppo_config)

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
):

    model, env = run_openai_baselines(
        reward_config_file,
        hysr_one_ball_config_file,
        ppo_config_file,
        log_episodes=False,
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
