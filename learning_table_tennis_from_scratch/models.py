import pathlib

from learning_table_tennis_from_scratch.hysr_one_ball_env import HysrOneBallEnv
from learning_table_tennis_from_scratch.rl_config import RLConfig
from learning_table_tennis_from_scratch.rl_config import OpenAIRLConfig

import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    # From https://stackoverflow.com/a/47626762 (CC BY-SA 4.0)
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class TensorEncoder(json.JSONEncoder):
    # From https://stackoverflow.com/a/47626762 (CC BY-SA 4.0)
    def default(self, obj):
        import torch

        if isinstance(obj, np.ndarray) or isinstance(obj, torch.Tensor):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def run_stable_baselines(
    reward_config_file,
    hysr_one_ball_config_file,
    rl_config_file,
    algorithm,
    log_episodes=False,
    seed=None,
):
    from stable_baselines3 import PPO
    from stable_baselines3 import SAC
    from stable_baselines3.common import logger
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.callbacks import CheckpointCallback

    if seed is not None:
        set_random_seed(seed)

    rl_config = RLConfig.from_json(rl_config_file, algorithm)

    tensorboard_logger = None
    checkpoint_callback = None
    if rl_config.log_path:
        tensorboard_logger = logger.configure(
            rl_config.log_path, ["stdout", "csv", "tensorboard"]
        )
        tensorboard_logger.set_level(logger.INFO)

        # Save a checkpoint every n_steps steps, or every 10000 steps if n_steps does
        # not exist (e.g. SAC)
        save_freq = getattr(rl_config, "n_steps", 10000)

        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=pathlib.Path(rl_config.log_path) / "checkpoints",
        )

    env_config = {
        "reward_config_file": reward_config_file,
        "hysr_one_ball_config_file": hysr_one_ball_config_file,
        "log_episodes": log_episodes,
        "logger": tensorboard_logger,
    }
    env = make_vec_env(HysrOneBallEnv, env_kwargs=env_config, seed=seed)

    model_type = {"ppo": PPO, "sac": SAC}

    if rl_config.load_path:
        print("loading policy from", rl_config.load_path)
        model = model_type[algorithm].load(
            rl_config.load_path, env, device="cpu", seed=seed
        )
    else:
        model = model_type[algorithm](
            "MlpPolicy",
            env,
            seed=seed,
            policy_kwargs={
                "net_arch": [rl_config.num_hidden] * rl_config.num_layers,
            },
            **rl_config.get_rl_params(),
        )

    # set custom logger, so we also get CSV output
    model.set_logger(tensorboard_logger)

    model.learn(total_timesteps=rl_config.num_timesteps, callback=checkpoint_callback)

    if rl_config.save_path:
        model.save(rl_config.save_path)

        # DEBUG: load the model again and compare
        reloaded = model_type[algorithm].load(rl_config.save_path, env)

        # some random observation to test the models
        test_obs = [
            -1.6734542846679688,
            0.956542432308197,
            0.3180750608444214,
            0.00026556311058811843,
            -2.6460986137390137,
            -3.60913348197937,
            -3.1900947093963623,
            0.003471289062872529,
            0.5342222452163696,
            0.28861111402511597,
            0.6971111297607422,
            0.6084444522857666,
            0.7795555591583252,
            0.4605555534362793,
            0.19538888335227966,
            0.5048888921737671,
            0.5625826716423035,
            0.5706161260604858,
            1.131399154663086,
            0.062083881348371506,
            -2.2659432888031006,
            2.0663530826568604,
        ]

        model_action, _states = model.predict(test_obs, deterministic=True)
        reloaded_action, _states = reloaded.predict(test_obs, deterministic=True)
        print("model action: {}".format(model_action))

        assert np.all(
            model_action == reloaded_action
        ), "ERROR: Models return different actions!"
        print("Models returned the same action.")

        # write parameters to json files for easier comparison
        # with open(rl_config.save_path + "_params.json", "w") as f:
        #     print("write file %s" % f.name)
        #     json.dump(model.get_parameters(), f, indent=2, cls=TensorEncoder)
        # with open(rl_config.save_path + "_params_reloaded.json", "w") as f:
        #     print("write file %s" % f.name)
        #     json.dump(reloaded.get_parameters(), f, indent=2, cls=TensorEncoder)


def eval_stable_baselines(
    reward_config_file,
    hysr_one_ball_config_file,
    rl_config_file,
    algorithm,
    log_episodes=False,
    seed=None,
):
    from stable_baselines3 import PPO
    from stable_baselines3 import SAC
    from stable_baselines3.common import logger
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.utils import set_random_seed

    if seed is not None:
        set_random_seed(seed)

    rl_config = RLConfig.from_json(rl_config_file, algorithm)

    tensorboard_logger = None
    if rl_config.log_path:
        tensorboard_logger = logger.configure(
            rl_config.log_path, ["stdout", "csv", "tensorboard"]
        )
        tensorboard_logger.set_level(logger.INFO)

    env_config = {
        "reward_config_file": reward_config_file,
        "hysr_one_ball_config_file": hysr_one_ball_config_file,
        "log_episodes": log_episodes,
        "logger": tensorboard_logger,
    }
    env = make_vec_env(HysrOneBallEnv, env_kwargs=env_config, seed=seed)

    model_type = {"ppo": PPO, "sac": SAC}

    assert rl_config.load_path

    print("loading policy from", rl_config.load_path)
    model = model_type[algorithm].load(
        rl_config.load_path, env, device="cpu", seed=seed
    )

    # set custom logger, so we also get CSV output
    model.set_logger(tensorboard_logger)

    # model.learn(total_timesteps=rl_config.num_timesteps, callback=checkpoint_callback)
    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        # env.render()
        if done:
            obs = env.reset()


def run_openai_baselines(
    reward_config_file,
    hysr_one_ball_config_file,
    rl_config_file,
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

    rl_config = OpenAIRLConfig.from_json(rl_config_file)

    if rl_config["log_tensorboard"]:
        tensorboard_logger = OpenaiLoggerWrapper(logger)
    else:
        tensorboard_logger = None
    del rl_config["log_tensorboard"]

    env_config = {
        "reward_config_file": reward_config_file,
        "hysr_one_ball_config_file": hysr_one_ball_config_file,
        "log_episodes": log_episodes,
        "logger": tensorboard_logger,
    }
    env = make_vec_env(HysrOneBallEnv, env_kwargs=env_config)

    total_timesteps = rl_config["num_timesteps"]
    del rl_config["num_timesteps"]
    save_path = rl_config["save_path"]
    del rl_config["save_path"]

    if rl_config["activation"] == "tf.tanh":
        rl_config["activation"] = tf.tanh

    # openai baselines only supported for ppo2 (legacy)
    alg = "ppo2"
    learn = get_alg_module_openai_baselines(alg).learn

    if model_file_path is None:
        print("total timesteps:", total_timesteps)
        model = learn(env=env, seed=seed, total_timesteps=total_timesteps, **rl_config)
        model.save("ppo2_openai_baselines_hysr_one_ball")

    else:
        rl_config["load_path"] = model_file_path
        model = learn(env=env, seed=seed, total_timesteps=0, **rl_config)

    if save_path:
        model.save(save_path)
        print("model saved to", save_path)

    return model, env


def replay_openai_baselines(
    model_file_path,
    nb_episodes,
    reward_config_file,
    hysr_one_ball_config_file,
    rl_config_file,
    log_episodes=False,
):

    model, env = run_openai_baselines(
        reward_config_file,
        hysr_one_ball_config_file,
        rl_config_file,
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
