import pathlib
import os

from learning_table_tennis_from_scratch.hysr_one_ball_env import HysrOneBallEnv
from learning_table_tennis_from_scratch.hysr_many_ball_env import HysrManyBallEnv
from learning_table_tennis_from_scratch.hysr_goal_env import HysrGoalEnv
from learning_table_tennis_from_scratch.rl_config import RLConfig
from learning_table_tennis_from_scratch.rl_config import OpenAIRLConfig
from learning_table_tennis_from_scratch.hysr_one_ball import HysrOneBallConfig
from learning_table_tennis_from_scratch.layernorm_extractor import LayerNormFeaturesExtractor
import torch as th
from rllte.xplore.reward import (
    Disagreement, E3B, ICM, NGU,
    PseudoCounts, RE3, RIDE, RND
)
from learning_table_tennis_from_scratch.rl_explore_on_policy import RLeXploreWithOnPolicyRL


import gymnasium as gym
import torch.nn as nn

def run_stable_baselines(
    reward_config_file,
    hysr_one_ball_config_file,
    rl_config_file,
    env_type,
    algorithm,
    log_episodes=False,
    seed=None,
):
    from stable_baselines3 import PPO
    from stable_baselines3 import SAC
    from stable_baselines3 import HerReplayBuffer
    from stable_baselines3.common import logger
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.evaluation import evaluate_policy

    if seed is not None:
        set_random_seed(seed)

    rl_config = RLConfig.from_json(rl_config_file, algorithm)
    hysr_config = HysrOneBallConfig.from_json(hysr_one_ball_config_file)


    tensorboard_logger = None
    checkpoint_callback = None
    if rl_config.log_path:
        tensorboard_logger = logger.configure(
            rl_config.log_path, ["stdout", "csv", "tensorboard"]
        )
        tensorboard_logger.set_level(logger.INFO)

        # Save a checkpoint every checkpoint_save_freq steps, or every 10000 steps if checkpoint_save_freq does
        # not exist
        save_freq = getattr(rl_config, "checkpoint_save_freq", 10000)

        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=pathlib.Path(rl_config.log_path) / "checkpoints",
        )

    

    if env_type in [HysrOneBallEnv, HysrGoalEnv, HysrManyBallEnv]:
        env_config = {
        "reward_config_file": reward_config_file,
        "hysr_one_ball_config_file": hysr_one_ball_config_file,
        "log_episodes": log_episodes,
        "logger": tensorboard_logger,
    }
    env = make_vec_env(env_type, env_kwargs=env_config, seed=seed)

    model_type = {"ppo": PPO, "sac": SAC, "sac_her": SAC, "sac_hsm": SAC, "sac_hsm_her": SAC}

    if rl_config.load_path:
        print("loading policy from", rl_config.load_path)
        # NOTE: It's important to set the seed when loading the model.  Otherwise the
        # RNG state will be restored from the loaded model, resulting in all runs to
        # behave the same.
        model = model_type[algorithm].load(rl_config.load_path, env, seed=seed)
        continue_training = True
    else:
        if algorithm in ["ppo", "sac"]:
            if rl_config.use_layer_norm:
                # Use custom features extractor with LayerNorm
                policy_kwargs = {
                    "features_extractor_class": LayerNormFeaturesExtractor,
                    "features_extractor_kwargs": {
                        "net_arch": [rl_config.num_hidden] * rl_config.num_layers,
                        "use_layer_norm": rl_config.use_layer_norm,
                        "hidden_layers_bias": rl_config.hidden_layers_bias,
                        "activation_fn": nn.ReLU,
                    },
                    "net_arch": []  # Empty since features extractor handles the network
                }
            else:
                assert rl_config.hidden_layers_bias, "hidden layers bias must be set to True for standard architecture"
                # Standard MlpPolicy
                policy_kwargs = {
                    "net_arch": [rl_config.num_hidden] * rl_config.num_layers
                }
            
            model = model_type[algorithm](
                    "MlpPolicy",
                    env,
                    seed=seed,
                    policy_kwargs=policy_kwargs,
                    **rl_config.get_rl_params(),
                )
            continue_training = False
        elif algorithm in ["sac_her"]:
            model = model_type[algorithm](
                "MultiInputPolicy",
                env,
                replay_buffer_class=HerReplayBuffer,
                seed=seed,
                policy_kwargs={
                    "net_arch": [rl_config.num_hidden] * rl_config.num_layers
                },
                replay_buffer_kwargs=dict(
                    n_sampled_goal = rl_config.n_sampled_goal,
                    goal_selection_strategy = rl_config.goal_selection_strategy,
                    online_sampling = rl_config.online_sampling,
                    max_episode_length = 200
                ),
                **rl_config.get_rl_params(),
            )
        elif algorithm in ["sac_hsm"]:
            model = model_type[algorithm](
                "MultiInputPolicy",
                env,
                replay_buffer_class=HerReplayBuffer,
                seed=seed,
                policy_kwargs={
                    "net_arch": [rl_config.num_hidden] * rl_config.num_layers
                },
                replay_buffer_kwargs=dict(
                    n_sampled_hindsight_states = rl_config.n_sampled_hindsight_states,
                    hindsight_state_selection_strategy = rl_config.hindsight_state_selection_strategy,
                    hindsight_state_selection_strategy_horizon = rl_config.hindsight_state_selection_strategy_horizon,
                    HSM_shape = rl_config.HSM_shape,
                    HSM_goal_env = rl_config.HSM_goal_env,
                    HSM_n_traj_freq = rl_config.HSM_n_traj_freq,
                    HSM_min_criterion = rl_config.HSM_min_criterion,
                    n_sampled_hindsight_states_change_per_step = rl_config.n_sampled_hindsight_states_change_rel / rl_config.num_timesteps * rl_config.n_sampled_hindsight_states,
                    HSM_criterion_change_per_step = rl_config.HSM_criterion_change / rl_config.num_timesteps,
                    HSM_use_likelihood_ratio = rl_config.HSM_use_likelihood_ratio,
                    HSM_likelihood_ratio_cutoff = rl_config.HSM_likelihood_ratio_cutoff,
                    prioritized_replay_baseline = rl_config.prioritized_replay_baseline,
                    online_sampling = False,
                    apply_HSM = True,
                    apply_HER = False,
                    max_episode_length = 200
                ),
                **rl_config.get_rl_params(),
            )
        elif algorithm in ["sac_hsm_her"]:
            model = model_type[algorithm](
                "MultiInputPolicy",
                env,
                replay_buffer_class=HerReplayBuffer,
                seed=seed,
                policy_kwargs={
                    "net_arch": [rl_config.num_hidden] * rl_config.num_layers,
                },
                replay_buffer_kwargs=dict(
                    n_sampled_hindsight_states = rl_config.n_sampled_hindsight_states,
                    hindsight_state_selection_strategy = rl_config.hindsight_state_selection_strategy,
                    hindsight_state_selection_strategy_horizon = rl_config.hindsight_state_selection_strategy_horizon,
                    HSM_shape = rl_config.HSM_shape,
                    HSM_goal_env = rl_config.HSM_goal_env,
                    HSM_n_traj_freq = rl_config.HSM_n_traj_freq,
                    HSM_min_criterion = rl_config.HSM_min_criterion,
                    n_sampled_hindsight_states_change_per_step = rl_config.n_sampled_hindsight_states_change_rel / rl_config.num_timesteps * rl_config.n_sampled_hindsight_states,
                    HSM_criterion_change_per_step = rl_config.HSM_criterion_change / rl_config.num_timesteps,
                    HSM_use_likelihood_ratio = rl_config.HSM_use_likelihood_ratio,
                    HSM_likelihood_ratio_cutoff = rl_config.HSM_likelihood_ratio_cutoff,
                    prioritized_replay_baseline = rl_config.prioritized_replay_baseline,
                    online_sampling = True,
                    apply_HSM = True,
                    apply_HER = True,
                    max_episode_length = 200,
                    n_sampled_goal = rl_config.n_sampled_goal,
                    goal_selection_strategy = rl_config.goal_selection_strategy
                ),
                **rl_config.get_rl_params(),
            )
        else:
            raise ValueError(f"Environment {env_type} not supported!")

    # set custom logger, so we also get CSV output
    model.set_logger(tensorboard_logger)
    
    if rl_config.load_path:
        del model
        print("loading policy from", rl_config.load_path)
        model = model_type[algorithm].load(rl_config.load_path, env, seed=seed)
        if rl_config.save_and_load_buffer:
            print("loading replay buffer from", rl_config.load_path)
            model.load_replay_buffer(rl_config.load_path+"_buf")
            if rl_config.delete_buffer_file_after_loading:
                os.remove(rl_config.load_path+"_buf.pkl")
                print("buffer file deleted")
        continue_training = True
    else:
        continue_training = False

    # set custom logger, so we also get CSV output
    model.set_logger(tensorboard_logger)

    # play
    # obs = env.reset()
    # print("playing...")
    # i=0
    # while True:
    #     action, _states = model.predict(obs, deterministic=False)
    #     obs, reward, done, info = env.step(action)
    #     if done:
    #         obs = env.reset()
    #         i+=1

    if rl_config.rl_explore:
        # Mapping of reward class names to their actual classes
        reward_classes = {
            "Disagreement": Disagreement,
            "E3B": E3B,
            "ICM": ICM,
            "NGU": NGU,
            "PseudoCounts": PseudoCounts,
            "RE3": RE3,
            "RIDE": RIDE,
            "RND": RND
        }

        
        # Get the reward class from config
        reward_class_name = getattr(rl_config, "rl_explore_reward_class", None)

        if reward_class_name not in reward_classes:
            raise ValueError(f"Unknown reward class '{reward_class_name}'")

        RewardClass = reward_classes[reward_class_name]
        device = 'cpu'
        vec_env = model.get_env()
        
        # Build kwargs for reward class with common parameters
        reward_kwargs = {
            'envs': vec_env,
            'device': device,
            'beta': getattr(rl_config, 'rl_explore_beta', 1.0),
            'rwd_norm_type': 'rms' if getattr(rl_config, 'rl_explore_reward_norm', True) else 'none',
            'obs_norm_type': 'rms' if getattr(rl_config, 'rl_explore_obs_norm', True) else 'none',
        }
        
        # Add update_proportion for algorithms that support it (all except RE3)
        if reward_class_name != "RE3":
            reward_kwargs['update_proportion'] = getattr(rl_config, 'rl_explore_update_proportion', 1.0)
        
        # Add method-specific parameters based on reward class
        method_idx = getattr(rl_config, 'rl_explore_method_specific_parameter_index', 0)
        
        # Methods with k parameter - different defaults for different algorithms
        if reward_class_name == "RE3":
            # RE3 has default k=5
            k_values = [5, 10]
            reward_kwargs['k'] = k_values[method_idx]
        elif reward_class_name in ["PseudoCounts", "NGU", "RIDE"]:
            # PseudoCounts, NGU, RIDE have default k=10
            k_values = [10, 5]
            reward_kwargs['k'] = k_values[method_idx]
        
        # Method with ensemble_size parameter: Disagreement (default=4)
        elif reward_class_name == "Disagreement":
            ensemble_sizes = [4, 5]
            reward_kwargs['ensemble_size'] = ensemble_sizes[method_idx]
        
        # Methods with latent_dim parameter - different defaults
        elif reward_class_name in ["ICM", "RND", "E3B"]:
            # E3B, ICM, RND have default latent_dim=128
            latent_dims = [128, 64]
            reward_kwargs['latent_dim'] = latent_dims[method_idx]
        elif reward_class_name == "RIDE":
            # RIDE has both k and latent_dim parameters
            k_values = [10, 5]
            latent_dim = 128
            reward_kwargs['k'] = k_values[method_idx]
            reward_kwargs['latent_dim'] = latent_dim

        print(reward_kwargs)

        # Create the reward instance
        irs = RewardClass(**reward_kwargs)

        rl_explore_callback = RLeXploreWithOnPolicyRL(irs=irs, verbose=1)


    if rl_config.num_timesteps > 0:
        callbacks = []
        if checkpoint_callback is not None:
            callbacks.append(checkpoint_callback)
        if rl_config.rl_explore:
            callbacks.append(rl_explore_callback)

        model.learn(
        total_timesteps=rl_config.num_timesteps,
        callback=callbacks if callbacks else None,
        reset_num_timesteps=not continue_training,
        log_interval=1,
    )
        
        if rl_config.save_path:
            model.save(rl_config.save_path)
            print("policy saved to", rl_config.save_path)
            if rl_config.save_and_load_buffer:
                model.save_replay_buffer(rl_config.save_path+"_buf")
                print("buffer saved to", rl_config.save_path+"_buf")

    if rl_config.eval:
        print("-- Evaluating policy --")
        if env.get_attr('_log_episodes')[0]:
            base_env = env.envs[0].unwrapped
            base_env._save_folder_traj = base_env._save_folder_traj + "eval/"
            if not os.path.exists(base_env._save_folder_traj):
                os.makedirs(base_env._save_folder_traj)
            print("Saving evaluation episodes to", base_env._save_folder_traj)
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=rl_config.eval_episodes)
        print("-- Evaluation finished --")
        print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f} (std)")

    env.close()


def run_openai_baselines(
    reward_config_file,
    hysr_one_ball_config_file,
    rl_config_file,
    env,
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
    env = make_vec_env(env, env_kwargs=env_config)

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
