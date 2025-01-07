import json
import os
import sys
import typing


ConfigDict = typing.Dict[str, typing.Any]


class RLConfig:
    """Configuration for stable_baselines3 RL."""

    # These parameters are directly passed to the stable_baselines3 model (e.g. stable_baselines3.PPO), see the
    # documentation there for their meaning.
    _algo_params_ppo = (
        "gamma",
        "n_steps",
        "ent_coef",
        "learning_rate",
        "clip_range",
        "clip_range_vf",
        "vf_coef",
        "max_grad_norm",
        "gae_lambda",
        "batch_size",
        "n_epochs",
    )

    _algo_params_sac = (
        "gamma",
        "ent_coef",
        "learning_rate",
        "batch_size",
        "gradient_steps",
        "train_freq",
        "train_freq_unit",
        "buffer_size",
        "learning_starts",
    )

    _params_her = (
        "n_sampled_goal",
        "goal_selection_strategy",
        "online_sampling"
    )

    _params_hsm = (
        "n_sampled_hindsight_states",
        "hindsight_state_selection_strategy",
        "hindsight_state_selection_strategy_horizon",
        "HSM_shape",
        "HSM_goal_env",
        "HSM_n_traj_freq",
        "HSM_min_criterion",
        "n_sampled_hindsight_states_change_rel",
        "HSM_criterion_change",
        "HSM_use_likelihood_ratio",
        "HSM_likelihood_ratio_cutoff",
        "prioritized_replay_baseline",
    )


    # Additional parameters
    _additional_params = (
        "num_timesteps",  # 'total_timesteps' passed to RL.learn()
        "load_path",  # If set the model is saved to the given path.
        "save_path",  # If set the model is saved to the given path.
        "save_and_load_buffer", #If set true, the buffer is saved/loaded together with the model.
        "delete_buffer_file_after_loading", #If set true, the buffer file is deleted after loading it.
        "load_path", #If set a pretrained model is loaded from the given path.
        "num_layers",  # Number of layers in the network
        "num_hidden",  # Size of each layer (all layers have same size)
        "log_path",  # Destination for checkpoints and log files
        "eval", # If set, the model is evaluated
        "eval_episodes", # Number of episodes to evaluate the model
    )

    _params_ppo = _algo_params_ppo + _additional_params
    _params_sac = _algo_params_sac + _additional_params
    _params_sac_her = _algo_params_sac + _params_her + _additional_params
    _params_sac_hsm = _algo_params_sac + _params_hsm + _additional_params
    _params_sac_hsm_her = _algo_params_sac + _params_hsm + _params_her + _additional_params


    def __init__(self, algorithm):
        if algorithm == "ppo":
            self._params = self._params_ppo
            self._algo_params = self._algo_params_ppo
        elif algorithm == "sac":
            self._params = self._params_sac
            self._algo_params = self._algo_params_sac
        elif algorithm == "sac_her":
            self._params = self._params_sac_her
            self._algo_params = self._algo_params_sac
        elif algorithm == "sac_hsm":
            self._params = self._params_sac_hsm
            self._algo_params = self._algo_params_sac
        elif algorithm == "sac_hsm_her":
            self._params = self._params_sac_hsm_her
            self._algo_params = self._algo_params_sac
        for s in self._params:
            setattr(self, s, None)

    def get(self) -> ConfigDict:
        """Get dictionary with all parameters."""
        return {attr: getattr(self, attr) for attr in self._params}

    def get_rl_params(self) -> ConfigDict:
        """Get dictionary with only the direct RL parameters.

        This includes only the parameters that can be passed as keyword arguments to
        stable_baselines3.RL.
        """
        # replace value of train_freq with (train_freq, train_freq_unit) and remove train_freq_unit key
        if "train_freq_unit" in self._algo_params:
            setattr(self, "train_freq", (getattr(self, "train_freq"), getattr(self, "train_freq_unit")))
            self._algo_params = [param for param in self._algo_params if param!="train_freq_unit"]
        return {attr: getattr(self, attr) for attr in self._algo_params}

    @classmethod
    def from_dict(cls, d: ConfigDict, algorithm) -> "RLConfig":
        instance = cls(algorithm)
        for s in instance._params:
            setattr(instance, s, d[s])
        return instance

    @classmethod
    def from_json(
        cls, jsonpath: typing.Union[str, os.PathLike], algorithm
    ) -> "RLConfig":
        if not os.path.isfile(jsonpath):
            raise FileNotFoundError(
                "failed to find RL configuration file: {}".format(jsonpath)
            )
        try:
            with open(jsonpath) as f:
                conf = json.load(f)
        except Exception as e:
            raise ValueError(
                "failed to parse RL json configuration file {}: {}".format(jsonpath, e)
            )

        instance = cls(algorithm)
        for s in instance._params:
            try:
                setattr(instance, s, conf[s])
            except Exception:
                raise ValueError(
                    "failed to find the attribute {} " "in {}".format(s, jsonpath)
                )
        return instance

    # FIXME Is this used anywhere?
    @staticmethod
    def default_path():
        return os.path.join(
            sys.prefix, "learning_table_tennis_from_scratch_config", "ppo_default.json"
        )


class OpenAIRLConfig:
    _params = (
        "gamma",
        "num_timesteps",
        "ent_coef",
        "lr",
        "cliprange",
        "vf_coef",
        "max_grad_norm",
        "lam",
        "nminibatches",
        "noptepochs",
        "network",
        "num_layers",
        "num_hidden",
        "activation",
        "nsteps",
        "save_path",
        "load_path",
        "log_interval",
        "save_interval",
        "log_tensorboard",
    )  # , "seed")

    def __init__(self):
        for s in self._params:
            setattr(self, s, None)

    def get(self):
        return {attr: getattr(self, attr) for attr in self._params}

    @classmethod
    def from_dict(cls, d):
        instance = cls()
        for s in instance._params:
            setattr(instance, s, d[s])
        return instance

    @classmethod
    def from_json(cls, jsonpath):
        if not os.path.isfile(jsonpath):
            raise FileNotFoundError(
                "failed to find reward configuration file: {}".format(jsonpath)
            )
        try:
            with open(jsonpath) as f:
                conf = json.load(f)
        except Exception as e:
            raise ValueError(
                "failed to parse reward json configuration file {}: {}".format(
                    jsonpath, e
                )
            )
        instance = cls()
        for s in cls._params:
            try:
                setattr(instance, s, conf[s])
            except Exception:
                raise ValueError(
                    "failed to find the attribute {} " "in {}".format(s, jsonpath)
                )
        return instance.get()

    @staticmethod
    def default_path():
        return os.path.join(
            "/home/sguist/Workspaces_local/isr/learning_table_tennis_from_scratch/config",
            "openai_ppo_default.json",
        )
