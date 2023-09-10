import json
import os
import sys
import typing

from typing import Callable

ConfigDict = typing.Dict[str, typing.Any]


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

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
        "buffer_size",
        "learning_starts",
    )

    # Additional parameters
    _additional_params = (
        "num_timesteps",  # 'total_timesteps' passed to RL.learn()
        "load_path",  # If set the model is saved to the given path.
        "save_path",  # If set the model is saved to the given path.
        "num_layers",  # Number of layers in the network
        "num_hidden",  # Size of each layer (all layers have same size)
        "log_path",  # Destination for checkpoints and log files
    )

    _params_ppo = _algo_params_ppo + _additional_params
    _params_sac = _algo_params_sac + _additional_params

    def __init__(self, algorithm):
        if algorithm == "ppo":
            print("ppo params...")
            self._params = self._params_ppo
            self._algo_params = self._algo_params_ppo
            
        elif algorithm == "sac":
            self._params = self._params_sac
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
        print("rl_params:", {attr: getattr(self, attr) for attr in self._algo_params})
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

        if "learning_rate_schedule" in conf:
            if conf["learning_rate_schedule"]:
                conf["learning_rate"] = linear_schedule(conf["learning_rate"])
                print("using linear schedule for learning rate")
                print(conf["learning_rate"])
            else:
                print("not using linear schedule for learning rate")
            del conf["learning_rate_schedule"]
        else:
            print("schedule not found")
            print(conf)

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
