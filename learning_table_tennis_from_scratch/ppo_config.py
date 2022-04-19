import json
import os
import sys
import typing


ConfigDict = typing.Dict[str, typing.Any]


class PPOConfig:
    """Configuration for stable_baselines3 PPO."""

    # These parameters are directly passed to stable_baselines3.PPO, see the
    # documentation there for their meaning.
    _ppo_params = (
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

    # Additional parameters
    _additional_params = (
        "num_timesteps",  # 'total_timesteps' passed to PPO.learn()
        "save_path",  # If set the model is saved to the given path.
        "num_layers",  # Number of layers in the network
        "num_hidden",  # Size of each layer (all layers have same size)
        "log_path",  # Destination for checkpoints and log files
    )

    __slots__ = _ppo_params + _additional_params

    def __init__(self):
        for s in self.__slots__:
            setattr(self, s, None)

    def get(self) -> ConfigDict:
        """Get dictionary with all parameters."""
        return {attr: getattr(self, attr) for attr in self.__slots__}

    def get_ppo_params(self) -> ConfigDict:
        """Get dictionary with only the direct PPO parameters.

        This includes only the parameters that can be passed as keyword arguments to
        stable_baselines3.PPO.
        """
        return {attr: getattr(self, attr) for attr in self._ppo_params}

    @classmethod
    def from_dict(cls, d: ConfigDict) -> "PPOConfig":
        instance = cls()
        for s in instance.__slots__:
            setattr(instance, s, d[s])
        return instance

    @classmethod
    def from_json(cls, jsonpath: typing.Union[str, os.PathLike]) -> "PPOConfig":
        if not os.path.isfile(jsonpath):
            raise FileNotFoundError(
                "failed to find PPO configuration file: {}".format(jsonpath)
            )
        try:
            with open(jsonpath) as f:
                conf = json.load(f)
        except Exception as e:
            raise ValueError(
                "failed to parse PPO json configuration file {}: {}".format(jsonpath, e)
            )
        instance = cls()
        for s in cls.__slots__:
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


class OpenAIPPOConfig:
    __slots__ = (
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
        for s in self.__slots__:
            setattr(self, s, None)

    def get(self):
        return {attr: getattr(self, attr) for attr in self.__slots__}

    @classmethod
    def from_dict(cls, d):
        instance = cls()
        for s in instance.__slots__:
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
        for s in cls.__slots__:
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
