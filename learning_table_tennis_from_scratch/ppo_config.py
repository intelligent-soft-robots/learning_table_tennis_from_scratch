import sys
import os
import json


class PPOConfig:

    __slots__ = (
        "gamma",
        "n_steps",
        "ent_coef",
        "learning_rate",
        "cliprange",
        "cliprange_vf",
        "vf_coef",
        "max_grad_norm",
        "lam",
        "nminibatches",
        "noptepochs",
    )

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
        "load_path"
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
