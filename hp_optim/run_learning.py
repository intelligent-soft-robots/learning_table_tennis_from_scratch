#!/usr/bin/env python3
"""Run learning with cluster_utils

This is a wrapper script around running hysr_start_robots and hysr_one_ball_ppo
that is intended to be used by cluster_utils (for hyperparameter optimisation).

See the accompanying README on how cluster_utils needs to be configured for
this to work.
"""
import csv
import json
import os
import pathlib
import subprocess
import sys
import time
import typing

import cluster


def prepare_config_file(
    name: str,
    template_file: typing.Union[str, os.PathLike],
    parameter_updates: dict,
    destination_directory: pathlib.PurePath,
) -> None:
    """Create config file based on the given template and parameter updates.

    The given template is expected to be a valid config file in JSON format.
    The *paramter_updates* dict can be used to overwrite some of the default
    values with custom ones.

    The resulting config will be written to "destination_directory/name.json".

    Args:
        name:  Name of the config file.
        template_file:  Path to the template file.
        parameter_updates:  Dictionary with parameters that are to be
            overwritten in the template file.
        destination_directory:  Directory to which the resulting configuration
            file is written.
    """
    with open(template_file, "r") as f:
        config = json.load(f)

    config.update(parameter_updates)

    destination_file = destination_directory / "{}.json".format(name)
    with open(destination_file, "w") as f:
        json.dump(config, f)


def setup_config(working_dir: pathlib.Path, params: dict):
    """Set up config files based on templates and parameters.

    The *params* dictionary is expected to have an entry "config_templates" which
    contains the path to a JSON file pointing to the different config template files.
    Expected entries in that file are:

    - "reward_config"
    - "hysr_config"
    - "pam_config"
    - "ppo_config"
    - "ppo_common_config"

    Further *params* may contain contain an entry for any of the template files listed
    above, which again contains a dictionary with parameter updates for that file.
    These parameters are overwritten in the template files.

    Example for *params*:

    .. code-block:: Python

        {
            "config_templates": "./templates/base_config.json",
            "ppo_config": {
                "num_timesteps": 1000000,
                "num_layers": 2
            }
        }

    Args:
        working_dir:  Directory in which the config setup is created.
        params:  Dictionary of parameters (see above).
    """
    with open(params["config_templates"], "r") as f:
        config_file_templates = json.load(f)

    config_dir = working_dir / "config"
    config_dir.mkdir(exist_ok=True, parents=True)

    # Set ppo_config.save_path to working_dir (unless a different value is
    # already explicitly set in params).
    # Need to convert params to normal dict, so it becomes mutable
    params = dict(params)
    if "ppo_config" not in params:
        params["ppo_config"] = {}
    if "save_path" not in params["ppo_config"]:
        params["ppo_config"] = dict(params["ppo_config"])
        params["ppo_config"]["save_path"] = os.fspath(working_dir / "model")

    base_config = {
        "reward_config": "./config/reward_config.json",
        "hysr_config": "./config/hysr_config.json",
        "pam_config": "./config/pam_config.json",
        "ppo_config": "./config/ppo_config.json",
        "ppo_common_config": "./config/ppo_common_config.json",
    }
    with open(working_dir / "config.json", "w") as f:
        json.dump(base_config, f)

    base_config_path = pathlib.PurePath(params["config_templates"]).parent
    for name in base_config.keys():
        template_path = base_config_path / config_file_templates[name]
        prepare_config_file(name, template_path, params.get(name, {}), config_dir)


def read_reward_from_log(log_dir: pathlib.PurePath):
    """Extract 'eprewmean' from the log.

    Args:
        log_dir:  Directory in which the logs of the learning are stored.

    Returns:
        "eprewmean" from the log.
    """
    eprewmean = None
    with open(log_dir / "progress.csv", "r") as f:
        reader = csv.DictReader(f)
        # go through all lines, get the last eprewmean value
        for line in reader:
            if line["eprewmean"]:
                eprewmean = float(line["eprewmean"])

    if eprewmean is None:
        raise RuntimeError("Failed to read eprewmean from log file.")

    return eprewmean


def main():
    # overwrite some values from the config templates to disable any graphical
    # interfaces
    default_params = {
        "config": {
            "hysr_config": {
                "graphics": False,
                "xterms": False,
            },
        }
    }

    # get parameters
    params = cluster.update_params_from_cmdline(default_params=default_params)

    print("Params:")
    print(params)
    print("---")

    working_dir = pathlib.Path(params.model_dir)

    # prepare environment
    env = dict(
        os.environ,
        OPENAI_LOG_FORMAT="log,csv,tensorboard",
        OPENAI_LOGDIR=os.fspath(working_dir / "training_logs"),
    )

    setup_config(working_dir, params.config)
    os.chdir(working_dir)

    try:
        print("Start robots")
        subprocess.run(["hysr_start_robots"], check=True)

        print("Start learning")
        start = time.time()
        subprocess.run(["hysr_one_ball_ppo"], env=env, check=True)
        duration = (time.time() - start) / 60
        print("\n\nlearning took %0.2f min" % duration)

    finally:
        print("We are done, shut down.")
        subprocess.run(["hysr_stop"])

    # extract eprewmean from the log
    log_dir = pathlib.Path(env["OPENAI_LOGDIR"])
    eprewmean = read_reward_from_log(log_dir)
    print("Final Reward:", float(eprewmean))

    # save the reward for cluster utils
    metrics = {"eprewmean": eprewmean}
    cluster.save_metrics_params(metrics, params)

    return 0


if __name__ == "__main__":
    sys.exit(main())
