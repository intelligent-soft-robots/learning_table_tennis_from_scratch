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

import numpy as np
import cluster
import smart_settings.param_classes


# Max. number of attempts.  If it fails this many times, do not try again.
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_LEARNING_RUNS_PER_JOB = 3
RESTART_INFO_FILENAME = "restarts.json"


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


def setup_config(working_dir: pathlib.Path, params: dict) -> pathlib.Path:
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

    Returns:
        Path to the main config file.
    """
    main_config_file = working_dir / "config.json"
    # if config is already there, skip creation (this is the case when
    # restarting after a failure)
    if main_config_file.exists():
        print("config.json already exists, skip setup.")
    else:
        with open(params["config_templates"], "r") as f:
            config_file_templates = json.load(f)

        config_dir = working_dir / "config"
        config_dir.mkdir(exist_ok=True, parents=True)

        base_config = {
            "reward_config": "./config/reward_config.json",
            "hysr_config": "./config/hysr_config.json",
            "pam_config": "./config/pam_config.json",
            "ppo_config": "./config/ppo_config.json",
            "ppo_common_config": "./config/ppo_common_config.json",
        }
        with open(main_config_file, "w") as f:
            json.dump(base_config, f)

        base_config_path = pathlib.PurePath(params["config_templates"]).parent
        for name in base_config.keys():
            template_path = base_config_path / config_file_templates[name]
            prepare_config_file(name, template_path, params.get(name, {}), config_dir)

    return main_config_file


def read_reward_from_log(log_dir: pathlib.PurePath) -> float:
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


# TODO maybe better to implement this in a class
def run_learning(config_file, env, proc_backend):  # TODO type hints
    print("\n\n#### Start learning\n", flush=True)
    start = time.time()
    proc_learning = subprocess.Popen(
        ["hysr_one_ball_ppo", os.fspath(config_file)], env=env
    )

    # monitor processes
    while True:
        time.sleep(5)

        if proc_backend.poll() is not None:
            # backend terminated, this is bad!
            # kill learning and fail
            proc_learning.kill()
            raise subprocess.CalledProcessError(
                proc_backend.returncode, proc_backend.args
            )

        if proc_learning.poll() is not None:
            if proc_learning.returncode == 0:
                break
            else:
                raise subprocess.CalledProcessError(
                    proc_learning.returncode, proc_learning.args
                )
                # there is no need to explicitly terminate the backend
                # here, this is done by hysr_stop below

    duration = (time.time() - start) / 60
    print("\n\nlearning took %0.2f min" % duration, flush=True)

    # extract eprewmean from the log
    log_dir = pathlib.Path(env["OPENAI_LOGDIR"])
    eprewmean = read_reward_from_log(log_dir)
    print("Final Reward:", eprewmean, flush=True)

    return eprewmean


def main():
    # get parameters (make mutable so defaults can easily be set later)
    params = cluster.read_params_from_cmdline(make_immutable=False)
    working_dir = pathlib.Path(params.working_dir)

    # Overwrite some values from the config templates to disable any graphical
    # interfaces and to save the model in working_dir (unless a different value
    # is already explicitly set in params).
    default_params = {
        "config": {
            "hysr_config": {
                "graphics": False,
                "xterms": False,
            },
            "ppo_config": {
                "save_path": os.fspath(working_dir / "model"),
            },
        },
        "learning_runs_per_job": DEFAULT_LEARNING_RUNS_PER_JOB,
        "max_attempts": DEFAULT_MAX_ATTEMPTS,
    }
    params = smart_settings.param_classes.update_recursive(
        params, default_params, overwrite=False
    )

    print("Params:")
    print(params)
    print("-------------")

    # prepare environment
    env = dict(
        os.environ,
        OPENAI_LOG_FORMAT="log,csv,tensorboard",
        OPENAI_LOGDIR=os.fspath(working_dir / "training_logs"),
    )

    config_file = setup_config(working_dir, params.config)
    os.chdir(working_dir)

    restart_info_path = working_dir / RESTART_INFO_FILENAME
    if restart_info_path.exists():
        with open(restart_info_path, "r") as f:
            restart_info = json.load(f)
    else:
        restart_info = {
            "finished_runs": 0,
            "eprewmean": [],
            "failed_runs": [0] * params.learning_runs_per_job,
        }

    run_number = restart_info["finished_runs"]
    failed_runs = restart_info["failed_runs"][run_number]
    run_id = f"{run_number}-{failed_runs}"

    try:
        print(
            f"\n\n############################# Start Run {run_id}\n",
            flush=True,
        )

        print("#### Start robots\n", flush=True)
        proc_backend = subprocess.Popen(["hysr_start_robots", os.fspath(config_file)])

        eprewmean = run_learning(config_file, env, proc_backend)

        restart_info["finished_runs"] += 1
        restart_info["eprewmean"].append(eprewmean)

    except subprocess.CalledProcessError as e:
        print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("RUN FAILED! (%s terminated with exit code %d)" % (e.cmd, e.returncode))

        failed_runs += 1
        restart_info["failed_runs"][run_number] = failed_runs

    finally:
        print("Stop backend [hysr_stop]")
        subprocess.run(["hysr_stop"])

    # rename training log directory, so it does not get overwritten by next run
    training_log_dir = pathlib.Path(env["OPENAI_LOGDIR"])
    if training_log_dir.exists():
        new_trainig_log_dir = env["OPENAI_LOGDIR"] + "_" + run_id
        training_log_dir.rename(new_trainig_log_dir)

    # store the restart log
    with open(restart_info_path, "w") as f:
        json.dump(restart_info, f)

    # if it failed too often, abort with error
    if failed_runs >= params.max_attempts:
        raise RuntimeError("Maximum number of retries is reached.  Exit with failure.")

    # if desired number of runs have not yet finished, exit for resume (i.e. restart
    # with a new cluster job)
    if restart_info["finished_runs"] < params.learning_runs_per_job:
        # print separator to both stdout and stderr
        print(
            f"Exit for restart [{run_id}].\n"
            "==========================================\n\n"
        )
        print(
            f"Exit for restart [{run_id}].\n"
            "==========================================\n\n",
            file=sys.stderr,
        )

        fraction_finished = restart_info["finished_runs"] / params.learning_runs_per_job
        cluster.announce_fraction_finished(fraction_finished)
        cluster.exit_for_resume()

    # if this line is reached, all N runs have finished --> save the results

    # save the reward for cluster utils
    metrics = {
        "mean_eprewmean": np.mean(restart_info["eprewmean"]),
        "std_eprewmean": np.std(restart_info["eprewmean"]),
    }
    print(
        "Result of {} runs: {mean_eprewmean:.4f} (std: {std_eprewmean:.4f})".format(
            params.learning_runs_per_job, **metrics
        )
    )
    cluster.save_metrics_params(metrics, params)

    return 0


if __name__ == "__main__":
    sys.exit(main())
