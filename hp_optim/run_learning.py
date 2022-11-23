#!/usr/bin/env python3
"""Run learning with cluster_utils

This is a wrapper script around running hysr_start_robots and hysr_one_ball_rl
that is intended to be used by cluster_utils (for hyperparameter optimisation).

See the accompanying README on how cluster_utils needs to be configured for
this to work.
"""
import csv
import json
import logging
import os
import re
import subprocess
import sys
import time
import typing
from pathlib import Path, PurePath

import numpy as np
import cluster  # type: ignore
import smart_settings.param_classes  # type: ignore

from utils import RestartInfo, init_logger


# Max. number of attempts.  If it fails this many times, do not try again.
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_LEARNING_RUNS_PER_JOB = 3
DEFAULT_TRAINING_ITERATIONS = 1
RESTART_INFO_FILENAME = "restarts.json"

_logger = None


def get_logger() -> logging.Logger:
    """Get global logger instance."""
    global _logger

    if _logger is None:
        _logger = init_logger()

    return _logger


def prepare_config_file(
    name: str,
    template_file: typing.Union[str, os.PathLike],
    parameter_updates: dict,
    destination_directory: PurePath,
) -> None:
    """Create config file based on the given template and parameter updates.

    The given template is expected to be a valid config file in JSON format.
    The *parameter_updates* dict can be used to overwrite some of the default
    values with custom ones.

    If a parameter name is suffixed with ``:#`` (where # is a number), it will assume
    that the value of this parameter is a list and will overwrite the value at index #.

    The resulting config will be written to "destination_directory/name.json".

    Args:
        name:  Name of the config file.
        template_file:  Path to the template file.
        parameter_updates:  Dictionary with parameters that are to be
            overwritten in the template file.
        destination_directory:  Directory to which the resulting configuration
            file is written.
    """
    logger = get_logger()

    with open(template_file, "r") as f:
        config = json.load(f)

    list_parameters = []

    # check if all provided parameter updates correspond to existing parameters
    for param in parameter_updates:

        # if parameter name ends with ":#", this indicates the #-th index of a parameter
        # whose value is a list
        m = re.match(r"(.*):([0-9])+$", param)
        if m is not None:
            param_name = m.group(1)
            index = int(m.group(2))
            logger.info("Process list parameter %s[%d]" % (param_name, index))
            list_parameters.append((param_name, index, parameter_updates[param]))

            # overwrite for following check
            param = param_name

        if param not in config:
            raise KeyError(
                (
                    "Provided update for parameter '{param}' in {config_name} [{file}],"
                    " but such parameter does not exist."
                ).format(config_name=name, file=template_file, param=param)
            )

    config.update(parameter_updates)

    for param, index, value in list_parameters:
        try:
            config[param][index] = value
        except IndexError:
            raise IndexError(
                (
                    "Provided update for parameter '{param}[{index}]' in {config_name}"
                    " [{file}], but index is out of bounds."
                ).format(config_name=name, file=template_file, param=param, index=index)
            )

    destination_file = destination_directory / "{}.json".format(name)
    with open(destination_file, "w") as f:
        json.dump(config, f, indent=4)


def setup_config(working_dir: Path, run_data_dir: Path, params: dict) -> Path:
    """Set up config files based on templates and parameters.

    The *params* dictionary is expected to have an entry "config_templates" which
    contains the path to a JSON file pointing to the different config template files.
    Expected entries in that file are:

    - "reward_config"
    - "hysr_config"
    - "pam_config"
    - "rl_config"
    - "rl_common_config"

    Further *params* may contain contain an entry for any of the template files listed
    above, which again contains a dictionary with parameter updates for that file.
    These parameters are overwritten in the template files.

    Example for *params*:

    .. code-block:: Python

        {
            "config_templates": "./templates/base_config.json",
            "rl_config": {
                "num_timesteps": 1000000,
                "num_layers": 2
            }
        }

    Args:
        working_dir:  Directory in which the job is executed.
        run_data_dir:  Directory in which the run-specific data should be stored.
        params:  Dictionary of parameters (see above).

    Returns:
        Path to the main config file.
    """
    logger = get_logger()

    base_config = {
        "reward_config": str(run_data_dir / "./config/reward_config.json"),
        "hysr_config": str(run_data_dir / "./config/hysr_config.json"),
        "pam_config": str(run_data_dir / "./config/pam_config.json"),
        "rl_config": str(run_data_dir / "./config/rl_config.json"),
        "rl_common_config": str(run_data_dir / "./config/rl_common_config.json"),
    }

    # raise error if there is any unexpected value in the config
    valid_param_keys = list(base_config.keys()) + ["config_templates"]
    for key in params.keys():
        if key not in valid_param_keys:
            raise KeyError("Unexpected key '{}' in params".format(key))

    main_config_file = working_dir / "config.json"

    with open(params["config_templates"], "r") as f:
        config_file_templates = json.load(f)

    config_dir = run_data_dir / "config"
    config_dir.mkdir(exist_ok=True, parents=True)

    with open(main_config_file, "w") as f:
        json.dump(base_config, f, indent=4)

    base_config_path = PurePath(params["config_templates"]).parent
    for name in base_config.keys():
        template_path = base_config_path / config_file_templates[name]
        prepare_config_file(name, template_path, params.get(name, {}), config_dir)

    return main_config_file


def read_reward_from_log(log_dir: PurePath) -> float:
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
            if line["rollout/ep_rew_mean"]:
                eprewmean = float(line["rollout/ep_rew_mean"])

    if eprewmean is None:
        raise RuntimeError("Failed to read eprewmean from log file.")

    return eprewmean


class Runner:
    """Run learning in subprocesses.

    This class takes care of starting the necessary scripts for the learning (using
    subprocesses), monitoring them while they are running (to detect failures) and
    acquire the final reward from the generated log files.
    """

    def __init__(
        self,
        config_file: typing.Union[str, os.PathLike],
        training_log_dir: typing.Union[str, os.PathLike],
    ):
        """
        Args:
            config_file: Path to the config file for hysr_one_ball_rl.
            training_log_dir: Directory to which leaning log files are written (used to
                set environment variable OPENAI_LOGDIR for the learning process).
        """
        # prepare environment
        self.training_log_dir = Path(training_log_dir)
        self.config_file = config_file

        self.logger = get_logger()

    def start_backend(self):
        self.logger.info("#### Start robots [hysr_start_robots]\n")
        self.proc_backend = subprocess.Popen(
            ["hysr_start_robots", os.fspath(self.config_file)]
        )

    def stop_backend(self):
        self.logger.info("#### Stop backend [hysr_stop]")
        subprocess.run(["hysr_stop"])

    def start_learning(self):
        self.logger.info("\n\n#### Start learning [hysr_one_ball_rl]\n")
        self.learning_start_time = time.time()
        self.proc_learning = subprocess.Popen(
            ["hysr_one_ball_rl", os.fspath(self.config_file)]
        )

    def monitor_processes(self):
        """Monitor running processes of backend and learning script.

        Blocks until the learning is finished.  While it is running, the processes are
        monitored, raising an CalledProcessError if one of them terminates unexpectedly.

        Raises:
            subprocess.CalledProcessError: if one of the processes fails.
        """
        while True:
            time.sleep(5)

            if self.proc_backend.poll() is not None:
                # backend terminated, this is bad! Kill learning and fail.
                self.logger.fatal(
                    "Back end unexpectedly terminated with return code %d.",
                    self.proc_backend.returncode,
                )
                self.logger.info("Kill learning process.")
                self.proc_learning.kill()
                raise subprocess.CalledProcessError(
                    self.proc_backend.returncode, self.proc_backend.args
                )

            if self.proc_learning.poll() is not None:
                self.logger.info(
                    "Learning process terminated with return code %d.",
                    self.proc_learning.returncode,
                )
                if self.proc_learning.returncode == 0:
                    # learning finished cleanly, all good
                    break
                else:
                    raise subprocess.CalledProcessError(
                        self.proc_learning.returncode, self.proc_learning.args
                    )
                    # there is no need to explicitly terminate the backend
                    # here, this is done by hysr_stop below

    def run(self) -> float:
        """Start the learning and monitor processes.

        Returns:
            The score of the learning (eprewmean).

        Raises:
            subprocess.CalledProcessError: if one of the processes fails.
        """
        self.start_backend()
        self.start_learning()

        # monitor processes
        self.monitor_processes()

        duration = (time.time() - self.learning_start_time) / 60.0
        self.logger.info("\n\nlearning took %0.2f min" % duration)

        # extract eprewmean from the log
        eprewmean = read_reward_from_log(self.training_log_dir)
        self.logger.info("Final Reward: %0.2f" % eprewmean)

        return eprewmean


def main() -> int:
    logger = get_logger()

    # get parameters (make mutable so defaults can easily be set later)
    params = cluster.read_params_from_cmdline(make_immutable=False)

    working_dir = Path(params.working_dir)
    restart_info_file = working_dir / RESTART_INFO_FILENAME

    restart_info = RestartInfo(restart_info_file)

    run_id = "{}.{}-{}".format(
        restart_info.finished_trainigs,
        restart_info.training_continuation_counter,
        restart_info.failed_attempts,
    )

    run_data_dir = working_dir / f"run_{run_id}"
    training_log_dir = run_data_dir / "training_logs"
    model_save_path = os.fspath(run_data_dir / "model")

    # Overwrite some values from the config templates to disable any graphical
    # interfaces and to save the model in working_dir (unless a different value
    # is already explicitly set in params).
    default_params = {
        "config": {
            "hysr_config": {
                "graphics": False,
                "xterms": False,
            },
            "rl_config": {
                "load_path": restart_info.unfinished_model,
                "save_path": model_save_path,
                "log_path": os.fspath(training_log_dir),
            },
        },
        "training_iterations": DEFAULT_TRAINING_ITERATIONS,
        "learning_runs_per_job": DEFAULT_LEARNING_RUNS_PER_JOB,
        "max_attempts": DEFAULT_MAX_ATTEMPTS,
    }
    params = smart_settings.param_classes.update_recursive(
        params, default_params, overwrite=False
    )
    logger.info("Params:\n%s\n-------------" % params)

    config_file = setup_config(working_dir, run_data_dir, params.config)
    os.chdir(working_dir)

    runner = Runner(config_file, training_log_dir)

    try:
        logger.info(f"\n\n############################# Start Run {run_id}\n")

        eprewmean = runner.run()

        restart_info.continue_training(model_save_path)
        if restart_info.training_continuation_counter >= params.training_iterations:
            restart_info.mark_training_finished(eprewmean)

    except subprocess.CalledProcessError as e:
        logger.error("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        logger.error(
            "RUN FAILED! (%s terminated with exit code %d)" % (e.cmd, e.returncode)
        )

        restart_info.mark_attempt_failed()

    finally:
        runner.stop_backend()

    # store the restart info
    restart_info.save()

    # if it failed too often, abort with error
    if restart_info.failed_attempts >= params.max_attempts:
        raise RuntimeError("Maximum number of retries is reached.  Exit with failure.")

    # if desired number of runs have not yet finished, exit for resume (i.e. restart
    # with a new cluster job)
    if restart_info.finished_trainigs < params.learning_runs_per_job:
        # print separator to both stdout and stderr (so we have it in both log files)
        logger.info(
            f"Exit for restart [{run_id}].\n"
            "==========================================\n\n"
        )
        logger.warning(
            f"Exit for restart [{run_id}].\n"
            "==========================================\n\n",
        )

        training_progress = (
            restart_info.training_continuation_counter / params.training_iterations
        )
        fraction_finished = (
            restart_info.finished_trainigs + training_progress
        ) / params.learning_runs_per_job
        cluster.announce_fraction_finished(fraction_finished)
        cluster.exit_for_resume()

    # if this line is reached, all N runs have finished --> save the results
    metrics = {
        "mean_eprewmean": np.mean(restart_info.rewards),
        "std_eprewmean": np.std(restart_info.rewards),
    }
    logger.info(
        "Result of {} runs: {mean_eprewmean:.4f} (std: {std_eprewmean:.4f})".format(
            params.learning_runs_per_job, **metrics
        )
    )
    cluster.save_metrics_params(metrics, params)

    return 0


if __name__ == "__main__":
    sys.exit(main())
