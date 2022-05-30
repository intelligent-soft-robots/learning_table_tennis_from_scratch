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
import pathlib
import re
import subprocess
import sys
import time
import typing

import numpy as np
import cluster
import smart_settings.param_classes

from utils import RestartInfo


# Max. number of attempts.  If it fails this many times, do not try again.
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_LEARNING_RUNS_PER_JOB = 3
RESTART_INFO_FILENAME = "restarts.json"

_logger = None


def init_logger(name: str = None) -> logging.Logger:
    """Initialise stdout/stderr-logger.

    The logger is configured to write messages with level <= INFO to stdout and any
    higher levels to stderr.

    Args:
        name: Name of the application (added to each message).
    """
    if name is None:
        name = pathlib.PurePath(__file__).name
    formatter = logging.Formatter(
        "[{} %(levelname)s %(asctime)s] %(message)s".format(name)
    )

    # Code below mostly by Zoey Greer, CC BY-SA 3.0
    # (https://stackoverflow.com/a/31459386, 2022-03-10)
    class LessThanFilter(logging.Filter):
        def __init__(self, exclusive_maximum, name=""):
            super(LessThanFilter, self).__init__(name)
            self.max_level = exclusive_maximum

        def filter(self, record):
            # non-zero return means we log this message
            return 1 if record.levelno < self.max_level else 0

    logger = logging.getLogger()
    logger.setLevel(logging.NOTSET)

    handler_stdout = logging.StreamHandler(sys.stdout)
    handler_stdout.setLevel(logging.DEBUG)
    handler_stdout.addFilter(LessThanFilter(logging.WARNING))
    handler_stdout.setFormatter(formatter)
    logger.addHandler(handler_stdout)

    handler_stderr = logging.StreamHandler(sys.stderr)
    handler_stderr.setLevel(logging.WARNING)
    handler_stderr.setFormatter(formatter)
    logger.addHandler(handler_stderr)

    return logger


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
    destination_directory: pathlib.PurePath,
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
        json.dump(config, f)


def setup_config(working_dir: pathlib.Path, params: dict) -> pathlib.Path:
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
        working_dir:  Directory in which the config setup is created.
        params:  Dictionary of parameters (see above).

    Returns:
        Path to the main config file.
    """
    logger = get_logger()

    base_config = {
        "reward_config": "./config/reward_config.json",
        "hysr_config": "./config/hysr_config.json",
        "pam_config": "./config/pam_config.json",
        "rl_config": "./config/rl_config.json",
        "rl_common_config": "./config/rl_common_config.json",
    }

    # raise error if there is any unexpected value in the config
    valid_param_keys = list(base_config.keys()) + ["config_templates"]
    for key in params.keys():
        if key not in valid_param_keys:
            raise KeyError("Unexpected key '{}' in params".format(key))

    main_config_file = working_dir / "config.json"
    # if config is already there, skip creation (this is the case when
    # restarting after a failure)
    if main_config_file.exists():
        logger.info("config.json already exists, skip setup.")
    else:
        with open(params["config_templates"], "r") as f:
            config_file_templates = json.load(f)

        config_dir = working_dir / "config"
        config_dir.mkdir(exist_ok=True, parents=True)

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
            if line["rollout/ep_rew_mean"]:
                eprewmean = float(line["rollout/ep_rew_mean"])

    if eprewmean is None:
        raise RuntimeError("Failed to read eprewmean from log file.")

    return eprewmean


def rename_directory(path: typing.Union[str, os.PathLike], suffix: str):
    """Rename the given directory by adding ``suffix`` at the end if it exists.

    If path exists, rename it by appending suffix (separated with an underscore) to its
    name.  If it does not exist, do nothing.

    Args:
        path: Path that is to be renamed.
        suffix: Suffix that is added to the current name.
    """
    path = pathlib.Path(path)
    if path.exists():
        new_path = "{}_{}".format(path, suffix)
        path.rename(new_path)


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
        self.training_log_dir = pathlib.Path(training_log_dir)
        self.config_file = config_file

    def start_backend(self):
        get_logger().info("#### Start robots [hysr_start_robots]\n")
        self.proc_backend = subprocess.Popen(
            ["hysr_start_robots", os.fspath(self.config_file)]
        )

    def stop_backend(self):
        get_logger().info("#### Stop backend [hysr_stop]")
        subprocess.run(["hysr_stop"])

    def start_learning(self):
        get_logger().info("\n\n#### Start learning [hysr_one_ball_rl]\n")
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
                # backend terminated, this is bad!
                # kill learning and fail
                self.proc_learning.kill()
                raise subprocess.CalledProcessError(
                    self.proc_backend.returncode, self.proc_backend.args
                )

            if self.proc_learning.poll() is not None:
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
        logger = get_logger()

        self.start_backend()
        self.start_learning()

        # monitor processes
        self.monitor_processes()

        duration = (time.time() - self.learning_start_time) / 60.0
        logger.info("\n\nlearning took %0.2f min" % duration)

        # extract eprewmean from the log
        eprewmean = read_reward_from_log(self.training_log_dir)
        logger.info("Final Reward: %0.2f" % eprewmean)

        return eprewmean


def main() -> int:
    logger = get_logger()

    # get parameters (make mutable so defaults can easily be set later)
    params = cluster.read_params_from_cmdline(make_immutable=False)

    working_dir = pathlib.Path(params.working_dir)
    training_log_dir = working_dir / "training_logs"
    restart_info_file = working_dir / RESTART_INFO_FILENAME

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
                "save_path": os.fspath(working_dir / "model"),
                "log_path": os.fspath(training_log_dir),
            },
        },
        "learning_runs_per_job": DEFAULT_LEARNING_RUNS_PER_JOB,
        "max_attempts": DEFAULT_MAX_ATTEMPTS,
    }
    params = smart_settings.param_classes.update_recursive(
        params, default_params, overwrite=False
    )
    logger.info("Params:\n%s\n-------------" % params)

    config_file = setup_config(working_dir, params.config)
    os.chdir(working_dir)

    restart_info = RestartInfo(restart_info_file)
    runner = Runner(config_file, training_log_dir)

    run_id = f"{restart_info.finished_runs}-{restart_info.failed_attempts}"
    try:
        logger.info(f"\n\n############################# Start Run {run_id}\n")

        eprewmean = runner.run()
        restart_info.mark_run_finished(eprewmean)

    except subprocess.CalledProcessError as e:
        logger.error("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        logger.error(
            "RUN FAILED! (%s terminated with exit code %d)" % (e.cmd, e.returncode)
        )

        restart_info.mark_attempt_failed()

    finally:
        runner.stop_backend()

    # rename training log directory, so it does not get overwritten by next run
    rename_directory(training_log_dir, run_id)

    # store the restart info
    restart_info.save()

    # if it failed too often, abort with error
    if restart_info.failed_attempts >= params.max_attempts:
        raise RuntimeError("Maximum number of retries is reached.  Exit with failure.")

    # if desired number of runs have not yet finished, exit for resume (i.e. restart
    # with a new cluster job)
    if restart_info.finished_runs < params.learning_runs_per_job:
        # print separator to both stdout and stderr (so we have it in both log files)
        logger.info(
            f"Exit for restart [{run_id}].\n"
            "==========================================\n\n"
        )
        logger.warning(
            f"Exit for restart [{run_id}].\n"
            "==========================================\n\n",
        )

        fraction_finished = restart_info.finished_runs / params.learning_runs_per_job
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
