"""Utils for the hyperparameter search with cluster_utils."""
import json
import logging
import os
import pathlib
import sys
import typing


class RestartInfo:
    """Provides access to the restart info.

    The "restart info" contains information about the number of already finished
    training runs as well as failed attempts of the current job.
    """

    __slots__ = ["logger", "file_path", "_data"]

    def __init__(self, file_path: typing.Union[str, os.PathLike]):
        """
        Args:
            file_path: Path to the file in which the restart info is stored.
        """
        self.logger = init_logger("RestartInfo")
        self.file_path = pathlib.Path(file_path)

        # load
        if self.file_path.exists():
            self.logger.info("Load restart info from %s", self.file_path)
            with open(self.file_path, "r") as f:
                self._data = json.load(f)
        else:
            self.logger.info("Initialise new restart info.")
            self._data = {
                "training_continuation_counter": 0,
                "unfinished_model": "",
                "finished_trainings": 0,
                "eprewmean": [],
                "failed_attempts": {},
            }

    def save(self):
        """Save changes to file.

        Call this to save changes on the restart info to the file.
        """
        self.logger.info("Save restart info to %s", self.file_path)
        with open(self.file_path, "w") as f:
            json.dump(self._data, f)

    @property
    def training_continuation_counter(self) -> int:
        """Get number of times the has been continued in a new job."""
        return self._data["training_continuation_counter"]

    @property
    def unfinished_model(self) -> str:
        """Get path to the unfinished model or empty string if there is none."""
        return self._data["unfinished_model"]

    @property
    def finished_trainigs(self) -> int:
        """Get number of already finished training runs.

        This also corresponds to the index of the current run (starting at zero).
        """
        return self._data["finished_trainings"]

    @property
    def failed_attempts(self) -> int:
        """Get number of failed attempts for the current run."""
        try:
            return self._data["failed_attempts"][str(self.finished_trainigs)]
        except KeyError:
            return 0

    @property
    def rewards(self) -> typing.List[float]:
        """Get list of rewards of all runs."""
        return self._data["eprewmean"]

    def continue_training(self, model_path: str):
        """Increases the training continuation counter by one.

        Args:
            model_path: Path to the model for which training should be continued.
        """
        self._data["unfinished_model"] = model_path
        self._data["training_continuation_counter"] += 1
        self.logger.info(
            "continue training (counter = %d)",
            self._data["training_continuation_counter"],
        )

    def mark_training_finished(self, eprewmean: float):
        """Mark the current run as successfully finished and log its reward.

        Args:
            eprewmean: Reward that was achieved by the run.
        """
        self.logger.info(
            "Mark training %d as finished.", self._data["finished_trainings"]
        )
        self._data["unfinished_model"] = ""
        self._data["training_continuation_counter"] = 0
        self._data["finished_trainings"] += 1
        self._data["eprewmean"].append(eprewmean)

    def mark_attempt_failed(self):
        """Mark the current attempt as failed."""
        try:
            self._data["failed_attempts"][str(self.finished_trainigs)] += 1
        except KeyError:
            self._data["failed_attempts"][str(self.finished_trainigs)] = 1

        self.logger.info(
            "Mark attempt failed (counter = %d)",
            self._data["failed_attempts"][str(self.finished_trainigs)],
        )


def init_logger(name: typing.Optional[str] = None) -> logging.Logger:
    """Initialise stdout/stderr-logger.

    The logger is configured to write messages with level <= INFO to stdout and any
    higher levels to stderr.

    Args:
        name: Name of the application (added to each message).
    """
    formatter = logging.Formatter("[%(name)s %(levelname)s %(asctime)s] %(message)s")

    # Code below mostly by Zoey Greer, CC BY-SA 3.0
    # (https://stackoverflow.com/a/31459386, 2022-03-10)
    class LessThanFilter(logging.Filter):
        def __init__(self, exclusive_maximum, name=""):
            super(LessThanFilter, self).__init__(name)
            self.max_level = exclusive_maximum

        def filter(self, record):
            # non-zero return means we log this message
            return 1 if record.levelno < self.max_level else 0

    logger = logging.getLogger(name)
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
