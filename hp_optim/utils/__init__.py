"""Utils for the hyperparameter search with cluster_utils."""
import json
import os
import pathlib
import typing


class RestartInfo:
    """Provides access to the restart info.

    The "restart info" contains information about the number of already finished
    training runs as well as failed attempts of the current job.
    """

    __slots__ = ["file_path", "_data"]

    def __init__(self, file_path: typing.Union[str, os.PathLike]):
        """
        Args:
            file_path: Path to the file in which the restart info is stored.
        """
        self.file_path = pathlib.Path(file_path)

        # load
        if self.file_path.exists():
            with open(self.file_path, "r") as f:
                self._data = json.load(f)
        else:
            self._data = {"finished_runs": 0, "eprewmean": [], "failed_attempts": {}}

    def save(self):
        """Save changes to file.

        Call this to save changes on the restart info to the file.
        """
        with open(self.file_path, "w") as f:
            json.dump(self._data, f)

    @property
    def finished_runs(self) -> int:
        """Get number of already finished runs.

        This also corresponds to the index of the current run (starting at zero).
        """
        return self._data["finished_runs"]

    @property
    def failed_attempts(self) -> int:
        """Get number of failed attempts for the current run."""
        try:
            return self._data["failed_attempts"][str(self.finished_runs)]
        except KeyError:
            return 0

    @property
    def rewards(self) -> typing.List[float]:
        """Get list of rewards of all runs."""
        return self._data["eprewmean"]

    def mark_run_finished(self, eprewmean: float):
        """Mark the current run as successfully finished and log its reward.

        Args:
            eprewmean: Reward that was achieved by the run.
        """
        self._data["finished_runs"] += 1
        self._data["eprewmean"].append(eprewmean)

    def mark_attempt_failed(self):
        """Mark the current attempt as failed."""
        try:
            self._data["failed_attempts"][str(self.finished_runs)] += 1
        except KeyError:
            self._data["failed_attempts"][str(self.finished_runs)] = 1
