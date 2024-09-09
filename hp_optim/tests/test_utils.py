import pytest

from utils import RestartInfo


@pytest.fixture
def restart_info(tmp_path):
    return RestartInfo(tmp_path / "restartinfo.json")


def test_restartinfo_init(restart_info):
    """Test if a new RestartInfo instance is initialised correctly."""
    assert restart_info.finished_trainigs == 0
    assert restart_info.failed_attempts == 0
    assert restart_info.rewards == []


def test_restartinfo_finish(restart_info):
    """Test if adding finished runs works correctly."""
    restart_info.mark_training_finished(42)
    assert restart_info.finished_runs == 1
    assert restart_info.failed_attempts == 0
    assert restart_info.rewards == [42]

    restart_info.mark_run_finished(13)
    assert restart_info.finished_runs == 2
    assert restart_info.failed_attempts == 0
    assert restart_info.rewards == [42, 13]


def test_restartinfo_failed(restart_info):
    """Test if marking runs as failed increases the counter correctly."""
    restart_info.mark_attempt_failed()
    assert restart_info.finished_runs == 0
    assert restart_info.failed_attempts == 1

    restart_info.mark_attempt_failed()
    assert restart_info.finished_runs == 0
    assert restart_info.failed_attempts == 2

    # marking finished should reset the failed attempts counter
    restart_info.mark_run_finished(42)
    assert restart_info.finished_runs == 1
    assert restart_info.failed_attempts == 0

    restart_info.mark_attempt_failed()
    assert restart_info.finished_runs == 1
    assert restart_info.failed_attempts == 1


def test_restartinfo_finish_save(restart_info):
    """Test if saving and reloading works with some finished runs."""
    restart_info.mark_run_finished(42)
    restart_info.mark_run_finished(13)
    restart_info.save()

    # load new instance from the saved file
    rs2 = RestartInfo(restart_info.file_path)
    assert restart_info._data == rs2._data


def test_restartinfo_failed_save(restart_info):
    """Test if saving and reloading works with some failed attempts."""
    restart_info.mark_attempt_failed()
    restart_info.mark_attempt_failed()
    restart_info.mark_run_finished(42)
    restart_info.mark_attempt_failed()
    restart_info.save()

    # load new instance from the saved file
    rs2 = RestartInfo(restart_info.file_path)
    assert restart_info._data == rs2._data
