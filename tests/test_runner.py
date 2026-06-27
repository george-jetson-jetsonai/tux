"""Tests for the command run log (execution itself is the untested real seam)."""

import pytest

from tux.runner import append_run
from tux.state import log_path


@pytest.fixture(autouse=True)
def _state_home(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Point the state dir at a temp dir so tests never touch the real home."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    return tmp_path / "state"


def test_log_path_lives_under_xdg_state_home(_state_home) -> None:
    """The run log sits beside the threads, at ``$XDG_STATE_HOME/tux/history.log``."""
    assert log_path() == _state_home / "tux" / "history.log"


def test_append_run_creates_parent_and_records_command_and_status() -> None:
    """A run record carries the command and its exit status, creating the dir."""
    append_run("ls -laS", 0)
    line = log_path().read_text(encoding="utf-8").strip()
    fields = line.split("\t")
    assert fields[1] == "0"
    assert fields[2] == "ls -laS"
    # A timestamp leads the record so the log is a dated trace.
    assert fields[0]


def test_append_run_appends_rather_than_overwrites() -> None:
    """Each run adds a line; an earlier record is never clobbered."""
    append_run("true", 0)
    append_run("false", 1)
    lines = log_path().read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert lines[0].endswith("true") and "\t0\t" in lines[0]
    assert lines[1].endswith("false") and "\t1\t" in lines[1]
