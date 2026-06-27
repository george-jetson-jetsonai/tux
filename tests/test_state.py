"""Tests for the per-terminal on-disk conversation thread store."""

import json
import time

import pytest

from tux import state
from tux.state import (
    THREAD_TTL,
    clear_thread,
    load_thread,
    save_thread,
    thread_path,
)


@pytest.fixture(autouse=True)
def _state_home(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Point the state dir at a temp dir so tests never touch the real home."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    return tmp_path / "state"


def test_thread_path_lives_under_xdg_state_home(_state_home) -> None:
    """Threads are stored under ``$XDG_STATE_HOME/tux/threads/<ppid>.json``."""
    assert thread_path(123) == _state_home / "tux" / "threads" / "123.json"


def test_save_then_load_round_trips_history() -> None:
    """A saved thread is read back as the same history for the same PID."""
    history = [
        {"role": "user", "content": "create foo.txt"},
        {"role": "assistant", "content": '{"command": "touch foo.txt"}'},
    ]
    save_thread(321, history)
    assert load_thread(321) == history


def test_load_missing_thread_returns_empty() -> None:
    """A PID with no thread file loads as a fresh, empty history."""
    assert load_thread(999) == []


def test_different_pids_have_separate_threads() -> None:
    """Saving under one PID does not leak into another PID's thread."""
    save_thread(1, [{"role": "user", "content": "from shell one"}])
    assert load_thread(2) == []


def test_clear_thread_discards_history() -> None:
    """``clear_thread`` removes a saved thread; the next load starts fresh."""
    save_thread(7, [{"role": "user", "content": "hello"}])
    clear_thread(7)
    assert load_thread(7) == []


def test_clear_missing_thread_is_noop() -> None:
    """Clearing a thread that was never saved does not raise."""
    clear_thread(424242)  # no file exists; must not error


def test_corrupt_thread_file_loads_as_fresh() -> None:
    """A non-JSON thread file degrades to an empty history instead of crashing."""
    path = thread_path(11)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not valid json", encoding="utf-8")
    assert load_thread(11) == []


def test_empty_thread_file_loads_as_fresh() -> None:
    """An empty thread file loads as a fresh history."""
    path = thread_path(12)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    assert load_thread(12) == []


def test_malformed_history_loads_as_fresh() -> None:
    """A structurally valid file whose history is malformed degrades to fresh."""
    path = thread_path(13)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"updated_at": time.time(), "history": [{"role": "user"}]}
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_thread(13) == []


def test_stale_thread_past_ttl_loads_as_fresh() -> None:
    """A thread older than the TTL is treated as fresh rather than resurrected."""
    save_thread(14, [{"role": "user", "content": "old"}])
    path = thread_path(14)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["updated_at"] = time.time() - THREAD_TTL - 1
    path.write_text(json.dumps(data), encoding="utf-8")
    assert load_thread(14) == []


def test_mismatched_shell_start_loads_as_fresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A thread bound to a different shell start time (reused PID) loads fresh."""
    save_thread(15, [{"role": "user", "content": "from the original shell"}])
    path = thread_path(15)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["shell_start"] = "111111"
    path.write_text(json.dumps(data), encoding="utf-8")
    # The PID now reports a different start time, so the thread is not resurrected.
    monkeypatch.setattr(state, "_shell_start", lambda pid: "999999")
    assert load_thread(15) == []
