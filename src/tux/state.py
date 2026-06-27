"""Per-terminal conversation threads persisted under the XDG state directory.

Each one-shot ``tux ask`` process is short-lived, so context can only carry
across invocations if it lives on disk. tux keys a thread to the parent shell's
PID (``os.getppid()``): consecutive ``tux ask`` calls from the *same* terminal
load that shell's thread, append the new turn, and save it, while a call from a
different terminal (a different shell PID) uses its own file.

Threads live at ``$XDG_STATE_HOME/tux/threads/<ppid>.json`` (falling back to
``~/.local/state/tux/threads/<ppid>.json`` when ``XDG_STATE_HOME`` is unset).
This is *state*, not config: the endpoint/model config (see :mod:`tux.config`)
stays under ``$XDG_CONFIG_HOME`` and is untouched here.

Because the OS eventually reuses shell PIDs, a thread is treated as fresh (an
empty history) rather than resurrected when it is stale: either older than
:data:`THREAD_TTL`, or recorded against a shell whose start time no longer
matches the PID's current start time. Any missing, empty, or corrupt file also
degrades to a fresh thread rather than raising.
"""

import json
import os
import time
from pathlib import Path

#: Conversation messages older than this many seconds are treated as a fresh
#: thread. Eight hours comfortably spans a working session while keeping a
#: long-idle terminal — or a PID the OS has since reused — from resurrecting an
#: unrelated conversation.
THREAD_TTL = 8 * 60 * 60


def state_dir() -> Path:
    """Return tux's state directory, honoring ``XDG_STATE_HOME``.

    The directory is returned whether or not it currently exists.
    """
    base = os.environ.get("XDG_STATE_HOME")
    root = Path(base) if base else Path.home() / ".local" / "state"
    return root / "tux"


def thread_path(ppid: int) -> Path:
    """Return the thread file path for the shell identified by ``ppid``."""
    return state_dir() / "threads" / f"{ppid}.json"


def log_path() -> Path:
    """Return the path of tux's command run log under the state directory.

    The log is the ``$XDG_STATE_HOME`` sibling of the conversation threads,
    living at ``$XDG_STATE_HOME/tux/history.log`` (falling back to
    ``~/.local/state/tux/history.log``). The path is returned whether or not the
    file currently exists.
    """
    return state_dir() / "history.log"


def load_thread(ppid: int) -> list[dict[str, str]]:
    """Return the stored conversation history for ``ppid``, oldest turn first.

    A missing, empty, corrupt, or stale thread yields an empty list so the
    caller simply starts a fresh conversation instead of crashing or carrying
    context onto an unrelated shell.
    """
    data = _read(thread_path(ppid))
    if data is None or _is_stale(data, ppid):
        return []
    history = data.get("history")
    if not _valid_history(history):
        return []
    return history


def save_thread(ppid: int, history: list[dict[str, str]]) -> None:
    """Persist ``history`` for ``ppid``, stamping the shell start time and clock.

    The write is atomic (write a temp file, then replace) so a process killed
    mid-write cannot leave a half-written thread behind.
    """
    path = thread_path(ppid)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ppid": ppid,
        "shell_start": _shell_start(ppid),
        "updated_at": time.time(),
        "history": history,
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(path)


def clear_thread(ppid: int) -> None:
    """Discard the stored thread for ``ppid``; a no-op when none exists."""
    thread_path(ppid).unlink(missing_ok=True)


def _read(path: Path) -> dict | None:
    """Return the parsed thread object, or ``None`` if absent/empty/corrupt."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    text = text.strip()
    if not text:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _is_stale(data: dict, ppid: int) -> bool:
    """Return whether a stored thread is too old or bound to a reused PID."""
    updated = data.get("updated_at")
    if not isinstance(updated, (int, float)) or time.time() - updated > THREAD_TTL:
        return True
    recorded = data.get("shell_start")
    current = _shell_start(ppid)
    # Only reject on a positive mismatch; if either start time is unavailable
    # the TTL above remains the safeguard.
    return recorded is not None and current is not None and recorded != current


def _valid_history(history: object) -> bool:
    """Return whether ``history`` is a list of well-formed chat messages."""
    if not isinstance(history, list):
        return False
    return all(
        isinstance(turn, dict)
        and isinstance(turn.get("role"), str)
        and isinstance(turn.get("content"), str)
        for turn in history
    )


def _shell_start(pid: int) -> str | None:
    """Return the process start time of ``pid`` from ``/proc``, or ``None``.

    Field 22 of ``/proc/<pid>/stat`` is the start time in clock ticks since
    boot, which distinguishes a live shell from a later process that reused its
    PID. The comm field (field 2) may itself contain spaces and parentheses, so
    parsing resumes after the final ``)``. Returns ``None`` off Linux or when
    the entry cannot be read, leaving the TTL as the sole staleness guard.
    """
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        after_comm = stat[stat.rindex(")") + 1:].split()
        return after_comm[19]
    except (ValueError, IndexError):
        return None
