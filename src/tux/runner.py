"""Execute a staged command in tux's own subprocess and record it to the run log.

tux proposes a command and the user must explicitly choose to run it; only then
is the command executed here, in a subshell that inherits tux's
stdin/stdout/stderr so the command's own output reaches the user and interactive
commands keep working. Because tux — not the user's shell — runs the command, it
never lands in the shell's history, so each command tux actually runs is
appended to an on-disk log for traceability.

Execution is a small callable seam (:data:`CommandRunner`) mirroring the model
client's injectable transport, so tests can assert run and log behavior without
spawning real processes. A run *tees* its output: the command's combined output
is shown to the user as it arrives while a bounded copy is captured and returned,
so a discovery step's result can be fed back to the model to resolve a later
step. The captured copy is deliberately never written to the run log (size and
secret-leakage risk).
"""

import subprocess
import sys
import time
from collections.abc import Callable

from tux.state import log_path

#: Upper bound on captured output characters. The capture exists to feed the
#: model and is bounded so a large output cannot blow the model context or bloat
#: the conversation thread; the value is a sane cap, not a hard contract.
MAX_CAPTURED_CHARS = 4000

#: A command runner executes a shell command and returns ``(status, output)`` —
#: its exit status and a bounded copy of what it printed. The default runs it in
#: a subshell, teeing output to the terminal; tests inject a fake so the walk is
#: exercised without spawning a real process.
CommandRunner = Callable[[str], tuple[int, str]]


def run_command(command: str) -> tuple[int, str]:
    """Run ``command`` in a subshell, teeing output, and return ``(status, output)``.

    The command's combined stdout/stderr is streamed straight to the user's
    terminal as it arrives — so progressive output stays live — while a bounded
    copy (capped at :data:`MAX_CAPTURED_CHARS`) is buffered and returned so a
    discovery step's result can be fed back to the model. The captured copy is
    for the model and the user only; it is never written to the run log.
    """
    captured: list[str] = []
    remaining = MAX_CAPTURED_CHARS
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    stream = process.stdout
    if stream is not None:
        for line in stream:
            sys.stdout.write(line)
            sys.stdout.flush()
            if remaining > 0:
                kept = line[:remaining]
                captured.append(kept)
                remaining -= len(kept)
    status = process.wait()
    return status, "".join(captured)


def append_run(command: str, status: int) -> None:
    """Append one run record — timestamp, exit status, command — to the run log.

    The parent directory is created if absent. Only the command tux actually ran
    is recorded; the command's own output is never written here.
    """
    path = log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{timestamp}\t{status}\t{command}\n")
