"""Live end-to-end test of a multi-turn conversation against ``llama-server``.

This drives the real :func:`tux.client.http_stream` transport over HTTP, so it
only runs where the configured endpoint is actually reachable. When the server
is down it is skipped (not failed), keeping the offline unit run green while the
tester environment — where the server is up — exercises the real path.
"""

import socket
from urllib.parse import urlparse

import pytest

from tux.cli import run_ask
from tux.client import DEFAULT_ENDPOINT, ModelClient
from tux.config import load_config
from tux.modes.command import assistant_turn

#: The endpoint the tester reaches; mirrors what the client builds from config.
ENDPOINT = load_config().get("endpoint", DEFAULT_ENDPOINT)


def _endpoint_reachable(endpoint: str) -> bool:
    """Return whether a TCP connection to ``endpoint`` succeeds within a second."""
    parsed = urlparse(endpoint)
    host = parsed.hostname or ""
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


@pytest.mark.skipif(
    not _endpoint_reachable(ENDPOINT),
    reason=f"llama-server endpoint {ENDPOINT} is not reachable",
)
def test_live_follow_up_uses_prior_turn() -> None:
    """A second turn against the live model reflects the file named in the first."""
    client = ModelClient.from_config()

    first = client.suggest("create an empty file called tux_ctx_probe.txt")
    assert first and all(step.command for step in first)

    history = [
        {"role": "user", "content": "create an empty file called tux_ctx_probe.txt"},
        assistant_turn(first),
    ]
    second = client.suggest("now delete that file", history)

    # The follow-up only makes sense given the first turn; a step must reference
    # the file introduced there rather than asking what to delete.
    assert any("tux_ctx_probe.txt" in step.command for step in second)


@pytest.mark.skipif(
    not _endpoint_reachable(ENDPOINT),
    reason=f"llama-server endpoint {ENDPOINT} is not reachable",
)
def test_live_conversational_follow_up_is_prose(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A command turn then a conversational follow-up: the reply is prose in context.

    Two separate ``run_ask`` calls share a thread on disk, as consecutive
    ``tux ask`` invocations from one terminal would. The second turn asks about
    the first conversationally; the answer should come back as ordinary prose
    that reflects the earlier permissions question, not a command proposal.
    """
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr("tux.cli.os.getppid", lambda: 31337)

    assert run_ask("How do I check a user's permissions in Linux?") == 0
    capsys.readouterr()  # drop the first turn's output

    assert run_ask("What did I just ask you about?") == 0
    answer = capsys.readouterr().out.strip()

    # A prose answer reflecting the first turn: multi-word, mentions permissions,
    # and is not a labelled title/command/description proposal.
    assert "permission" in answer.lower()
    assert len(answer.split()) > 3
