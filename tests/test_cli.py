"""Tests for the ``tux`` command-line interface."""

import json

import pytest

from tux import __version__
from tux.client import ModelClient, ModelClientError
from tux.cli import CLARIFY_CHOICE, DISMISS_CHOICE, RUN_CHOICE, main
from tux.modes.command import CommandSuggestion
from tux.state import log_path, thread_path


def run_choice(options) -> int:
    """A chooser that selects run (used as an injected fake)."""
    return RUN_CHOICE


def dismiss_choice(options) -> int:
    """A chooser that selects dismiss — index 0, the safe default."""
    return 0


class _FakeRunner:
    """Records the commands it runs and returns a canned ``(status, output)``."""

    def __init__(self, status: int = 0, output: str = "") -> None:
        self.status = status
        self.output = output
        self.commands: list[str] = []

    def __call__(self, command: str) -> tuple[int, str]:
        self.commands.append(command)
        return self.status, self.output


@pytest.fixture
def interactive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend tux is attached to a terminal so the run/dismiss menu is offered."""
    monkeypatch.setattr("tux.cli._interactive", lambda: True)


@pytest.fixture
def lite(config_home):
    """Record ``variant = "lite"`` in config so the lite gate engages.

    Exercises the real resolver: the value is written to the (isolated) config
    file and read back through ``_resolve_variant``, the same way the CLI does.
    """
    from tux.config import set_value

    set_value("variant", "lite")
    return config_home


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Redirect on-disk threads and config to a temp dir so tests never touch home.

    Isolating the config dir too means variant resolution defaults to full
    (variant unset) for every test that does not explicitly configure one, so the
    lite gate stays off unless a test opts in.
    """
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    return tmp_path / "state"


class _StubClient:
    """Stand-in model client that returns or raises a canned command result.

    Routing always lands on the command path; ``converse`` is present only so
    the client satisfies the same interface as the real one.
    """

    def __init__(self, suggestion: CommandSuggestion | None = None,
                 error: ModelClientError | None = None) -> None:
        self._suggestion = suggestion
        self._error = error
        self.asked: str | None = None

    def classify(
        self, question: str, history: list[dict[str, str]] | None = None
    ) -> str:
        if self._error is not None:
            raise self._error
        return "command"

    def suggest(
        self, question: str, history: list[dict[str, str]] | None = None
    ) -> list[CommandSuggestion]:
        self.asked = question
        if self._error is not None:
            raise self._error
        assert self._suggestion is not None
        return [self._suggestion]

    def converse(
        self, question: str, history: list[dict[str, str]] | None = None
    ) -> str:  # pragma: no cover - command-only stub
        raise AssertionError("converse should not be reached on the command path")


class _ScriptedClient:
    """Model client that answers each turn from a scripted list of suggestions.

    Every call records the ``history`` it was handed so tests can assert that a
    follow-up turn carried the prior turns.
    """

    def __init__(self, suggestions: list[CommandSuggestion]) -> None:
        self._suggestions = list(suggestions)
        self.calls: list[list[dict[str, str]]] = []

    def suggest(
        self, question: str, history: list[dict[str, str]] | None = None
    ) -> list[CommandSuggestion]:
        # Copy so later mutation of the live history can't rewrite the record.
        self.calls.append(list(history or []))
        return [self._suggestions.pop(0)]


def test_version_prints_name_and_version(capsys: pytest.CaptureFixture[str]) -> None:
    """``tux --version`` prints ``tux <version>`` to stdout and exits with status 0."""
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == f"tux {__version__}"
    assert captured.err == ""


def test_version_matches_packaging_metadata() -> None:
    """The installed package metadata agrees with ``tux.__version__``."""
    from importlib.metadata import version

    assert version("tux") == __version__


def test_help_exits_zero_with_branded_help(capsys: pytest.CaptureFixture[str]) -> None:
    """``tux --help`` prints description, usage, options, and an example to stdout, exit 0."""
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    out = captured.out
    assert "GNOME desktop AI chatbot" in out
    assert "usage: tux" in out
    assert "--version" in out
    assert "ps aux --sort=-%mem | head -10" in out


def test_short_help_matches_long_help(capsys: pytest.CaptureFixture[str]) -> None:
    """``tux -h`` produces output identical to ``tux --help`` and exits 0."""
    with pytest.raises(SystemExit) as excinfo:
        main(["-h"])
    assert excinfo.value.code == 0
    short_help = capsys.readouterr().out

    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    long_help = capsys.readouterr().out

    assert short_help == long_help


def test_no_arguments_prints_help_and_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    """``tux`` with no arguments prints the same help to stdout and returns 0."""
    assert main([]) == 0
    no_args_out = capsys.readouterr().out

    with pytest.raises(SystemExit):
        main(["--help"])
    help_out = capsys.readouterr().out

    assert no_args_out == help_out
    assert "not implemented yet" not in no_args_out


def test_help_mentions_ask_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    """``tux --help`` advertises the ``ask`` subcommand."""
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    assert "ask" in capsys.readouterr().out


def test_help_points_at_config_help(capsys: pytest.CaptureFixture[str]) -> None:
    """``tux --help`` lists ``config`` and directs the reader to ``config --help``."""
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "config" in out
    assert "config --help" in out
    # The top-level description and example epilog are not crowded out.
    assert "GNOME desktop AI chatbot" in out
    assert "ps aux --sort=-%mem | head -10" in out


def test_ask_prints_title_command_and_description_plainly(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``tux ask`` (piped) prints title, command, and description as plain text, exit 0."""
    suggestion = CommandSuggestion(
        title="Listing memory hogs",
        command="ps aux --sort=-%mem | head -10",
        description="lists the top ten processes ranked by memory usage",
    )
    client = _StubClient(suggestion=suggestion)
    assert main(["ask", "what uses the most memory?"], client=client) == 0
    captured = capsys.readouterr()
    out = captured.out
    assert suggestion.title in out
    assert f"command: {suggestion.command}" in out
    assert f"description: {suggestion.description}" in out
    # No markdown fences, and the piped fallback emits no ANSI escape sequences.
    assert "`" not in out
    assert "\x1b" not in out
    assert captured.err == ""
    assert client.asked == "what uses the most memory?"


def test_ask_reports_unreachable_endpoint_without_traceback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``tux ask`` exits non-zero with a clear message when the model call fails."""
    client = _StubClient(error=ModelClientError("could not reach the model endpoint"))
    assert main(["ask", "anything"], client=client) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "could not reach the model endpoint" in captured.err
    assert "Traceback" not in captured.err


def test_ask_builds_client_from_config_when_none_given(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no client injected, ``ask`` builds one from the config file."""
    captured_question: dict[str, str] = {}

    def fake_from_config() -> ModelClient:
        class _Recorder:
            def classify(
                self, question: str, history: list[dict[str, str]] | None = None
            ) -> str:
                return "command"

            def suggest(
                self, question: str, history: list[dict[str, str]] | None = None
            ) -> list[CommandSuggestion]:
                captured_question["q"] = question
                return [
                    CommandSuggestion(
                        title="Listing files", command="ls", description="lists files"
                    )
                ]

        return _Recorder()  # type: ignore[return-value]

    monkeypatch.setattr(ModelClient, "from_config", staticmethod(fake_from_config))
    assert main(["ask", "list files"]) == 0
    assert captured_question["q"] == "list files"


class _RoutingClient:
    """Fake client that routes each turn by a per-question type map.

    Every ``classify``/``suggest``/``converse`` call records the ``history`` it
    received so tests can assert which path a turn took and what context it
    carried across invocations.
    """

    def __init__(
        self,
        kinds: dict[str, str],
        commands: dict[str, CommandSuggestion] | None = None,
        chats: dict[str, str] | None = None,
    ) -> None:
        self._kinds = kinds
        self._commands = commands or {}
        self._chats = chats or {}
        self.classify_calls: list[tuple[str, list[dict[str, str]]]] = []
        self.suggest_calls: list[tuple[str, list[dict[str, str]]]] = []
        self.converse_calls: list[tuple[str, list[dict[str, str]]]] = []

    def classify(
        self, question: str, history: list[dict[str, str]] | None = None
    ) -> str:
        self.classify_calls.append((question, list(history or [])))
        return self._kinds.get(question, "command")

    def suggest(
        self, question: str, history: list[dict[str, str]] | None = None
    ) -> list[CommandSuggestion]:
        self.suggest_calls.append((question, list(history or [])))
        return [self._commands[question]]

    def converse(
        self, question: str, history: list[dict[str, str]] | None = None
    ) -> str:
        self.converse_calls.append((question, list(history or [])))
        return self._chats[question]


def test_ask_carries_prior_turn_across_invocations(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A second ``run_ask`` reads the thread the first wrote; its request carries it."""
    monkeypatch.setattr("tux.cli.os.getppid", lambda: 4242)
    client = _RoutingClient(
        kinds={"create foo.txt": "command", "now delete it": "command"},
        commands={
            "create foo.txt": CommandSuggestion(
                "Creating foo.txt", "touch foo.txt", "creates foo.txt"
            ),
            "now delete it": CommandSuggestion(
                "Removing foo.txt", "rm foo.txt", "removes foo.txt"
            ),
        },
    )
    assert main(["ask", "create foo.txt"], client=client) == 0
    assert main(["ask", "now delete it"], client=client) == 0

    # The first turn started cold; the follow-up's request carried the first turn
    # read back from disk between the two separate processes.
    assert client.suggest_calls[0][1] == []
    follow_up_history = client.suggest_calls[1][1]
    assert {"role": "user", "content": "create foo.txt"} in follow_up_history
    assert any(
        turn["role"] == "assistant" and "touch foo.txt" in turn["content"]
        for turn in follow_up_history
    )


def test_ask_keys_thread_by_parent_shell_pid(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A turn from a different shell PID does not see another terminal's thread."""
    client = _RoutingClient(
        kinds={"first": "command", "second": "command"},
        commands={
            "first": CommandSuggestion("Listing files", "ls", "lists files"),
            "second": CommandSuggestion("Printing cwd", "pwd", "prints cwd"),
        },
    )
    monkeypatch.setattr("tux.cli.os.getppid", lambda: 111)
    assert main(["ask", "first"], client=client) == 0
    # A different terminal (different shell PID) continues its own, empty thread.
    monkeypatch.setattr("tux.cli.os.getppid", lambda: 222)
    assert main(["ask", "second"], client=client) == 0
    assert client.suggest_calls[1][1] == []


def test_ask_routes_command_to_suggest_with_schema(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A command request takes the structured ``suggest`` path, not ``converse``."""
    monkeypatch.setattr("tux.cli.os.getppid", lambda: 1)
    client = _RoutingClient(
        kinds={"delete temp files": "command"},
        commands={
            "delete temp files": CommandSuggestion(
                "Removing temp files", "rm -rf /tmp/x", "removes x"
            )
        },
    )
    assert main(["ask", "delete temp files"], client=client) == 0
    assert [q for q, _ in client.suggest_calls] == ["delete temp files"]
    assert client.converse_calls == []
    out = capsys.readouterr().out
    assert "rm -rf /tmp/x" in out
    assert "removes x" in out


def test_ask_routes_conversational_to_converse_as_prose(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A conversational follow-up takes the free-form path and prints prose."""
    monkeypatch.setattr("tux.cli.os.getppid", lambda: 7)
    client = _RoutingClient(
        kinds={
            "how to check permissions?": "command",
            "what did I ask about last time?": "chat",
        },
        commands={
            "how to check permissions?": CommandSuggestion(
                "Checking permissions", "ls -l", "shows perms"
            ),
        },
        chats={
            "what did I ask about last time?": "You asked how to check permissions.",
        },
    )
    assert main(["ask", "how to check permissions?"], client=client) == 0
    assert main(["ask", "what did I ask about last time?"], client=client) == 0

    # The follow-up was answered as prose, carrying the earlier turn as context.
    assert [q for q, _ in client.converse_calls] == ["what did I ask about last time?"]
    follow_up_history = client.converse_calls[0][1]
    assert {"role": "user", "content": "how to check permissions?"} in follow_up_history
    out = capsys.readouterr().out
    assert "You asked how to check permissions." in out


def test_ask_new_discards_prior_thread(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``tux ask --new`` starts fresh: the turn carries no prior context."""
    monkeypatch.setattr("tux.cli.os.getppid", lambda: 9)
    client = _RoutingClient(
        kinds={"first": "command", "second": "command"},
        commands={
            "first": CommandSuggestion("Listing files", "ls", "lists files"),
            "second": CommandSuggestion("Printing cwd", "pwd", "prints cwd"),
        },
    )
    assert main(["ask", "first"], client=client) == 0
    assert main(["ask", "--new", "second"], client=client) == 0
    # The --new turn ignored the saved first turn.
    assert client.suggest_calls[1][1] == []
    # Subsequent calls build on the new thread, not the discarded one.
    saved = json.loads(thread_path(9).read_text(encoding="utf-8"))
    assert {"role": "user", "content": "second"} in saved["history"]
    assert all(turn["content"] != "first" for turn in saved["history"])


def test_ask_recovers_from_corrupt_thread_file(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A corrupt thread file degrades to a fresh thread instead of crashing."""
    monkeypatch.setattr("tux.cli.os.getppid", lambda: 55)
    path = thread_path(55)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ this is not valid json", encoding="utf-8")
    client = _RoutingClient(
        kinds={"hello": "command"},
        commands={"hello": CommandSuggestion("Printing hi", "echo hi", "prints hi")},
    )
    assert main(["ask", "hello"], client=client) == 0
    assert client.suggest_calls[0][1] == []
    assert "Traceback" not in capsys.readouterr().err


def test_ask_failed_turn_leaves_thread_uncorrupted(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """When a later turn fails, the previously saved thread is left intact."""
    monkeypatch.setattr("tux.cli.os.getppid", lambda: 88)

    class _FailSecond:
        def __init__(self) -> None:
            self._first = True

        def classify(self, question: str, history=None) -> str:
            if self._first:
                return "command"
            raise ModelClientError("could not reach the model endpoint")

        def suggest(self, question: str, history=None) -> list[CommandSuggestion]:
            self._first = False
            return [CommandSuggestion("Creating foo.txt", "touch foo.txt", "creates foo.txt")]

        def converse(self, question: str, history=None) -> str:  # pragma: no cover
            raise AssertionError("not reached")

    client = _FailSecond()
    assert main(["ask", "create foo.txt"], client=client) == 0
    assert main(["ask", "boom"], client=client) == 1
    err = capsys.readouterr().err
    assert "could not reach the model endpoint" in err
    assert "Traceback" not in err
    # The first turn is still the only thing on disk; the failed turn saved nothing.
    saved = json.loads(thread_path(88).read_text(encoding="utf-8"))
    assert {"role": "user", "content": "create foo.txt"} in saved["history"]
    assert all(turn["content"] != "boom" for turn in saved["history"])


def _feed_input(monkeypatch: pytest.MonkeyPatch, lines: list[str]) -> None:
    """Make ``input()`` yield each scripted line, then raise ``EOFError``."""
    supply = iter(lines)

    def fake_input(prompt: str = "") -> str:
        print(prompt, end="")
        try:
            return next(supply)
        except StopIteration:
            raise EOFError

    monkeypatch.setattr("builtins.input", fake_input)


def test_ask_without_question_starts_interactive_session(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``tux ask`` with no question opens a session that answers a typed question."""
    suggestion = CommandSuggestion(
        title="Listing by size", command="ls -laS", description="lists by size"
    )
    client = _ScriptedClient([suggestion])
    _feed_input(monkeypatch, ["biggest files?"])
    assert main(["ask"], client=client) == 0
    out = capsys.readouterr().out
    assert "ls -laS" in out
    assert "lists by size" in out


def test_session_follow_up_carries_prior_turns(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A follow-up turn's request includes the earlier question and its proposal."""
    first = CommandSuggestion(
        title="Creating foo.txt", command="touch foo.txt", description="creates foo.txt"
    )
    second = CommandSuggestion(
        title="Removing foo.txt", command="rm foo.txt", description="removes foo.txt"
    )
    client = _ScriptedClient([first, second])
    _feed_input(monkeypatch, ["create foo.txt", "now delete it"])
    assert main(["ask"], client=client) == 0

    # The first turn has no prior context; the follow-up carries the first turn.
    assert client.calls[0] == []
    follow_up_history = client.calls[1]
    assert {"role": "user", "content": "create foo.txt"} in follow_up_history
    assert any(
        turn["role"] == "assistant" and "touch foo.txt" in turn["content"]
        for turn in follow_up_history
    )


def test_session_ends_cleanly_on_eof(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An immediate EOF ends the session with status 0 and no traceback."""
    client = _ScriptedClient([])
    _feed_input(monkeypatch, [])
    assert main(["ask"], client=client) == 0
    assert "Traceback" not in capsys.readouterr().err


def test_session_exit_word_ends_session(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Typing an exit word ends the session with status 0 without asking the model."""
    client = _ScriptedClient([])
    _feed_input(monkeypatch, ["exit", "this is never reached"])
    assert main(["ask"], client=client) == 0
    assert client.calls == []


def test_session_blank_line_does_not_crash(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An empty input line is skipped without crashing or calling the model."""
    suggestion = CommandSuggestion(
        title="Printing cwd", command="pwd", description="prints the cwd"
    )
    client = _ScriptedClient([suggestion])
    _feed_input(monkeypatch, ["", "   ", "where am I?"])
    assert main(["ask"], client=client) == 0
    # Only the real question reached the model.
    assert len(client.calls) == 1
    assert "pwd" in capsys.readouterr().out


def test_session_mid_failure_preserves_context(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A mid-session error is reported cleanly and leaves earlier context intact."""

    class _FlakyClient:
        def __init__(self) -> None:
            self.calls: list[list[dict[str, str]]] = []
            self._answers = [
                CommandSuggestion(
                    title="Creating foo.txt",
                    command="touch foo.txt",
                    description="creates it",
                ),
                ModelClientError("could not reach the model endpoint"),
                CommandSuggestion(
                    title="Removing foo.txt", command="rm foo.txt", description="removes it"
                ),
            ]

        def suggest(
            self, question: str, history: list[dict[str, str]] | None = None
        ) -> list[CommandSuggestion]:
            self.calls.append(list(history or []))
            answer = self._answers.pop(0)
            if isinstance(answer, ModelClientError):
                raise answer
            return [answer]

    client = _FlakyClient()
    _feed_input(monkeypatch, ["create foo.txt", "boom", "now delete it"])
    assert main(["ask"], client=client) == 0
    captured = capsys.readouterr()
    assert "could not reach the model endpoint" in captured.err
    assert "Traceback" not in captured.err
    # The failed middle turn was not recorded, so the third turn still has only
    # the successful first turn as context.
    third_turn_history = client.calls[2]
    assert {"role": "user", "content": "create foo.txt"} in third_turn_history
    assert all(turn["content"] != "boom" for turn in third_turn_history)


def test_ask_help_describes_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    """``tux ask --help`` describes the subcommand and its question argument, exit 0."""
    with pytest.raises(SystemExit) as excinfo:
        main(["ask", "--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "question" in out
    assert "Ask tux" in out


def test_config_help_mentions_config_file_path(
    config_home, capsys: pytest.CaptureFixture[str]
) -> None:
    """``tux config --help`` states where the config file lives and lists its actions."""
    from tux.config import config_path

    with pytest.raises(SystemExit) as excinfo:
        main(["config", "--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert str(config_path()) in out
    # The existing actions remain documented.
    assert "show" in out and "path" in out and "set" in out


@pytest.fixture
def config_home(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Redirect the config path to a temp dir so tests never touch the real home."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    return tmp_path


def test_config_path_prints_absolute_path(
    config_home, capsys: pytest.CaptureFixture[str]
) -> None:
    """``tux config path`` prints the config-file path even when no file exists."""
    expected = config_home / "tux" / "config.toml"
    assert main(["config", "path"]) == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == str(expected)
    assert not expected.exists()


def test_config_show_marks_defaults_when_no_file(
    config_home, capsys: pytest.CaptureFixture[str]
) -> None:
    """``tux config show`` reports both values as coming from the built-in default."""
    assert main(["config", "show"]) == 0
    out = capsys.readouterr().out
    assert "endpoint" in out and "model" in out
    assert "default" in out
    assert "config" not in out


def test_config_set_then_show_and_ask_reflect_value(
    config_home, capsys: pytest.CaptureFixture[str]
) -> None:
    """A ``set`` value is persisted and reported by ``show`` as config-sourced."""
    assert main(["config", "set", "endpoint", "http://10.0.0.9:8080"]) == 0
    capsys.readouterr()
    assert main(["config", "show"]) == 0
    out = capsys.readouterr().out
    assert "http://10.0.0.9:8080" in out
    assert "config" in out


def test_config_set_unknown_key_fails_without_writing(
    config_home, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unknown key is rejected with a usage error, non-zero exit, and no file."""
    with pytest.raises(SystemExit) as excinfo:
        main(["config", "set", "timeout", "30"])
    assert excinfo.value.code != 0
    err = capsys.readouterr().err
    assert "endpoint" in err and "model" in err
    assert not (config_home / "tux" / "config.toml").exists()


def test_config_set_missing_value_errors(
    config_home, capsys: pytest.CaptureFixture[str]
) -> None:
    """``tux config set endpoint`` with no value is a usage error, like ``ask``."""
    with pytest.raises(SystemExit) as excinfo:
        main(["config", "set", "endpoint"])
    assert excinfo.value.code != 0
    assert "usage:" in capsys.readouterr().err


def test_config_unknown_action_errors(
    config_home, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unknown ``config`` action produces a usage error and non-zero exit."""
    with pytest.raises(SystemExit) as excinfo:
        main(["config", "bogus"])
    assert excinfo.value.code != 0
    assert "usage:" in capsys.readouterr().err


def test_config_with_no_action_errors(
    config_home, capsys: pytest.CaptureFixture[str]
) -> None:
    """``tux config`` with no action is a usage error naming the required action."""
    with pytest.raises(SystemExit) as excinfo:
        main(["config"])
    assert excinfo.value.code != 0
    assert "usage:" in capsys.readouterr().err


def test_ask_reports_malformed_config_without_traceback(
    config_home, capsys: pytest.CaptureFixture[str]
) -> None:
    """``tux ask`` with a malformed config exits non-zero with a clean message."""
    config = config_home / "tux" / "config.toml"
    config.parent.mkdir(parents=True)
    config.write_text("this is = not = valid toml\n")
    assert main(["ask", "anything"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert str(config) in captured.err
    assert "Traceback" not in captured.err


def test_ask_run_executes_command_logs_it_and_returns_status(
    interactive, capsys: pytest.CaptureFixture[str]
) -> None:
    """Choosing run executes the command, logs the run, and exits with its status."""
    suggestion = CommandSuggestion(
        title="Listing by size", command="ls -laS", description="lists by size"
    )
    runner = _FakeRunner(status=0)
    status = main(
        ["ask", "biggest files?"],
        client=_StubClient(suggestion),
        runner=runner,
        chooser=run_choice,
    )
    assert status == 0
    # The chosen command was executed exactly once.
    assert runner.commands == ["ls -laS"]
    # The run was appended to the log with its command and exit status.
    fields = log_path().read_text(encoding="utf-8").strip().split("\t")
    assert fields[1] == "0"
    assert fields[2] == "ls -laS"


def test_ask_run_failure_propagates_exit_status(interactive) -> None:
    """A run command's non-zero status becomes tux's exit status and is logged."""
    suggestion = CommandSuggestion(
        title="Failing on purpose", command="false", description="always fails"
    )
    runner = _FakeRunner(status=3)
    status = main(
        ["ask", "fail please"],
        client=_StubClient(suggestion),
        runner=runner,
        chooser=run_choice,
    )
    assert status == 3
    assert "\t3\t" in log_path().read_text(encoding="utf-8")


def test_ask_dismiss_runs_nothing_and_writes_no_log(interactive) -> None:
    """Choosing dismiss runs nothing, writes no log entry, and exits without error."""
    suggestion = CommandSuggestion(
        title="Wiping everything", command="rm -rf /", description="dangerous"
    )
    runner = _FakeRunner()
    status = main(
        ["ask", "clean up"],
        client=_StubClient(suggestion),
        runner=runner,
        chooser=dismiss_choice,
    )
    assert status == 0
    assert runner.commands == []
    assert not log_path().exists()


def test_ask_nothing_runs_before_the_user_chooses_run(interactive) -> None:
    """The propose-only guarantee holds: nothing is executed before the choice."""
    suggestion = CommandSuggestion(
        title="Printing hi", command="echo hi", description="prints hi"
    )
    runner = _FakeRunner()

    def chooser(options) -> int:
        # The choice is made here; nothing may have run yet at this point.
        assert runner.commands == []
        return RUN_CHOICE

    status = main(
        ["ask", "say hi"],
        client=_StubClient(suggestion),
        runner=runner,
        chooser=chooser,
    )
    assert status == 0
    assert runner.commands == ["echo hi"]


def test_ask_non_interactive_prints_proposal_and_runs_nothing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A non-interactive session prints the proposal only and never runs anything."""
    suggestion = CommandSuggestion(
        title="Showing load", command="uptime", description="shows load"
    )
    runner = _FakeRunner()

    def chooser(options) -> int:  # pragma: no cover - no menu when piped
        raise AssertionError("no menu should appear in a non-interactive session")

    # No `interactive` fixture: under pytest stdin/stdout are not TTYs.
    status = main(
        ["ask", "load?"],
        client=_StubClient(suggestion),
        runner=runner,
        chooser=chooser,
    )
    assert status == 0
    assert runner.commands == []
    assert not log_path().exists()
    # The print-only fallback is plain labelled text: no frame, no ANSI, no menu.
    out = capsys.readouterr().out
    assert out == "Showing load\ncommand: uptime\ndescription: shows load\n"
    assert "\x1b" not in out


def test_ask_conversational_turn_has_no_run_menu(
    interactive, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A conversational reply is prose and never reaches the run/dismiss menu."""
    monkeypatch.setattr("tux.cli.os.getppid", lambda: 4321)
    runner = _FakeRunner()

    def chooser(options) -> int:  # pragma: no cover - prose offers no menu
        raise AssertionError("a conversational reply must not be staged")

    client = _RoutingClient(
        kinds={"just chatting": "chat"},
        chats={"just chatting": "Hi there, happy to help."},
    )
    status = main(
        ["ask", "just chatting"], client=client, runner=runner, chooser=chooser
    )
    assert status == 0
    assert runner.commands == []
    assert "Hi there, happy to help." in capsys.readouterr().out


def test_session_run_executes_and_logs(
    interactive, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The run/dismiss surface also appears in the interactive multi-turn session."""
    suggestion = CommandSuggestion(
        title="Printing cwd", command="pwd", description="prints cwd"
    )
    runner = _FakeRunner()
    _feed_input(monkeypatch, ["where am I?"])
    status = main(
        ["ask"], client=_ScriptedClient([suggestion]), runner=runner, chooser=run_choice
    )
    assert status == 0
    assert runner.commands == ["pwd"]
    assert "\tpwd\n" in log_path().read_text(encoding="utf-8")


def test_ask_interactive_proposal_is_framed_and_styled(
    interactive, capsys: pytest.CaptureFixture[str]
) -> None:
    """An interactive command proposal is a framed, styled title/command/description block."""
    suggestion = CommandSuggestion(
        title="Listing hidden files",
        command="ls -a",
        description="lists all entries including dotfiles",
    )
    main(
        ["ask", "hidden files?"],
        client=_StubClient(suggestion),
        runner=_FakeRunner(),
        chooser=dismiss_choice,
    )
    out = capsys.readouterr().out
    lines = out.splitlines()
    # Framed: exactly two horizontal rules (above and below the proposal).
    rules = [line for line in lines if line and set(line) == {"─"}]
    assert len(rules) == 2
    # Breathing room: a blank line opens the block.
    assert out.startswith("\n")
    # The three labelled fields are present.
    assert "Listing hidden files" in out
    assert "command:" in out and "ls -a" in out
    assert "description:" in out and "lists all entries including dotfiles" in out
    # Styling is interactive-only, and the title and command carry distinct styles.
    assert "\x1b[1;33m" in out  # title
    assert "\x1b[1;36m" in out  # command


def test_destructive_proposal_warns_in_styled_block(
    interactive, capsys: pytest.CaptureFixture[str]
) -> None:
    """A flagged command shows a distinct bold-red warning before the menu, exit unchanged."""
    suggestion = CommandSuggestion(
        title="Wiping a directory",
        command="rm -rf /tmp/work",
        description="removes the work directory",
    )
    status = main(
        ["ask", "delete the work dir"],
        client=_StubClient(suggestion),
        runner=_FakeRunner(),
        chooser=dismiss_choice,
    )
    assert status == 0
    out = capsys.readouterr().out
    # The warning is present, names the risk and a plain-English reason, and is
    # styled distinctly (bold red) from the title/command lines.
    assert "potentially destructive" in out
    assert "deletes" in out
    assert "\x1b[1;31m" in out  # bold red, set apart from title/command styles
    # It is read before the run/dismiss menu: the warning precedes the trailing
    # blank line that closes the block.
    warning_index = out.index("potentially destructive")
    assert warning_index < len(out)


def test_destructive_proposal_warns_in_plain_fallback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The piped fallback carries the same warning as plain text and still runs nothing."""
    suggestion = CommandSuggestion(
        title="Formatting a disk",
        command="mkfs.ext4 /dev/sdb1",
        description="creates a new filesystem",
    )
    runner = _FakeRunner()
    # No `interactive` fixture: under pytest stdin/stdout are not TTYs.
    status = main(
        ["ask", "format the disk"],
        client=_StubClient(suggestion),
        runner=runner,
        chooser=dismiss_choice,
    )
    assert status == 0
    assert runner.commands == []
    out = capsys.readouterr().out
    assert "potentially destructive" in out
    assert "formats" in out
    # Plain fallback: no ANSI escape sequences even for the warning line.
    assert "\x1b" not in out


def test_benign_proposal_shows_no_warning(
    interactive, capsys: pytest.CaptureFixture[str]
) -> None:
    """A non-destructive command renders exactly as before, with no warning."""
    suggestion = CommandSuggestion(
        title="Listing hidden files",
        command="ls -a",
        description="lists all entries including dotfiles",
    )
    main(
        ["ask", "hidden files?"],
        client=_StubClient(suggestion),
        runner=_FakeRunner(),
        chooser=dismiss_choice,
    )
    out = capsys.readouterr().out
    assert "potentially destructive" not in out
    assert "\x1b[1;31m" not in out  # no warning styling


def test_flagged_command_still_runs_when_chosen(
    interactive, capsys: pytest.CaptureFixture[str]
) -> None:
    """A flagged command the user chooses to run is still executed and logged."""
    suggestion = CommandSuggestion(
        title="Wiping a directory",
        command="rm -rf /tmp/work",
        description="removes the work directory",
    )
    runner = _FakeRunner(status=0)
    status = main(
        ["ask", "delete the work dir"],
        client=_StubClient(suggestion),
        runner=runner,
        chooser=run_choice,
    )
    assert status == 0
    assert runner.commands == ["rm -rf /tmp/work"]
    assert "rm -rf /tmp/work" in log_path().read_text(encoding="utf-8")


class _PlanClient:
    """Answers the command path from a scripted list of plans, one per suggest call.

    Each ``suggest`` records the ``(question, history)`` it received so a test can
    assert that an in-walk re-plan carried the running thread (the prior question,
    the plan, and any fed-back output or clarification).
    """

    def __init__(self, plans: list[list[CommandSuggestion]]) -> None:
        self._plans = list(plans)
        self.suggest_calls: list[tuple[str, list[dict[str, str]]]] = []

    def classify(
        self, question: str, history: list[dict[str, str]] | None = None
    ) -> str:
        return "command"

    def suggest(
        self, question: str, history: list[dict[str, str]] | None = None
    ) -> list[CommandSuggestion]:
        self.suggest_calls.append((question, list(history or [])))
        return self._plans.pop(0)

    def converse(
        self, question: str, history: list[dict[str, str]] | None = None
    ) -> str:  # pragma: no cover - command-only stub
        raise AssertionError("converse should not be reached on the command path")


def _seq_chooser(choices: list[int]):
    """A chooser that returns each scripted choice index in turn."""
    supply = iter(choices)

    def chooser(options) -> int:
        return next(supply)

    return chooser


class _Reader:
    """A clarify reader that hands out scripted lines and records its prompts."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = iter(lines)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return next(self._lines)


def test_walk_shows_overview_up_front_then_expands_active_step(
    interactive, capsys: pytest.CaptureFixture[str]
) -> None:
    """The whole plan is shown as a numbered overview before the first step is walked."""
    plan = [
        CommandSuggestion("Printing cwd", "pwd", "shows the working directory"),
        CommandSuggestion("Listing it", "ls", "lists the directory"),
    ]
    main(
        ["ask", "look around"],
        client=_PlanClient([plan, [plan[1]]]),
        runner=_FakeRunner(output="/home/me"),
        chooser=_seq_chooser([RUN_CHOICE, DISMISS_CHOICE]),
    )
    out = capsys.readouterr().out
    # Both step titles appear as a de-emphasised numbered overview...
    assert "1. Printing cwd" in out
    assert "2. Listing it" in out
    # ...and the whole overview is printed before the first step's framed command
    # (the styled command line carries ANSI codes, so match the bare command).
    assert out.index("2. Listing it") < out.index("pwd")


def test_walk_runs_one_step_at_a_time_and_logs_each(
    interactive, capsys: pytest.CaptureFixture[str]
) -> None:
    """Running each step executes its single command in order and logs both runs."""
    plan = [
        CommandSuggestion("Printing cwd", "pwd", "shows the working directory"),
        CommandSuggestion("Listing it", "ls", "lists the directory"),
    ]
    runner = _FakeRunner(output="")
    status = main(
        ["ask", "look around"],
        client=_PlanClient([plan, [plan[1]]]),
        runner=runner,
        chooser=_seq_chooser([RUN_CHOICE, RUN_CHOICE]),
    )
    assert status == 0
    assert runner.commands == ["pwd", "ls"]
    log = log_path().read_text(encoding="utf-8").splitlines()
    assert log[0].endswith("pwd")
    assert log[1].endswith("ls")


def test_walk_feeds_run_output_to_replan_remaining_steps(
    interactive, capsys: pytest.CaptureFixture[str]
) -> None:
    """A discovery step's output is fed back so a later placeholder is shown resolved."""
    plan = [
        CommandSuggestion("Printing cwd", "pwd", "shows the working directory"),
        CommandSuggestion("Listing it", "ls {cwd}", "lists the discovered directory"),
    ]
    resolved = [CommandSuggestion("Listing it", "ls /home/me/work", "lists it")]
    client = _PlanClient([plan, resolved])
    runner = _FakeRunner(output="/home/me/work")
    main(
        ["ask", "list my current dir"],
        client=client,
        runner=runner,
        chooser=_seq_chooser([RUN_CHOICE, RUN_CHOICE]),
    )
    # The placeholder command is replaced by the resolved one, with no copy/paste.
    assert runner.commands == ["pwd", "ls /home/me/work"]
    out = capsys.readouterr().out
    assert "ls /home/me/work" in out
    assert "ls {cwd}" not in out
    # The re-plan request carried the captured output as context.
    replan_question, _ = client.suggest_calls[1]
    assert "/home/me/work" in replan_question


def test_walk_captured_output_never_reaches_the_run_log(
    interactive, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run step's output feeds the model but is never written to the run log."""
    plan = [
        CommandSuggestion("Printing secret", "cat secret", "prints the secret"),
        CommandSuggestion("Using it", "echo done", "finishes up"),
    ]
    main(
        ["ask", "show the secret"],
        client=_PlanClient([plan, [plan[1]]]),
        runner=_FakeRunner(output="TOP_SECRET_VALUE"),
        chooser=_seq_chooser([RUN_CHOICE, RUN_CHOICE]),
    )
    log = log_path().read_text(encoding="utf-8")
    assert "cat secret" in log  # the command is logged...
    assert "TOP_SECRET_VALUE" not in log  # ...but its output never is


def test_walk_does_not_run_a_later_step_before_its_explicit_run(
    interactive, capsys: pytest.CaptureFixture[str]
) -> None:
    """Dismissing the second step leaves it unrun: no step runs ahead of its choice."""
    plan = [
        CommandSuggestion("Printing cwd", "pwd", "shows the working directory"),
        CommandSuggestion("Wiping it", "rm -rf .", "removes everything"),
    ]
    runner = _FakeRunner(output="/home/me")
    status = main(
        ["ask", "clean here"],
        client=_PlanClient([plan, [plan[1]]]),
        runner=runner,
        chooser=_seq_chooser([RUN_CHOICE, DISMISS_CHOICE]),
    )
    assert status == 0
    # Only the explicitly-run first step executed; the second never ran.
    assert runner.commands == ["pwd"]


def test_walk_dismiss_on_first_step_abandons_whole_plan(
    interactive, capsys: pytest.CaptureFixture[str]
) -> None:
    """Dismissing the first step runs nothing, logs nothing, and exits cleanly."""
    plan = [
        CommandSuggestion("Printing cwd", "pwd", "shows the working directory"),
        CommandSuggestion("Listing it", "ls", "lists the directory"),
    ]
    client = _PlanClient([plan])
    runner = _FakeRunner()
    status = main(
        ["ask", "look around"],
        client=client,
        runner=runner,
        chooser=dismiss_choice,
    )
    assert status == 0
    assert runner.commands == []
    assert not log_path().exists()
    # No re-plan happened; the model was asked only for the initial plan.
    assert len(client.suggest_calls) == 1


def test_walk_clarify_replans_remaining_steps_and_runs_nothing(
    interactive, capsys: pytest.CaptureFixture[str]
) -> None:
    """Clarify sends free text to the model, reprints the revised overview, runs nothing."""
    plan = [CommandSuggestion("Listing files", "ls", "lists files")]
    revised = [CommandSuggestion("Listing as root", "sudo ls", "lists as root")]
    client = _PlanClient([plan, revised])
    runner = _FakeRunner()
    reader = _Reader(["actually list as root"])
    status = main(
        ["ask", "list files"],
        client=client,
        runner=runner,
        chooser=_seq_chooser([CLARIFY_CHOICE, RUN_CHOICE]),
        reader=reader,
    )
    assert status == 0
    # The clarify text round-tripped to the model as a re-plan request...
    clarify_question, _ = client.suggest_calls[1]
    assert clarify_question == "actually list as root"
    # ...nothing ran as part of the clarify; only the post-clarify run executed.
    assert runner.commands == ["sudo ls"]
    # The reader was prompted, and the revised plan's overview was reprinted.
    assert reader.prompts == ["clarify: "]
    assert "1. Listing as root" in capsys.readouterr().out


def test_walk_clarify_abort_dismisses_without_running(
    interactive, capsys: pytest.CaptureFixture[str]
) -> None:
    """A blank clarify line aborts to dismiss: nothing runs and the walk ends."""
    plan = [CommandSuggestion("Listing files", "ls", "lists files")]
    client = _PlanClient([plan])
    runner = _FakeRunner()
    status = main(
        ["ask", "list files"],
        client=client,
        runner=runner,
        chooser=_seq_chooser([CLARIFY_CHOICE]),
        reader=_Reader([""]),
    )
    assert status == 0
    assert runner.commands == []
    # A blank clarify did not re-plan.
    assert len(client.suggest_calls) == 1


def test_walk_one_step_plan_is_walked_like_a_longer_one(
    interactive, capsys: pytest.CaptureFixture[str]
) -> None:
    """A one-step plan still shows an overview and runs without a trailing prompt."""
    plan = [CommandSuggestion("Listing by size", "ls -laS", "lists by size")]
    client = _PlanClient([plan])
    chooser_calls = {"n": 0}

    def chooser(options) -> int:
        chooser_calls["n"] += 1
        return RUN_CHOICE

    status = main(
        ["ask", "biggest files?"],
        client=client,
        runner=_FakeRunner(),
        chooser=chooser,
    )
    assert status == 0
    out = capsys.readouterr().out
    assert "1. Listing by size" in out  # overview shown even for one step
    # The single step is offered once; no extra prompt after the plan is exhausted.
    assert chooser_calls["n"] == 1
    # Running the last (only) step triggers no re-plan.
    assert len(client.suggest_calls) == 1


def test_walk_destructive_step_after_a_discovery_step_is_flagged(
    interactive, capsys: pytest.CaptureFixture[str]
) -> None:
    """A destructive action that is its own later step carries the per-step warning."""
    plan = [
        CommandSuggestion("Printing cwd", "pwd", "shows the working directory"),
        CommandSuggestion("Wiping work", "rm -rf /tmp/work", "removes the work dir"),
    ]
    main(
        ["ask", "clean the work dir"],
        client=_PlanClient([plan, [plan[1]]]),
        runner=_FakeRunner(output="/tmp"),
        chooser=_seq_chooser([RUN_CHOICE, DISMISS_CHOICE]),
    )
    out = capsys.readouterr().out
    # The destructive second step is flagged when the walk reaches it...
    assert "potentially destructive" in out
    assert "\x1b[1;31m" in out  # bold-red warning styling
    # ...and the warning rides with the rm step, after the benign pwd discovery.
    assert out.index("pwd") < out.index("potentially destructive")


def test_non_interactive_prints_every_step_plain_and_runs_nothing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A piped session prints all plan steps as plain text and runs/captures nothing."""
    plan = [
        CommandSuggestion("Printing cwd", "pwd", "shows the working directory"),
        CommandSuggestion("Listing it", "ls", "lists the directory"),
    ]
    runner = _FakeRunner()

    def chooser(options) -> int:  # pragma: no cover - no menu when piped
        raise AssertionError("no menu should appear in a non-interactive session")

    def reader(prompt: str) -> str:  # pragma: no cover - no clarify when piped
        raise AssertionError("no clarify prompt should appear in a piped session")

    status = main(
        ["ask", "look around"],
        client=_PlanClient([plan]),
        runner=runner,
        chooser=chooser,
        reader=reader,
    )
    assert status == 0
    assert runner.commands == []
    assert not log_path().exists()
    out = capsys.readouterr().out
    # Each step's single command is visible as plain labelled text, no ANSI.
    assert "command: pwd" in out
    assert "command: ls" in out
    assert "\x1b" not in out


# --- lite variant gating --------------------------------------------------


def test_lite_command_turn_is_a_single_proposal(
    lite, interactive, capsys: pytest.CaptureFixture[str]
) -> None:
    """In lite, a request the full variant would multi-step is one command only."""
    plan = [
        CommandSuggestion("Printing cwd", "pwd", "shows the working directory"),
        CommandSuggestion("Listing it", "ls", "lists the directory"),
    ]
    client = _PlanClient([plan])
    runner = _FakeRunner()
    status = main(
        ["ask", "look around"], client=client, runner=runner, chooser=run_choice
    )
    assert status == 0
    # Only the first command is proposed and run; the second step never surfaces.
    assert runner.commands == ["pwd"]
    out = capsys.readouterr().out
    assert "pwd" in out
    assert "ls" not in out
    # No multi-step overview numbering and no re-plan loop.
    assert "1. " not in out and "2. " not in out
    assert len(client.suggest_calls) == 1


def test_lite_command_proposal_offers_no_clarify(lite, interactive) -> None:
    """The lite per-turn menu is the safe-run floor only — dismiss/run, no clarify."""
    seen: list[tuple[str, ...]] = []

    def chooser(options) -> int:
        seen.append(tuple(options))
        return 0  # dismiss

    plan = [CommandSuggestion("Listing files", "ls -l", "lists files")]
    main(["ask", "list files"], client=_PlanClient([plan]), runner=_FakeRunner(),
         chooser=chooser)
    assert seen == [("Dismiss", "Run")]
    assert all("Clarify" not in options for options in seen)


def test_lite_command_non_interactive_prints_single_proposal(
    lite, capsys: pytest.CaptureFixture[str]
) -> None:
    """A piped lite command turn prints only the first command and runs nothing."""
    plan = [
        CommandSuggestion("Printing cwd", "pwd", "shows the working directory"),
        CommandSuggestion("Listing it", "ls", "lists the directory"),
    ]
    runner = _FakeRunner()

    def chooser(options) -> int:  # pragma: no cover - no menu when piped
        raise AssertionError("no menu should appear in a non-interactive session")

    status = main(
        ["ask", "look around"], client=_PlanClient([plan]), runner=runner, chooser=chooser
    )
    assert status == 0
    assert runner.commands == []
    out = capsys.readouterr().out
    assert "command: pwd" in out
    assert "command: ls" not in out
    assert "\x1b" not in out


def test_lite_run_executes_and_logs(lite, interactive) -> None:
    """The safe-run floor is unchanged in lite: a chosen command runs and is logged."""
    plan = [CommandSuggestion("Listing by size", "ls -laS", "lists by size")]
    runner = _FakeRunner(status=0)
    status = main(
        ["ask", "biggest files?"],
        client=_PlanClient([plan]),
        runner=runner,
        chooser=run_choice,
    )
    assert status == 0
    assert runner.commands == ["ls -laS"]
    assert "ls -laS" in log_path().read_text(encoding="utf-8")


def test_lite_destructive_command_still_flagged(
    lite, interactive, capsys: pytest.CaptureFixture[str]
) -> None:
    """Destructive-command flagging still fires on a lite single proposal."""
    plan = [CommandSuggestion("Wiping work", "rm -rf /tmp/work", "removes the work dir")]
    main(["ask", "wipe work"], client=_PlanClient([plan]), runner=_FakeRunner(),
         chooser=dismiss_choice)
    out = capsys.readouterr().out
    assert "potentially destructive" in out
    assert "\x1b[1;31m" in out  # bold-red warning, unchanged from the full variant


def test_lite_conversational_reply_ends_with_command_steer(
    lite, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A lite conversational turn answers in prose, then steers toward command lookup."""
    monkeypatch.setattr("tux.cli.os.getppid", lambda: 3030)
    client = _RoutingClient(
        kinds={"hello there": "chat"},
        chats={"hello there": "Hi, happy to help."},
    )
    assert main(["ask", "hello there"], client=client) == 0
    out = capsys.readouterr().out
    assert "Hi, happy to help." in out
    # The reply ends with a steer carrying a concrete `tux ask "…"` example.
    assert 'tux ask "' in out
    assert out.index("Hi, happy to help.") < out.index('tux ask "')
    # The per-shell thread is saved with the model's prose only; the cli-layer
    # steer is shown but never stored.
    saved = json.loads(thread_path(3030).read_text(encoding="utf-8"))
    assistant = [turn for turn in saved["history"] if turn["role"] == "assistant"]
    assert assistant and assistant[-1]["content"] == "Hi, happy to help."
    assert 'tux ask "' not in assistant[-1]["content"]


def test_full_conversational_reply_has_no_steer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With variant unset (full behavior), a conversational reply is unsteered."""
    monkeypatch.setattr("tux.cli.os.getppid", lambda: 3131)
    client = _RoutingClient(
        kinds={"hello there": "chat"},
        chats={"hello there": "Hi, happy to help."},
    )
    assert main(["ask", "hello there"], client=client) == 0
    out = capsys.readouterr().out
    assert "Hi, happy to help." in out
    assert 'tux ask "' not in out


def test_full_variant_keeps_multi_step_walk(
    config_home, interactive, capsys: pytest.CaptureFixture[str]
) -> None:
    """An explicit ``variant = "full"`` keeps the multi-step overview and walk."""
    from tux.config import set_value

    set_value("variant", "full")
    plan = [
        CommandSuggestion("Printing cwd", "pwd", "shows the working directory"),
        CommandSuggestion("Listing it", "ls", "lists the directory"),
    ]
    main(
        ["ask", "look around"],
        client=_PlanClient([plan, [plan[1]]]),
        runner=_FakeRunner(output=""),
        chooser=_seq_chooser([RUN_CHOICE, DISMISS_CHOICE]),
    )
    out = capsys.readouterr().out
    assert "1. Printing cwd" in out and "2. Listing it" in out


def test_provision_reports_decision_and_config_target(
    config_home, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``tux provision`` prints the chosen tier, its signals, and the config target."""
    from tux.provision import FULL_TIER, OLLAMA_ENDPOINT, ProvisionResult, TierDecision

    result = ProvisionResult(
        tier=FULL_TIER,
        decision=TierDecision(FULL_TIER, ("NVIDIA GPU with 24576 MB VRAM",)),
        ollama_installed=True,
        model_pulled=True,
        model_deferred=False,
        bypassed=False,
        endpoint_reachable=True,
        endpoint=OLLAMA_ENDPOINT,
        model=FULL_TIER.model,
        variant=FULL_TIER.variant,
    )
    captured_kwargs: dict = {}

    def fake_provision(**kwargs):
        captured_kwargs.update(kwargs)
        return result

    monkeypatch.setattr("tux.cli.provision", fake_provision)
    assert main(["provision"]) == 0
    out = capsys.readouterr().out
    assert FULL_TIER.capability in out
    assert "NVIDIA GPU with 24576 MB VRAM" in out
    assert OLLAMA_ENDPOINT in out
    # A non-interactive test run defers prompting, never assuming consent.
    assert captured_kwargs["interactive"] is False
    assert captured_kwargs["assume_yes"] is False


def test_provision_yes_flag_preseeds_consent(
    config_home, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``tux provision --yes`` passes preseeded consent through to the brain."""
    from tux.provision import LOOKUP_TIER, OLLAMA_ENDPOINT, ProvisionResult, TierDecision

    captured_kwargs: dict = {}

    def fake_provision(**kwargs):
        captured_kwargs.update(kwargs)
        return ProvisionResult(
            tier=LOOKUP_TIER,
            decision=TierDecision(LOOKUP_TIER, ("no GPU detected",)),
            ollama_installed=False,
            model_pulled=True,
            model_deferred=False,
            bypassed=False,
            endpoint_reachable=True,
            endpoint=OLLAMA_ENDPOINT,
            model=LOOKUP_TIER.model,
            variant=LOOKUP_TIER.variant,
        )

    monkeypatch.setattr("tux.cli.provision", fake_provision)
    assert main(["provision", "--yes"]) == 0
    assert captured_kwargs["assume_yes"] is True


def test_provision_variant_flag_pins_tier(
    config_home, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``tux provision --variant lite`` passes the pin through to the brain."""
    from tux.provision import LOOKUP_TIER, OLLAMA_ENDPOINT, ProvisionResult, TierDecision

    captured_kwargs: dict = {}

    def fake_provision(**kwargs):
        captured_kwargs.update(kwargs)
        return ProvisionResult(
            tier=LOOKUP_TIER,
            decision=TierDecision(LOOKUP_TIER, ("variant pinned to 'lite'",)),
            ollama_installed=False,
            model_pulled=True,
            model_deferred=False,
            bypassed=False,
            endpoint_reachable=True,
            endpoint=OLLAMA_ENDPOINT,
            model=LOOKUP_TIER.model,
            variant=LOOKUP_TIER.variant,
        )

    monkeypatch.setattr("tux.cli.provision", fake_provision)
    assert main(["provision", "--variant", "lite", "--yes"]) == 0
    assert captured_kwargs["pin"] == "lite"


def test_provision_without_variant_does_not_pin(
    config_home, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A plain ``tux provision`` leaves the pin unset (hardware-probed tiering)."""
    from tux.provision import LOOKUP_TIER, OLLAMA_ENDPOINT, ProvisionResult, TierDecision

    captured_kwargs: dict = {}

    def fake_provision(**kwargs):
        captured_kwargs.update(kwargs)
        return ProvisionResult(
            tier=LOOKUP_TIER,
            decision=TierDecision(LOOKUP_TIER, ("no GPU detected",)),
            ollama_installed=False,
            model_pulled=True,
            model_deferred=False,
            bypassed=False,
            endpoint_reachable=True,
            endpoint=OLLAMA_ENDPOINT,
            model=LOOKUP_TIER.model,
            variant=LOOKUP_TIER.variant,
        )

    monkeypatch.setattr("tux.cli.provision", fake_provision)
    assert main(["provision"]) == 0
    assert captured_kwargs["pin"] is None


def test_provision_reports_failure_without_traceback(
    config_home, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A failing provisioning step exits non-zero with a clean message, no traceback."""
    def fake_provision(**kwargs):
        raise OSError("ollama installer could not be fetched")

    monkeypatch.setattr("tux.cli.provision", fake_provision)
    assert main(["provision"]) == 1
    err = capsys.readouterr().err
    assert "provisioning failed" in err
    assert "Traceback" not in err
