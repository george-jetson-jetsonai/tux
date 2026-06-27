"""Tests for the ``tux`` model client against canned SSE streams (no server)."""

import json

import pytest

from tux.client import (
    DEFAULT_ENDPOINT,
    DEFAULT_MODEL,
    ModelClient,
    ModelClientError,
)
from tux.modes.command import CommandSuggestion


def _sse(obj: dict) -> str:
    """Render a chat-completion chunk as an SSE ``data:`` line."""
    return f"data: {json.dumps(obj)}\n"


def _content_chunk(text: str) -> str:
    return _sse({"choices": [{"delta": {"content": text}}]})


def _finish_chunk(reason: str = "stop") -> str:
    return _sse({"choices": [{"delta": {}, "finish_reason": reason}]})


def _suggestion_stream(
    command: str, description: str, title: str = "Doing the task"
) -> list[str]:
    """A canned stream that emits a one-step plan object then a finish marker."""
    body = json.dumps(
        {"steps": [{"title": title, "command": command, "description": description}]}
    )
    return [
        _content_chunk(body[: len(body) // 2]),
        _content_chunk(body[len(body) // 2:]),
        _finish_chunk(),
    ]


def test_suggest_parses_streamed_structured_output() -> None:
    """``suggest`` reassembles streamed chunks into an ordered plan of suggestions."""
    chunks = _suggestion_stream(
        "ls -laS", "lists files by size, largest first", title="Listing by size"
    )

    def transport(url: str, payload: dict, timeout: float) -> list[str]:
        return chunks

    client = ModelClient(transport=transport)
    result = client.suggest("biggest files?")
    assert result == [
        CommandSuggestion(
            title="Listing by size",
            command="ls -laS",
            description="lists files by size, largest first",
        )
    ]


def test_suggest_parses_a_multi_step_plan() -> None:
    """``suggest`` returns each step in order for a multi-step plan."""
    body = json.dumps(
        {
            "steps": [
                {"title": "Printing cwd", "command": "pwd", "description": "shows cwd"},
                {"title": "Listing it", "command": "ls {cwd}", "description": "lists it"},
            ]
        }
    )

    def transport(url: str, payload: dict, timeout: float) -> list[str]:
        return [_content_chunk(body), _finish_chunk()]

    client = ModelClient(transport=transport)
    assert client.suggest("clean up here") == [
        CommandSuggestion("Printing cwd", "pwd", "shows cwd"),
        CommandSuggestion("Listing it", "ls {cwd}", "lists it"),
    ]


def test_suggest_sends_question_to_chat_completions() -> None:
    """The request targets ``/v1/chat/completions`` and carries the question + model."""
    seen: dict = {}

    def transport(url: str, payload: dict, timeout: float) -> list[str]:
        seen["url"] = url
        seen["payload"] = payload
        return _suggestion_stream("pwd", "prints the working directory")

    client = ModelClient(endpoint="http://example:8080", model="m.gguf", transport=transport)
    client.suggest("where am I?")
    assert seen["url"] == "http://example:8080/v1/chat/completions"
    assert seen["payload"]["model"] == "m.gguf"
    assert seen["payload"]["stream"] is True
    assert seen["payload"]["messages"][-1] == {"role": "user", "content": "where am I?"}
    assert seen["payload"]["response_format"]["type"] == "json_schema"


def test_suggest_includes_prior_turns_in_request() -> None:
    """A follow-up request carries the system prompt, the prior turns, then the new one."""
    seen: dict = {}

    def transport(url: str, payload: dict, timeout: float) -> list[str]:
        seen["payload"] = payload
        return _suggestion_stream("rm foo.txt", "removes foo.txt")

    history = [
        {"role": "user", "content": "create foo.txt"},
        {"role": "assistant", "content": '{"title": "Creating a file", '
         '"command": "touch foo.txt", "description": "creates foo.txt"}'},
    ]
    client = ModelClient(transport=transport)
    client.suggest("now delete it", history)

    messages = seen["payload"]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1:3] == history
    assert messages[-1] == {"role": "user", "content": "now delete it"}


def test_from_config_uses_defaults_when_file_absent(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no config file, the client targets the documented defaults."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    seen: dict = {}

    def transport(url: str, payload: dict, timeout: float) -> list[str]:
        seen["url"] = url
        seen["payload"] = payload
        return _suggestion_stream("ls", "lists files")

    client = ModelClient.from_config(transport=transport)
    client.suggest("list files")
    assert seen["url"] == f"{DEFAULT_ENDPOINT}/v1/chat/completions"
    assert seen["payload"]["model"] == DEFAULT_MODEL


def test_from_config_reads_overrides(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Config-file ``endpoint`` and ``model`` override the defaults."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    config = tmp_path / "tux" / "config.toml"
    config.parent.mkdir(parents=True)
    config.write_text('endpoint = "http://10.0.0.5:9000"\nmodel = "custom.gguf"\n')
    seen: dict = {}

    def transport(url: str, payload: dict, timeout: float) -> list[str]:
        seen["url"] = url
        seen["payload"] = payload
        return _suggestion_stream("ls", "lists files")

    client = ModelClient.from_config(transport=transport)
    client.suggest("list files")
    assert seen["url"] == "http://10.0.0.5:9000/v1/chat/completions"
    assert seen["payload"]["model"] == "custom.gguf"


def test_from_config_ignores_legacy_env_vars(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The retired ``TUX_ENDPOINT`` / ``TUX_MODEL`` env vars no longer have any effect."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("TUX_ENDPOINT", "http://should-be-ignored:1234")
    monkeypatch.setenv("TUX_MODEL", "ignored.gguf")
    seen: dict = {}

    def transport(url: str, payload: dict, timeout: float) -> list[str]:
        seen["url"] = url
        seen["payload"] = payload
        return _suggestion_stream("ls", "lists files")

    client = ModelClient.from_config(transport=transport)
    client.suggest("list files")
    assert seen["url"] == f"{DEFAULT_ENDPOINT}/v1/chat/completions"
    assert seen["payload"]["model"] == DEFAULT_MODEL


def test_suggest_raises_clear_error_on_network_failure() -> None:
    """A transport raising ``OSError`` becomes a clean ``ModelClientError``."""

    def transport(url: str, payload: dict, timeout: float):
        raise ConnectionRefusedError("connection refused")
        yield  # pragma: no cover - makes this a generator

    client = ModelClient(endpoint="http://unreachable:8080", transport=transport)
    with pytest.raises(ModelClientError) as excinfo:
        client.suggest("anything")
    assert "unreachable:8080" in str(excinfo.value)


def test_suggest_raises_on_unparseable_content() -> None:
    """Non-JSON model content surfaces as a ``ModelClientError``, not a crash."""

    def transport(url: str, payload: dict, timeout: float) -> list[str]:
        return [_content_chunk("not json at all"), _finish_chunk()]

    client = ModelClient(transport=transport)
    with pytest.raises(ModelClientError):
        client.suggest("anything")


def test_suggest_raises_on_missing_fields() -> None:
    """A well-formed JSON object missing required fields is rejected."""

    def transport(url: str, payload: dict, timeout: float) -> list[str]:
        body = json.dumps({"command": "ls"})
        return [_content_chunk(body), _finish_chunk()]

    client = ModelClient(transport=transport)
    with pytest.raises(ModelClientError):
        client.suggest("anything")


def _text_stream(text: str) -> list[str]:
    """A canned stream that emits ``text`` as one chunk then a finish marker."""
    return [_content_chunk(text), _finish_chunk()]


@pytest.mark.parametrize(
    ("reply", "expected"),
    [("COMMAND", "command"), ("CHAT", "chat"), ("chat please", "chat")],
)
def test_classify_routes_without_schema(reply: str, expected: str) -> None:
    """``classify`` maps the model's word to a route and sends no ``json_schema``."""
    seen: dict = {}

    def transport(url: str, payload: dict, timeout: float) -> list[str]:
        seen["payload"] = payload
        return _text_stream(reply)

    client = ModelClient(transport=transport)
    assert client.classify("do something") == expected
    # The routing call is plain chat: no structured-output constraint.
    assert "response_format" not in seen["payload"]


def test_converse_returns_prose_without_schema_and_carries_history() -> None:
    """``converse`` returns plain text, omits ``json_schema``, and includes history."""
    seen: dict = {}

    def transport(url: str, payload: dict, timeout: float) -> list[str]:
        seen["payload"] = payload
        return _text_stream("You asked about file permissions earlier.")

    history = [
        {"role": "user", "content": "how do I check permissions?"},
        {"role": "assistant", "content": '{"title": "Checking permissions", '
         '"command": "ls -l", "description": "x"}'},
    ]
    client = ModelClient(transport=transport)
    answer = client.converse("what did I ask about last time?", history)

    assert answer == "You asked about file permissions earlier."
    assert "response_format" not in seen["payload"]
    messages = seen["payload"]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1:3] == history
    assert messages[-1] == {
        "role": "user",
        "content": "what did I ask about last time?",
    }


def test_converse_raises_clear_error_on_network_failure() -> None:
    """A transport ``OSError`` surfaces as a clean ``ModelClientError``."""

    def transport(url: str, payload: dict, timeout: float):
        raise ConnectionRefusedError("connection refused")
        yield  # pragma: no cover - makes this a generator

    client = ModelClient(endpoint="http://unreachable:8080", transport=transport)
    with pytest.raises(ModelClientError) as excinfo:
        client.converse("anything")
    assert "unreachable:8080" in str(excinfo.value)


def test_converse_raises_on_empty_reply() -> None:
    """An empty conversational reply is rejected rather than returned blank."""

    def transport(url: str, payload: dict, timeout: float) -> list[str]:
        return [_content_chunk("   "), _finish_chunk()]

    client = ModelClient(transport=transport)
    with pytest.raises(ModelClientError):
        client.converse("anything")
