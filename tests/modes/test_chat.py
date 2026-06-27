"""Tests for the pure conversational-path request shaping and routing."""

import pytest

from tux.modes.chat import (
    build_chat_payload,
    build_classify_payload,
    interpret_route,
)


def test_build_classify_payload_omits_response_format() -> None:
    """The routing call is plain chat with no structured-output constraint."""
    payload = build_classify_payload("m.gguf", "delete temp files", [])
    assert "response_format" not in payload
    assert payload["messages"][-1] == {"role": "user", "content": "delete temp files"}


def test_build_chat_payload_omits_schema_and_orders_messages() -> None:
    """A conversational body carries history in order and no schema."""
    history = [{"role": "user", "content": "how do I check permissions?"}]
    payload = build_chat_payload("m.gguf", "what did I just ask?", history)

    assert "response_format" not in payload
    messages = payload["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "how do I check permissions?"}
    assert messages[-1] == {"role": "user", "content": "what did I just ask?"}


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("COMMAND", "command"),
        (" command ", "command"),
        ("CHAT", "chat"),
        ("anything else", "chat"),
        ("", "chat"),
    ],
)
def test_interpret_route_maps_reply_to_mode(reply: str, expected: str) -> None:
    """The router's word maps to a mode; non-command votes fall back to chat."""
    assert interpret_route(reply) == expected
