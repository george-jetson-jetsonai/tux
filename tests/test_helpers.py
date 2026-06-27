"""Tests for the shared request/stream helpers."""

import json

from tux.helpers import build_messages, collect_stream


def _content_chunk(text: str) -> str:
    return f"data: {json.dumps({'choices': [{'delta': {'content': text}}]})}\n"


def _finish_chunk(reason: str = "stop") -> str:
    return f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': reason}]})}\n"


def test_build_messages_orders_system_history_then_question() -> None:
    """``build_messages`` returns the system prompt, prior turns, then the new turn."""
    history = [
        {"role": "user", "content": "earlier"},
        {"role": "assistant", "content": "reply"},
    ]
    messages = build_messages("be helpful", "now this", history)
    assert messages[0] == {"role": "system", "content": "be helpful"}
    assert messages[1:3] == history
    assert messages[-1] == {"role": "user", "content": "now this"}


def test_collect_stream_stops_at_finish_reason_ignoring_later_chunks() -> None:
    """The stream stops on ``finish_reason`` and never reads past it."""
    body = json.dumps({"command": "echo hi", "description": "prints hi"})

    def lines():
        yield _content_chunk(body)
        yield _finish_chunk()
        raise AssertionError("stream consumed past finish_reason")

    assert collect_stream(lines()) == body


def test_collect_stream_ignores_done_sentinel_and_non_data_lines() -> None:
    """Blank/non-``data:`` lines are skipped; ``[DONE]`` terminates if it arrives."""
    chunks = [
        ": keep-alive comment\n",
        "\n",
        _content_chunk("part"),
        "data: [DONE]\n",
        _content_chunk("unreached"),
    ]
    assert collect_stream(chunks) == "part"
