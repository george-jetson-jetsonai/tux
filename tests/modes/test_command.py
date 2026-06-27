"""Tests for the pure command-path request shaping and parsing."""

import json

import pytest

from tux.modes.command import (
    CommandSuggestion,
    assistant_turn,
    build_payload,
    output_message,
    parse_plan,
)


def test_build_payload_constrains_output_and_orders_messages() -> None:
    """The command body carries the strict step schema and system/history/question order."""
    history = [{"role": "user", "content": "earlier"}]
    payload = build_payload("m.gguf", "biggest files?", history)

    assert payload["model"] == "m.gguf"
    assert payload["stream"] is True
    assert payload["response_format"]["type"] == "json_schema"
    schema = payload["response_format"]["json_schema"]
    assert schema["strict"] is True
    # The reply is now an ordered array of single-command steps.
    assert schema["schema"]["required"] == ["steps"]
    step = schema["schema"]["properties"]["steps"]["items"]
    assert step["required"] == ["title", "command", "description"]
    messages = payload["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "earlier"}
    assert messages[-1] == {"role": "user", "content": "biggest files?"}


def test_build_payload_prompt_asks_for_ordered_discovery_first_plan() -> None:
    """The system prompt teaches an ordered, one-command-per-step, discovery-first plan."""
    prompt = build_payload("m.gguf", "x", [])["messages"][0]["content"].lower()
    assert "ordered" in prompt
    assert "discovery" in prompt
    # No chaining within a step, and a simple lookup is a single step.
    assert "&&" in prompt
    assert "single-step" in prompt or "single step" in prompt


def test_parse_plan_round_trips_a_multi_step_plan() -> None:
    """A well-formed ``steps`` array parses into an ordered list of suggestions."""
    content = json.dumps(
        {
            "steps": [
                {"title": "Printing cwd", "command": "pwd", "description": "shows cwd"},
                {
                    "title": "Removing it",
                    "command": "rm -rf {cwd}",
                    "description": "removes the cwd",
                },
            ]
        }
    )
    assert parse_plan(content) == [
        CommandSuggestion("Printing cwd", "pwd", "shows cwd"),
        CommandSuggestion("Removing it", "rm -rf {cwd}", "removes the cwd"),
    ]


def test_parse_plan_round_trips_a_one_step_plan() -> None:
    """A simple lookup comes back as a single-element plan."""
    content = json.dumps(
        {"steps": [{"title": "Listing by size", "command": "ls -laS", "description": "by size"}]}
    )
    assert parse_plan(content) == [CommandSuggestion("Listing by size", "ls -laS", "by size")]


@pytest.mark.parametrize(
    "content",
    [
        "",
        "   ",
        "not json",
        json.dumps({"steps": []}),
        json.dumps({"steps": [{"command": "ls", "description": "x"}]}),
        json.dumps({"title": "x", "command": "ls", "description": "x"}),
    ],
)
def test_parse_plan_rejects_bad_content(content: str) -> None:
    """Empty, unparseable, step-less, or incomplete content raises ``ValueError``."""
    with pytest.raises(ValueError):
        parse_plan(content)


def test_assistant_turn_renders_schema_shaped_plan_message() -> None:
    """``assistant_turn`` round-trips a plan into a schema-shaped chat message."""
    message = assistant_turn(
        [
            CommandSuggestion("Listing files", "ls", "lists files"),
            CommandSuggestion("Counting them", "ls | wc -l", "counts them"),
        ]
    )
    assert message["role"] == "assistant"
    assert json.loads(message["content"]) == {
        "steps": [
            {"title": "Listing files", "command": "ls", "description": "lists files"},
            {"title": "Counting them", "command": "ls | wc -l", "description": "counts them"},
        ]
    }


def test_output_message_carries_command_and_output() -> None:
    """The re-planning prompt embeds the command and its captured output."""
    message = output_message("pwd", "/home/me/work")
    assert "pwd" in message
    assert "/home/me/work" in message
