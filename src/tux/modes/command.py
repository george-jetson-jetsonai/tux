"""Command-path request shaping and parsing for the tux model client.

This module owns the *structured command* turn: the system prompt, the strict
``json_schema`` response format that guarantees an ordered ``steps`` plan — each
step a ``{title, command, description}`` object — the parsing of that reply, and
rendering a prior plan back into a chat message for the running thread.

A command turn is answered as a step-by-step guided plan rather than a single
command: a discovery step (e.g. ``pwd``) comes before the action step that uses
its result, every step is exactly one command, and a simple lookup is a one-step
plan. A later step may carry a placeholder that a prior step's output resolves.

It is deliberately pure — it builds dicts and parses strings and knows nothing
about HTTP, streaming, or the conversational path. The transport and error
translation live in :mod:`tux.client`; the free-form path lives in
:mod:`tux.modes.chat`. Parsing problems are raised as :class:`ValueError` for the
client to translate into its public error type.
"""

import json
from dataclasses import dataclass

from tux.helpers import build_messages

#: Upper bound on tokens for a plan; a handful of short steps, each a title, a
#: command, and a one-line description, all stay brief.
MAX_TOKENS = 512

SYSTEM_PROMPT = (
    "You are a Linux command assistant for Ubuntu. The user describes a task in "
    "plain English; you answer with an ORDERED, step-by-step plan, never running "
    "anything yourself. Return a 'steps' array; each step has title (a few words, "
    "e.g. 'Listing hidden files'), command (ONE bare shell command, no "
    "markdown/backticks, and no '&&', ';' or '|' chaining), and description (one "
    "short plain sentence). Teach the user the system context: put a discovery "
    "step that gathers context (e.g. 'pwd', 'ls', 'which') BEFORE the action step "
    "that depends on it, rather than hiding the context in one compound command. "
    "A later step may use a {placeholder} that an earlier step's output fills in. "
    "A simple lookup that needs only one command is a single-step plan. When you "
    "are given the output of a step the user just ran, or a clarification, return "
    "the revised remaining steps only."
)


@dataclass(frozen=True)
class CommandSuggestion:
    """A single proposed shell command with a short title and one-line description."""

    title: str
    command: str
    description: str


#: An ordered guided plan: one or more single-command steps walked in turn. A
#: simple lookup is just a one-element plan.
Plan = list[CommandSuggestion]


def build_payload(model: str, question: str, history: list[dict[str, str]]) -> dict:
    """Build the OpenAI-compatible request body with strict structured output.

    The message list is the system prompt, then any prior turns, then this
    turn's question, so the model sees the running conversation. The structured
    reply is an ordered ``steps`` array, each item a single-command step.
    """
    return {
        "model": model,
        "stream": True,
        "max_tokens": MAX_TOKENS,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "command_plan",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "steps": {
                            "type": "array",
                            "description": "the ordered steps of the plan, one "
                            "shell command per step, discovery before action",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": {
                                        "type": "string",
                                        "description": "a few words summarising "
                                        "this step, e.g. 'Listing hidden files'",
                                    },
                                    "command": {
                                        "type": "string",
                                        "description": "the single shell command "
                                        "for this step, bare, no markdown",
                                    },
                                    "description": {
                                        "type": "string",
                                        "description": "one short plain-text "
                                        "sentence explaining it, no backticks",
                                    },
                                },
                                "required": ["title", "command", "description"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["steps"],
                    "additionalProperties": False,
                },
            },
        },
        "messages": build_messages(SYSTEM_PROMPT, question, history),
    }


def parse_plan(content: str) -> Plan:
    """Parse accumulated stream content into an ordered ``Plan``.

    Raises:
        ValueError: If the content is empty, not JSON, carries no steps, or a
            step is missing a field. The client translates this into its public
            ``ModelClientError``.
    """
    content = content.strip()
    if not content:
        raise ValueError("the model returned an empty response")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"could not parse the model response ({exc})") from exc
    try:
        steps = parsed["steps"]
    except (KeyError, TypeError) as exc:
        raise ValueError("the model response was missing the expected fields") from exc
    if not isinstance(steps, list) or not steps:
        raise ValueError("the model response contained no steps")
    plan: Plan = []
    for step in steps:
        try:
            plan.append(
                CommandSuggestion(
                    title=step["title"],
                    command=step["command"],
                    description=step["description"],
                )
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "the model response was missing the expected fields"
            ) from exc
    return plan


def assistant_turn(plan: Plan) -> dict[str, str]:
    """Render a prior plan as an assistant message for the next request.

    The content mirrors the structured ``json_schema`` shape the model itself
    emits, so the running thread stays consistent with the response format.
    """
    content = json.dumps(
        {
            "steps": [
                {
                    "title": step.title,
                    "command": step.command,
                    "description": step.description,
                }
                for step in plan
            ]
        }
    )
    return {"role": "assistant", "content": content}


def output_message(command: str, output: str) -> str:
    """Phrase a run step's captured output as a re-planning prompt.

    Fed back to the model after the user runs a step so a later step's
    placeholder resolves from the real output, without the user copy/pasting.
    """
    return (
        f"I ran the command: {command}\n"
        f"Its output was:\n{output}\n"
        "Using this output, give the revised remaining steps of the plan, "
        "resolving any placeholders."
    )
