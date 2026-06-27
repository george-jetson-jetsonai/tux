"""Conversational-path request shaping and routing for the tux model client.

This module owns the *free-form* turn: a lightweight routing prompt that decides
whether a message is a command request or ordinary conversation, and a chat
prompt that answers conversationally. Neither uses a ``json_schema`` response
format, so a plain question — including one that refers back to an earlier turn —
is answered as prose instead of being forced into a command.

Like :mod:`tux.modes.command`, it is pure: it builds request bodies and interprets the
routing reply, knowing nothing about HTTP or streaming. The transport lives in
:mod:`tux.client`.
"""

from tux.helpers import build_messages

#: Routing needs only a single word back; conversational replies need room for a
#: short paragraph.
CLASSIFY_MAX_TOKENS = 4
CHAT_MAX_TOKENS = 512

#: Prompt for the lightweight routing call that decides a turn's type before the
#: real answer is produced. It must reply with exactly one of two words.
CLASSIFY_PROMPT = (
    "You route messages for a Linux assistant. Read the user's latest message in "
    "the context of the conversation and decide its type. Reply with exactly one "
    "word and nothing else: COMMAND if it asks to do something on the system that "
    "is best answered with a shell command, or CHAT if it is ordinary "
    "conversation (a greeting, a question about the conversation itself, or a "
    "request for an explanation)."
)

#: System prompt for a free-form conversational turn. No shell command is
#: proposed here; the model answers in plain prose, in context.
CHAT_SYSTEM_PROMPT = (
    "You are tux, a friendly Linux assistant for Ubuntu. Answer the user's "
    "message conversationally, in plain prose, drawing on the earlier turns of "
    "this conversation when relevant. Do not propose or run a shell command in "
    "this reply; just talk with the user."
)


def build_classify_payload(
    model: str, question: str, history: list[dict[str, str]]
) -> dict:
    """Build the request body for the routing call (no structured output)."""
    return _build_payload(model, CLASSIFY_PROMPT, question, history, CLASSIFY_MAX_TOKENS)


def build_chat_payload(
    model: str, question: str, history: list[dict[str, str]]
) -> dict:
    """Build the request body for a free-form conversational reply."""
    return _build_payload(model, CHAT_SYSTEM_PROMPT, question, history, CHAT_MAX_TOKENS)


def interpret_route(reply: str) -> str:
    """Map the router's reply to ``"command"`` or ``"chat"``.

    Anything that is not an explicit command vote falls back to ``"chat"`` so a
    stray word never forces an unwanted command proposal.
    """
    return "command" if "COMMAND" in reply.strip().upper() else "chat"


def _build_payload(
    model: str,
    system_prompt: str,
    question: str,
    history: list[dict[str, str]],
    max_tokens: int,
) -> dict:
    """Build a plain chat request body with **no** structured-output format."""
    return {
        "model": model,
        "stream": True,
        "max_tokens": max_tokens,
        "messages": build_messages(system_prompt, question, history),
    }
