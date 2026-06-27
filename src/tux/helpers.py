"""Shared, pure helpers for both the command and conversational model paths.

These have no knowledge of HTTP or of which path is calling them: they assemble
the chat message list and accumulate a streamed reply. Keeping them here lets
:mod:`tux.modes.command` and :mod:`tux.modes.chat` build requests the same way and lets
:mod:`tux.client` collect any stream without either path duplicating the logic.
"""

import json
from collections.abc import Iterable


def build_messages(
    system_prompt: str, question: str, history: list[dict[str, str]]
) -> list[dict[str, str]]:
    """Return ``[system, *history, user]`` for a chat request.

    The system prompt leads, prior turns follow oldest-first, and the new
    question comes last, so the model always sees the running conversation in
    order.
    """
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": question})
    return messages


def collect_stream(lines: Iterable[str]) -> str:
    """Accumulate ``delta.content`` from SSE lines, stopping at ``finish_reason``.

    Termination is driven by ``finish_reason`` rather than the trailing ``[DONE]``
    sentinel so the request finishes promptly when the model signals completion.
    """
    buffer: list[str] = []
    for line in lines:
        raw = line.rstrip("\n")
        if not raw.startswith("data:"):
            continue
        data = raw[len("data:"):].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        choice = chunk["choices"][0]
        content = choice.get("delta", {}).get("content")
        if content:
            buffer.append(content)
        if choice.get("finish_reason"):
            break
    return "".join(buffer)
