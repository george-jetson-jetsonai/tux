"""Per-turn request shaping for tux's two response modes.

Each module here owns one mode and is pure — it builds request bodies and parses
replies, knowing nothing about HTTP or streaming (those live in
:mod:`tux.client`):

* :mod:`tux.modes.command` — the structured command path (strict ``json_schema``
  yielding ``{title, command, description}``).
* :mod:`tux.modes.chat` — the free-form conversational path (no schema) plus the
  lightweight routing that decides which mode a turn is.
"""
