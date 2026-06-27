"""Static, local inspection of a proposed command for destructive patterns.

This module is a pure safety net: :func:`destructive_reason` maps a proposed
shell command string to an optional short, plain-English reason explaining why
the command is potentially destructive. It trusts nothing from the model — the
reason is derived solely from the command text — so a dangerous command is
flagged even when the model's own description says nothing about risk.

Detection deliberately favours few false positives over exhaustive coverage:
benign everyday commands (listing, navigating, reading, a plain non-recursive
delete of a named file) never trigger a warning, so the flag stays meaningful
and users do not learn to ignore it. The single public function is a natural
unit-test seam, mirroring how the model client and runner are kept testable.
"""

import re
import shlex

#: Commands that elevate privileges; skipped so the real command behind, say,
#: ``sudo rm -rf /`` is still inspected.
_PRIVILEGE_COMMANDS = frozenset({"sudo", "doas"})

#: Operator tokens that separate one command from the next on a single line, so
#: ``echo hi | dd of=/dev/sda`` is inspected segment by segment.
_SEPARATORS = frozenset({"|", "||", "&&", ";", "&", "|&"})

#: Short flag letters (bundled, e.g. ``-rf``) and long flags that make ``rm``
#: recursive or forced — either is enough to flag the deletion.
_RM_FLAGS_SHORT = frozenset({"r", "R", "f"})
_RM_FLAGS_LONG = frozenset({"--recursive", "--force", "--no-preserve-root"})

#: Recursive flag for permission/ownership changes (``chmod``/``chown``/``chgrp``
#: spell recursion only with an uppercase ``-R`` or ``--recursive``).
_RECURSIVE_SHORT = frozenset({"R"})
_RECURSIVE_LONG = frozenset({"--recursive"})

#: Commands that change permissions or ownership; flagged only when recursive
#: *and* aimed at a broad system path, since a scoped recursive change is common
#: and benign.
_OWNERSHIP_COMMANDS = frozenset({"chmod", "chown", "chgrp"})

#: A raw block device — writing to one bypasses the filesystem and overwrites the
#: disk. Matched as a prefix so partitions (``/dev/sda1``, ``/dev/nvme0n1``) count.
_BLOCK_DEVICE = re.compile(r"^/dev/(sd[a-z]|nvme\d|hd[a-z]|vd[a-z]|mmcblk\d)")

#: A broad path whose recursive permission/ownership change is high-impact: the
#: root, the home shorthand, a bare glob, or a top-level system directory.
_BROAD_PATH = re.compile(
    r"^(/|/\*|\*|~|/(bin|boot|etc|home|lib|opt|proc|root|sbin|sys|usr|var)/?)$"
)

_RM_REASON = "recursively or forcibly deletes files and directories"
_DD_REASON = "writes raw data directly to a device, bypassing the filesystem"
_MKFS_REASON = "formats a filesystem, erasing any data already on the device"
_SHRED_REASON = "overwrites files to destroy their contents irrecoverably"
_WIPEFS_REASON = "erases the filesystem signature from a device"
_BLOCK_DEVICE_REASON = "writes straight to a raw disk device, overwriting whatever is on it"
_FORK_BOMB_REASON = "is a fork bomb that spawns processes until the system is starved"
_OWNERSHIP_REASONS = {
    "chmod": "recursively changes permissions across a broad system path",
    "chown": "recursively changes ownership across a broad system path",
    "chgrp": "recursively changes group ownership across a broad system path",
}


def destructive_reason(command: str) -> str | None:
    """Return why ``command`` is potentially destructive, or ``None`` if it is not.

    The command is tokenised with :func:`shlex.split` (falling back to a plain
    whitespace split when it cannot be parsed, e.g. unbalanced quotes) so flags
    separate cleanly and ``rm -rf`` is matched without misfiring on an unrelated
    word that merely contains those letters. Each pipe/``;``/``&&`` segment is
    inspected in turn and the first match wins; the reason is a short,
    learning-oriented sentence rather than a bare "dangerous" label.
    """
    text = command.strip()
    if not text:
        return None
    if _is_fork_bomb(text):
        return _FORK_BOMB_REASON
    for segment in _segments(_tokenise(text)):
        reason = _segment_reason(segment)
        if reason is not None:
            return reason
    return None


def _tokenise(text: str) -> list[str]:
    """Split ``text`` into shell tokens, degrading to a whitespace split if it cannot parse."""
    try:
        return shlex.split(text)
    except ValueError:
        return text.split()


def _is_fork_bomb(text: str) -> bool:
    """Return whether ``text`` is the classic ``:(){ :|:& };:`` fork bomb.

    Whitespace is collapsed first so spacing variants normalise to one form.
    """
    return ":(){:|:&};:" in re.sub(r"\s+", "", text)


def _segments(tokens: list[str]) -> list[list[str]]:
    """Split a token list on shell separators into per-command token lists."""
    segments: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        if token in _SEPARATORS:
            if current:
                segments.append(current)
            current = []
        else:
            current.append(token)
    if current:
        segments.append(current)
    return segments


def _segment_reason(segment: list[str]) -> str | None:
    """Return the destructive reason for one command segment, or ``None``."""
    if _writes_block_device(segment):
        return _BLOCK_DEVICE_REASON
    body = _strip_privilege_prefix(segment)
    if not body:
        return None
    name = _command_name(body[0])
    args = body[1:]
    if name == "rm" and _has_flag(args, _RM_FLAGS_SHORT, _RM_FLAGS_LONG):
        return _RM_REASON
    if name == "dd":
        return _DD_REASON
    if name.startswith("mkfs"):
        return _MKFS_REASON
    if name == "shred":
        return _SHRED_REASON
    if name == "wipefs":
        return _WIPEFS_REASON
    if (
        name in _OWNERSHIP_COMMANDS
        and _has_flag(args, _RECURSIVE_SHORT, _RECURSIVE_LONG)
        and any(_BROAD_PATH.match(arg) for arg in args)
    ):
        return _OWNERSHIP_REASONS[name]
    return None


def _writes_block_device(segment: list[str]) -> bool:
    """Return whether ``segment`` redirects output onto a raw block device."""
    for index, token in enumerate(segment):
        if token in (">", ">>"):
            target = segment[index + 1] if index + 1 < len(segment) else ""
        elif token.startswith(">>"):
            target = token[2:]
        elif token.startswith(">"):
            target = token[1:]
        else:
            continue
        if _BLOCK_DEVICE.match(target):
            return True
    return False


def _strip_privilege_prefix(segment: list[str]) -> list[str]:
    """Drop a leading ``sudo``/``doas`` and any env-assignment tokens before the command."""
    index = 0
    while index < len(segment) and (
        _command_name(segment[index]) in _PRIVILEGE_COMMANDS or "=" in segment[index]
    ):
        index += 1
    return segment[index:]


def _command_name(token: str) -> str:
    """Return the bare command name, dropping any leading path (``/sbin/mkfs`` → ``mkfs``)."""
    return token.rsplit("/", 1)[-1]


def _has_flag(
    args: list[str], short_letters: frozenset[str], long_flags: frozenset[str]
) -> bool:
    """Return whether ``args`` carries any of the given short letters or long flags.

    Short letters are matched inside a bundled group such as ``-rf`` so order and
    combination do not matter; a ``--`` long flag must match in full.
    """
    for arg in args:
        if arg in long_flags:
            return True
        if len(arg) > 1 and arg[0] == "-" and arg[1] != "-":
            if any(letter in arg[1:] for letter in short_letters):
                return True
    return False
