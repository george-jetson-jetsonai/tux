"""Interactive single-choice menu driven by the arrow keys, standard-library only.

This is the run/dismiss selection surface: tux prints a short list of options and
the user moves a highlight with the up/down arrow keys (or ``j``/``k``) and
presses Enter to choose. It puts the terminal into cbreak mode with
:mod:`termios` and :mod:`tty`, reads one keypress at a time, and always restores
the terminal state, so it is terminal-only by design and lives behind the
injectable :data:`Chooser` seam in :mod:`tux.cli` — tests supply a fake, so this
raw reader is never exercised under pytest.

The safe option is the default: it starts highlighted, and any abort —
end-of-input (Ctrl-D), Escape, or interrupt — resolves to it rather than to an
action, so a caller that lists the safe choice first never runs anything on a
stray keystroke.
"""

import sys
import termios
import tty
from collections.abc import Callable, Sequence

#: A chooser shows the given option labels and returns the chosen index. The
#: default highlights — and falls back to — index ``0``, so callers list the
#: safe option first; tests inject a fake to drive the run/dismiss flow.
Chooser = Callable[[Sequence[str]], int]

#: Bytes that confirm the highlighted option.
_ENTER = ("\r", "\n")
#: End-of-transmission (Ctrl-D) byte; in cbreak mode it arrives as data, not EOF.
_EOT = "\x04"
#: Escape, both on its own and as the lead byte of an arrow-key CSI sequence.
_ESC = "\x1b"

#: Inverse-video on/off, used to highlight the currently-selected option.
_INVERSE = "\x1b[7m"
_RESET = "\x1b[0m"

#: Box-drawing pieces for the border around the menu.
_TOP_LEFT, _TOP_RIGHT = "┌", "┐"
_BOTTOM_LEFT, _BOTTOM_RIGHT = "└", "┘"
_HORIZONTAL, _VERTICAL = "─", "│"


def select(options: Sequence[str], *, default: int = 0) -> int:
    """Show ``options`` as an arrow-navigable menu and return the chosen index.

    ``default`` is highlighted first and returned on any abort (Ctrl-D, Escape,
    or Ctrl-C), so the caller's safe choice stays the default.
    """
    index = default
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        _draw(options, index, redraw=False)
        while True:
            key = _read_key()
            if key in _ENTER:
                return index
            if key is None or key == _ESC:
                return default
            if key == "up":
                index = (index - 1) % len(options)
            elif key == "down":
                index = (index + 1) % len(options)
            else:
                continue
            _draw(options, index, redraw=True)
    except KeyboardInterrupt:
        return default
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        # Leave the cursor on a fresh line below the menu we drew.
        print()


def _draw(options: Sequence[str], index: int, *, redraw: bool) -> None:
    """Render the bordered menu, highlighting ``options[index]``, in place.

    The options sit inside a box-drawing border; the selected row is shown in
    inverse video so the highlight tracks the arrow keys. A redraw first steps
    the cursor back up over the whole box — border rows included — so each
    keystroke repaints the same lines rather than scrolling.
    """
    width = max(len(option) for option in options) + 3  # pointer plus two spaces
    if redraw:
        sys.stdout.write(f"\x1b[{len(options) + 2}A")
    top = f"{_TOP_LEFT}{_HORIZONTAL * width}{_TOP_RIGHT}"
    bottom = f"{_BOTTOM_LEFT}{_HORIZONTAL * width}{_BOTTOM_RIGHT}"
    sys.stdout.write(f"\r{top}\x1b[K\n")
    for i, option in enumerate(options):
        pointer = "❯" if i == index else " "
        cell = f" {pointer} {option}".ljust(width)
        if i == index:
            cell = f"{_INVERSE}{cell}{_RESET}"
        sys.stdout.write(f"\r{_VERTICAL}{cell}{_VERTICAL}\x1b[K\n")
    sys.stdout.write(f"\r{bottom}\x1b[K\n")
    sys.stdout.flush()


def _read_key() -> str | None:
    """Read one logical keypress from the terminal.

    Returns ``"up"``/``"down"`` for the arrow keys (or ``k``/``j``), the raw
    character otherwise, and ``None`` on end-of-input (Ctrl-D or a closed stream)
    so the caller can fall back to its default.
    """
    char = sys.stdin.read(1)
    if char in ("", _EOT):
        return None
    if char == _ESC:
        # A CSI arrow sequence is ESC [ A/B; a bare Escape has no trailing bytes.
        if sys.stdin.read(1) != "[":
            return _ESC
        final = sys.stdin.read(1)
        return {"A": "up", "B": "down"}.get(final, _ESC)
    if char == "k":
        return "up"
    if char == "j":
        return "down"
    return char
