"""Tests for the arrow-key chooser's key decoding.

The full :func:`tux.chooser.select` loop drives a real terminal via ``termios``
and is the untested raw seam (tests inject a fake chooser into the CLI instead).
The keypress decoding is pure, so it is covered here against a fake stdin.
"""

import pytest

from tux import chooser
from tux.chooser import _draw, _read_key


class _FakeStdin:
    """A stand-in stdin that hands out ``read(n)`` bytes from a fixed string."""

    def __init__(self, data: str) -> None:
        self._data = data

    def read(self, count: int) -> str:
        chunk = self._data[:count]
        self._data = self._data[count:]
        return chunk


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ("\x1b[A", "up"),
        ("\x1b[B", "down"),
        ("k", "up"),
        ("j", "down"),
        ("\r", "\r"),
        ("\n", "\n"),
        ("", None),
        ("\x04", None),
        ("\x1b", "\x1b"),
        ("q", "q"),
    ],
)
def test_read_key_decodes_keypresses(
    monkeypatch: pytest.MonkeyPatch, data: str, expected: str | None
) -> None:
    """Arrow sequences, vi keys, Enter, and end-of-input each decode as expected."""
    monkeypatch.setattr(chooser.sys, "stdin", _FakeStdin(data))
    assert _read_key() == expected


def test_draw_renders_bordered_box_and_highlights_selection(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The menu is drawn inside a border with the selected row in inverse video."""
    _draw(["Dismiss", "Run"], index=1, redraw=False)
    out = capsys.readouterr().out
    # A box-drawing border encloses the options.
    assert all(corner in out for corner in ("┌", "┐", "└", "┘"))
    lines = out.splitlines()
    run_line = next(line for line in lines if "Run" in line)
    dismiss_line = next(line for line in lines if "Dismiss" in line)
    # Only the selected row (Run) carries the inverse-video highlight.
    assert "\x1b[7m" in run_line
    assert "\x1b[7m" not in dismiss_line
