"""PyInstaller entry point for the bundled ``tux`` executable.

PyInstaller freezes this script (and the tux package and a Python interpreter)
into the single ``/usr/bin/tux`` shipped in the ``.deb``. It simply forwards to
the console-script entry point so the frozen binary behaves like ``tux`` does
when installed from source.
"""

from tux.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
