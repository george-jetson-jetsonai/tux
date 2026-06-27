"""Build tux's self-contained ``.deb`` — the heavy, non-offline build step.

Run from the repository root::

    python3 packaging/build_deb.py [--output-dir DIR] [--keep-build]

This bundles a Python interpreter and the tux package into a single executable
with PyInstaller, lays that binary out into a ``dpkg-deb`` tree via
:mod:`tux.packaging`, builds the ``.deb``, and lints it with ``lintian`` when
available. The version comes from :data:`tux.__version__` and the architecture
from ``dpkg --print-architecture``, so nothing is hardcoded.

PyInstaller, ``dpkg-deb``, and ``lintian`` are build-time tools; none of this
script runs in the offline unit suite, which exercises the metadata, maintainer
scripts, and build wiring directly via :mod:`tux.packaging`.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"

# Import tux.packaging from the source tree without requiring an install.
sys.path.insert(0, str(SRC))

from tux.packaging import (  # noqa: E402  (path set up just above)
    PACKAGE_NAME,
    build_deb,
    deb_filename,
    host_architecture,
    lay_out_package,
    package_version,
)

ENTRYPOINT = Path(__file__).resolve().parent / "entrypoint.py"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the build script's command-line arguments."""
    parser = argparse.ArgumentParser(description="Build tux's self-contained .deb.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "dist",
        help="Directory to write the built .deb into (default: ./dist).",
    )
    parser.add_argument(
        "--keep-build",
        action="store_true",
        help="Keep the intermediate build tree instead of removing it.",
    )
    parser.add_argument(
        "--variant",
        choices=("lite", "full"),
        default=None,
        help="Build a variant package (e.g. 'lite' for tux-lite, which pins the "
        "lite tier at install time) instead of the base tux package.",
    )
    return parser.parse_args(argv)


def build_binary(work: Path) -> Path:
    """Freeze tux into a single interpreter-bundling executable; return its path."""
    dist = work / "pyinstaller-dist"
    subprocess.run(
        [
            "pyinstaller",
            "--onefile",
            "--clean",
            "--noconfirm",
            "--name",
            PACKAGE_NAME,
            "--paths",
            str(SRC),
            "--distpath",
            str(dist),
            "--workpath",
            str(work / "pyinstaller-build"),
            "--specpath",
            str(work),
            str(ENTRYPOINT),
        ],
        check=True,
    )
    return dist / PACKAGE_NAME


def lint(deb: Path) -> None:
    """Lint the built ``.deb`` with ``lintian`` if it is installed.

    Reports the outcome but does not abort the build on warnings; the acceptance
    bar is no *errors*, which ``--fail-on error`` surfaces with a non-zero exit.
    """
    if shutil.which("lintian") is None:
        print("lintian not found; skipping lint (install lintian to validate).")
        return
    result = subprocess.run(["lintian", "--fail-on", "error", str(deb)], check=False)
    if result.returncode != 0:
        print("lintian reported errors (see above).", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    """Build the ``.deb`` and return a process exit status."""
    args = parse_args(argv)
    version = package_version()
    architecture = host_architecture()

    work = REPO_ROOT / "build" / "deb"
    if work.exists():
        shutil.rmtree(work)
    tree = work / "tree"
    tree.mkdir(parents=True)

    binary = build_binary(work)
    lay_out_package(
        tree, binary, version=version, architecture=architecture, variant=args.variant
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / deb_filename(version, architecture, args.variant)
    build_deb(tree, output)
    lint(output)

    if not args.keep_build:
        shutil.rmtree(work, ignore_errors=True)

    print(f"Built {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
