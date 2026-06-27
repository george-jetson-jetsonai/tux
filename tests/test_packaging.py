"""Tests for the Debian packaging artifacts, all external build effects mocked.

No PyInstaller bundle, real ``dpkg``/``dpkg-deb``, install, model download, or
GPU is touched: the control metadata, debconf templates, and maintainer scripts
are asserted as strings, the package tree is laid out around a stub binary, and
the ``dpkg``/``dpkg-deb`` invocations go through an injected runner. The whole
suite runs offline.
"""

import gzip
import stat

import pytest

from tux import __version__
from tux.packaging import (
    BINARY_NAME,
    MAINTAINER,
    PACKAGE_NAME,
    PROVISION_QUESTION,
    build_deb,
    changelog,
    changelog_filename,
    control_file,
    deb_filename,
    host_architecture,
    lay_out_package,
    package_name,
    package_version,
    postinst,
)


class RecordingRunner:
    """A subprocess runner that records calls and returns a canned result."""

    def __init__(self, stdout: str = "") -> None:
        self.stdout = stdout
        self.calls: list[dict] = []

    def __call__(self, args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        return _CompletedProcess(self.stdout)


class _CompletedProcess:
    """Minimal stand-in for ``subprocess.CompletedProcess`` carrying ``stdout``."""

    def __init__(self, stdout: str) -> None:
        self.stdout = stdout
        self.returncode = 0


def test_package_version_is_project_version():
    """The package version comes from tux's single source of truth, not a literal."""
    assert package_version() == __version__


def test_deb_filename_follows_debian_convention():
    """The artifact name is ``name_version_arch.deb``."""
    assert deb_filename("1.2.3", "amd64") == "tux_1.2.3_amd64.deb"


def test_control_file_declares_required_metadata():
    """control carries the name, version, arch, maintainer, deps, and description."""
    control = control_file(version="0.1.0", architecture="amd64")
    assert "Package: tux\n" in control
    assert "Version: 0.1.0\n" in control
    assert "Architecture: amd64\n" in control
    assert f"Maintainer: {MAINTAINER}\n" in control
    assert "Section: utils\n" in control
    assert "Priority: optional\n" in control
    # The maintainer scripts source debconf's confmodule, so the package must
    # depend on debconf alongside the bundled binary's libc.
    assert "Depends: libc6, debconf (>= 0.5) | debconf-2.0\n" in control
    assert "Recommends: curl\n" in control
    # A synopsis line plus an indented extended body.
    assert "Description: " in control
    assert "\n This package bundles its own Python interpreter" in control


def test_control_maintainer_is_well_formed():
    """The maintainer field is in Debian ``Name <email>`` form."""
    assert "<" in MAINTAINER and MAINTAINER.endswith(">")


def test_control_installed_size_optional():
    """Installed-Size appears only when supplied (computed at lay-out time)."""
    assert "Installed-Size:" not in control_file(version="0.1.0", architecture="amd64")
    sized = control_file(version="0.1.0", architecture="amd64", installed_size=42)
    assert "Installed-Size: 42\n" in sized


def test_lay_out_package_writes_tree_with_bundled_binary(tmp_path):
    """The laid-out tree has the binary on PATH and a complete DEBIAN control area."""
    stub_binary = tmp_path / "tux-binary"
    stub_binary.write_bytes(b"\x7fELF stub bundled interpreter")
    dest = tmp_path / "tree"

    lay_out_package(dest, stub_binary, version="0.1.0", architecture="amd64")

    installed = dest / "usr" / "bin" / PACKAGE_NAME
    assert installed.read_bytes() == stub_binary.read_bytes()
    assert installed.stat().st_mode & stat.S_IXUSR

    debian = dest / "DEBIAN"
    for name in ("control", "templates", "config", "postinst", "postrm"):
        assert (debian / name).exists()

    # control records the binary's installed size in KiB.
    assert "Installed-Size: 1\n" in (debian / "control").read_text()


def test_lay_out_package_maintainer_scripts_are_executable(tmp_path):
    """postinst/postrm/config are 0755; control/templates are not executable."""
    stub_binary = tmp_path / "tux-binary"
    stub_binary.write_bytes(b"stub")
    dest = tmp_path / "tree"
    lay_out_package(dest, stub_binary, version="0.1.0", architecture="amd64")
    debian = dest / "DEBIAN"

    for name in ("postinst", "postrm", "config"):
        assert debian.joinpath(name).stat().st_mode & stat.S_IXUSR, name
    for name in ("control", "templates"):
        assert not debian.joinpath(name).stat().st_mode & stat.S_IXUSR, name


def test_postinst_defers_by_default_and_invokes_provision_on_opt_in(tmp_path):
    """postinst defers by default and only provisions on explicit debconf opt-in."""
    stub = tmp_path / "b"
    stub.write_bytes(b"x")
    dest = tmp_path / "tree"
    lay_out_package(dest, stub, version="0.1.0", architecture="amd64")
    postinst = (dest / "DEBIAN" / "postinst").read_text()

    assert postinst.startswith("#!/bin/sh\n")
    assert "set -e" in postinst
    # Debian-correct: sources debconf, never assumes a TTY.
    assert ". /usr/share/debconf/confmodule" in postinst
    # Provisioning reuses 8a's brain, only with consent recorded, never silently.
    assert "tux provision --yes </dev/null" in postinst
    assert 'if [ "$RET" = "true" ]' in postinst
    # The default branch defers rather than pulling.
    assert "Run 'tux provision'" in postinst
    # Packaging never pulls a model itself; that stays in 8a's brain.
    assert "ollama pull" not in postinst


def test_config_asks_provision_question_at_low_priority(tmp_path):
    """The debconf config script asks the question at low priority (so it defers)."""
    stub = tmp_path / "b"
    stub.write_bytes(b"x")
    dest = tmp_path / "tree"
    lay_out_package(dest, stub, version="0.1.0", architecture="amd64")
    config = (dest / "DEBIAN" / "config").read_text()

    assert ". /usr/share/debconf/confmodule" in config
    assert f"db_input low {PROVISION_QUESTION}" in config


def test_templates_default_to_deferring(tmp_path):
    """The debconf template is a boolean question defaulting to false (defer)."""
    stub = tmp_path / "b"
    stub.write_bytes(b"x")
    dest = tmp_path / "tree"
    lay_out_package(dest, stub, version="0.1.0", architecture="amd64")
    templates = (dest / "DEBIAN" / "templates").read_text()

    assert f"Template: {PROVISION_QUESTION}" in templates
    assert "Type: boolean" in templates
    assert "Default: false" in templates


def test_postrm_purge_only_clears_debconf_and_spares_user_data(tmp_path):
    """postrm clears debconf on purge but never removes user config or Ollama."""
    stub = tmp_path / "b"
    stub.write_bytes(b"x")
    dest = tmp_path / "tree"
    lay_out_package(dest, stub, version="0.1.0", architecture="amd64")
    postrm = (dest / "DEBIAN" / "postrm").read_text()

    assert 'case "$1" in' in postrm
    assert "purge)" in postrm
    assert "db_purge" in postrm
    # Removal must not destroy user config/state or the Ollama runtime/models:
    # the script runs no file-removal command at all (dpkg removes the package's
    # own files; everything else is deliberately left in place).
    assert "rm " not in postrm
    assert "rmdir" not in postrm


def test_changelog_filename_native_vs_non_native():
    """A native version (no Debian revision) ships changelog.gz, else .Debian.gz."""
    assert changelog_filename("0.1.0") == "changelog.gz"
    assert changelog_filename("0.1.0-1") == "changelog.Debian.gz"


def test_changelog_is_valid_debian_format():
    """The changelog entry has a Debian header line, a bullet, and a dated trailer."""
    entry = changelog(version="0.1.0", date="Sat, 27 Jun 2026 12:00:00 +0000")
    assert entry.startswith("tux (0.1.0) unstable; urgency=medium\n")
    assert "\n  * " in entry
    # The trailer matches the control maintainer and carries the date.
    assert f"\n -- {MAINTAINER}  Sat, 27 Jun 2026 12:00:00 +0000\n" in entry


def test_lay_out_package_ships_changelog_and_copyright(tmp_path):
    """The tree carries /usr/share/doc/tux/{copyright,changelog.gz} (lintian C3)."""
    stub = tmp_path / "b"
    stub.write_bytes(b"x")
    dest = tmp_path / "tree"
    lay_out_package(
        dest,
        stub,
        version="0.1.0",
        architecture="amd64",
        date="Sat, 27 Jun 2026 12:00:00 +0000",
    )

    doc = dest / "usr" / "share" / "doc" / PACKAGE_NAME
    copyright_text = (doc / "copyright").read_text()
    assert "Copyright (C)" in copyright_text

    changelog_gz = doc / "changelog.gz"
    assert changelog_gz.exists()
    decompressed = gzip.decompress(changelog_gz.read_bytes()).decode("utf-8")
    assert decompressed.startswith("tux (0.1.0) unstable;")


def test_lay_out_package_directories_are_0755(tmp_path):
    """Every packaged directory is 0755, never a build-umask 0775 (lintian warn)."""
    stub = tmp_path / "b"
    stub.write_bytes(b"x")
    dest = tmp_path / "tree"
    lay_out_package(dest, stub, version="0.1.0", architecture="amd64")

    for directory in dest.rglob("*"):
        if directory.is_dir():
            assert stat.S_IMODE(directory.stat().st_mode) == 0o755, directory


def test_host_architecture_reads_dpkg(monkeypatch):
    """host_architecture returns dpkg's printed architecture, stripped."""
    runner = RecordingRunner(stdout="arm64\n")
    assert host_architecture(runner=runner) == "arm64"
    assert runner.calls[0]["args"] == ["dpkg", "--print-architecture"]
    assert runner.calls[0]["kwargs"]["check"] is True


# --- tux-lite variant package ---------------------------------------------


def test_package_name_suffixes_variant():
    """The base package is ``tux``; a variant is ``tux-<variant>``."""
    assert package_name() == "tux"
    assert package_name(None) == "tux"
    assert package_name("lite") == "tux-lite"


def test_lite_control_uses_variant_package_name():
    """The tux-lite control file names the tux-lite package, shared metadata else."""
    control = control_file(version="0.1.0", architecture="amd64", variant="lite")
    assert "Package: tux-lite\n" in control
    assert "Package: tux\n" not in control
    # The rest of the metadata is shared with the base package.
    assert f"Maintainer: {MAINTAINER}\n" in control
    assert "Depends: libc6, debconf (>= 0.5) | debconf-2.0\n" in control


def test_lite_deb_filename_uses_variant_name():
    """The lite artifact is ``tux-lite_version_arch.deb``."""
    assert deb_filename("1.2.3", "amd64", "lite") == "tux-lite_1.2.3_amd64.deb"


def test_lite_changelog_header_names_variant():
    """The lite changelog header names the tux-lite package."""
    entry = changelog(
        version="0.1.0", date="Sat, 27 Jun 2026 12:00:00 +0000", variant="lite"
    )
    assert entry.startswith("tux-lite (0.1.0) unstable; urgency=medium\n")


def test_lite_postinst_pins_the_lite_variant():
    """The lite postinst provisions with the lite tier pinned."""
    script = postinst("lite")
    assert "tux provision --yes --variant lite </dev/null" in script
    # The base package stays hardware-probed (no pin).
    assert "--variant" not in postinst()
    assert "tux provision --yes </dev/null" in postinst()


def test_lite_lay_out_ships_tux_binary_under_lite_package(tmp_path):
    """The lite tree ships /usr/bin/tux but names the package and doc dir tux-lite."""
    stub = tmp_path / "b"
    stub.write_bytes(b"x")
    dest = tmp_path / "tree"
    lay_out_package(dest, stub, version="0.1.0", architecture="amd64", variant="lite")

    # The installed command is still tux (one codebase, gated at runtime).
    assert (dest / "usr" / "bin" / BINARY_NAME).exists()
    assert BINARY_NAME == "tux"
    # control and the doc dir carry the tux-lite package name (lintian wants the
    # doc dir to match the package).
    assert "Package: tux-lite\n" in (dest / "DEBIAN" / "control").read_text()
    assert (dest / "usr" / "share" / "doc" / "tux-lite").is_dir()
    # The maintainer script pins lite at install time.
    assert "--variant lite" in (dest / "DEBIAN" / "postinst").read_text()


def test_build_deb_invokes_dpkg_deb_with_root_owner(tmp_path):
    """build_deb shells out to dpkg-deb with root ownership and the right paths."""
    runner = RecordingRunner()
    tree = tmp_path / "tree"
    output = tmp_path / "out.deb"

    result = build_deb(tree, output, runner=runner)

    assert result == output
    assert runner.calls[0]["args"] == [
        "dpkg-deb",
        "--root-owner-group",
        "--build",
        str(tree),
        str(output),
    ]
    assert runner.calls[0]["kwargs"]["check"] is True
