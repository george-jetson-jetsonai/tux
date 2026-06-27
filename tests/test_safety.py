"""Tests for the static destructive-command detection in :mod:`tux.safety`."""

import pytest

from tux.safety import destructive_reason

#: Representative destructive forms; each must yield a non-empty reason.
DESTRUCTIVE = [
    "rm -rf /",
    "rm -r build",
    "rm -f keep.txt",
    "rm --recursive --force /tmp/x",
    "rm --no-preserve-root -rf /",
    "sudo rm -rf /var/lib",
    "dd if=/dev/zero of=/dev/sda",
    "dd if=disk.img of=/dev/nvme0n1 bs=4M",
    "mkfs.ext4 /dev/sdb1",
    "sudo mkfs -t ext4 /dev/sdb",
    "shred -u secret.key",
    "wipefs -a /dev/sdc",
    "cat image.iso > /dev/sda",
    "echo boom >> /dev/nvme0n1",
    "echo hi | dd of=/dev/sda",
    ":(){ :|:& };:",
    ":(){ :|: & };:",
    "chmod -R 777 /",
    "chown -R nobody /etc",
    "sudo chgrp -R staff /usr",
]

#: Representative benign commands; each must yield no warning at all.
BENIGN = [
    "ls -la",
    "cd /tmp",
    "cat notes.txt",
    "rm notes.txt",
    "rm ./old.log",
    "echo 'rm -rf is just text'",
    "grep -rf pattern.txt src",
    "ps aux --sort=-%mem | head -10",
    "cat /dev/null",
    "ddrescue /dev/sr0 image.iso",
    "chmod 644 config.toml",
    "chmod -R 755 ./myproject",
    "chown user file.txt",
    "git status",
    "mkdir -p build/output",
    "df -h /dev/sda1",
]


@pytest.mark.parametrize("command", DESTRUCTIVE)
def test_destructive_commands_are_flagged(command: str) -> None:
    """A known destructive form yields a short, non-empty reason."""
    reason = destructive_reason(command)
    assert reason is not None
    assert reason.strip() == reason and reason != ""


@pytest.mark.parametrize("command", BENIGN)
def test_benign_commands_are_not_flagged(command: str) -> None:
    """An everyday command yields no warning, avoiding alarm fatigue."""
    assert destructive_reason(command) is None


def test_empty_command_is_not_flagged() -> None:
    """An empty or whitespace-only command is never flagged."""
    assert destructive_reason("") is None
    assert destructive_reason("   ") is None


def test_reason_is_specific_per_pattern() -> None:
    """Different patterns produce distinct, learning-oriented reasons."""
    assert "deletes" in destructive_reason("rm -rf /tmp/x")
    assert "filesystem" in destructive_reason("dd if=a of=/dev/sda")
    assert "formats" in destructive_reason("mkfs.ext4 /dev/sdb")
    assert "fork bomb" in destructive_reason(":(){ :|:& };:")


def test_path_prefixed_command_is_flagged() -> None:
    """A command given by absolute path is matched on its bare name."""
    assert destructive_reason("/bin/rm -rf /data") is not None
    assert destructive_reason("/sbin/mkfs.ext4 /dev/sdb1") is not None


def test_unparseable_command_does_not_crash() -> None:
    """A command shlex cannot tokenise still returns a result without raising."""
    # Unbalanced quote: tokenising falls back to a whitespace split.
    assert destructive_reason("rm -rf 'unterminated") is not None
