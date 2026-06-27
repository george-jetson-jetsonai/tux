"""Read and write tux's TOML config file for the endpoint, model, and variant.

The config file lives at ``$XDG_CONFIG_HOME/tux/config.toml`` (falling back to
``~/.config/tux/config.toml`` when ``XDG_CONFIG_HOME`` is unset). It holds three
optional top-level string keys: ``endpoint``, ``model``, and ``variant`` (the
hardware-aware capability tier recorded by provisioning). Any key absent from
the file falls back to a built-in default supplied by the caller, so a missing
file simply means "use the defaults".

Reading uses the standard-library ``tomllib``. Writing hand-formats the known
scalar keys as TOML basic strings — ``json.dumps`` produces escaping that is a
valid subset of TOML basic-string syntax and round-trips through ``tomllib``.
"""

import json
import os
import tomllib
from pathlib import Path

#: The only keys tux reads from or writes to the config file. ``variant`` records
#: the capability tier provisioning chose for this host so the (separate)
#: variant-gating item can read it back.
ALLOWED_KEYS = ("endpoint", "model", "variant")


class ConfigError(Exception):
    """Raised when the config file is malformed or an unknown key is requested."""


def config_path() -> Path:
    """Return the absolute path tux uses for its config file.

    The path is returned whether or not the file currently exists.
    """
    base = os.environ.get("XDG_CONFIG_HOME")
    root = Path(base) if base else Path.home() / ".config"
    return root / "tux" / "config.toml"


def load_config() -> dict[str, str]:
    """Return the configured ``endpoint``/``model`` overrides from the config file.

    An absent file yields an empty mapping. Keys outside :data:`ALLOWED_KEYS` are
    ignored so out-of-scope content does not leak into the client.

    Raises:
        ConfigError: If the file exists but is not valid TOML.
    """
    path = config_path()
    if not path.exists():
        return {}
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"config file at {path} is not valid TOML ({exc})") from exc
    return {key: data[key] for key in ALLOWED_KEYS if key in data}


def set_value(key: str, value: str) -> None:
    """Persist ``key = value`` to the config file, preserving the other key.

    The parent directory and file are created when absent. Existing allowed keys
    already in the file are kept, so setting one key never drops the other.

    Raises:
        ConfigError: If ``key`` is not one of :data:`ALLOWED_KEYS`, or if the
            existing file is malformed. Nothing is written in either case.
    """
    if key not in ALLOWED_KEYS:
        allowed = ", ".join(ALLOWED_KEYS)
        raise ConfigError(f"unknown config key {key!r}; allowed keys are: {allowed}")
    config = load_config()
    config[key] = value
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_dump_toml(config), encoding="utf-8")


def resolved_settings(defaults: dict[str, str]) -> list[tuple[str, str, str]]:
    """Return ``(key, value, source)`` for every key in ``defaults``.

    ``source`` is ``"config"`` when the value comes from the config file and
    ``"default"`` when it falls back to the built-in default.

    Raises:
        ConfigError: If the config file exists but is not valid TOML.
    """
    overrides = load_config()
    settings: list[tuple[str, str, str]] = []
    for key, default in defaults.items():
        if key in overrides:
            settings.append((key, overrides[key], "config"))
        else:
            settings.append((key, default, "default"))
    return settings


def _dump_toml(config: dict[str, str]) -> str:
    """Render ``config`` as TOML with each value as a basic string."""
    return "".join(f"{key} = {json.dumps(value)}\n" for key, value in config.items())
