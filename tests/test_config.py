"""Tests for the TOML config layer, kept hermetic via a redirected config path."""

import tomllib

import pytest

from tux.config import (
    ConfigError,
    config_path,
    load_config,
    resolved_settings,
    set_value,
)


@pytest.fixture
def config_home(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Point ``XDG_CONFIG_HOME`` at a temp dir so no real home dir is touched."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    return tmp_path


def test_config_path_uses_xdg_config_home(config_home) -> None:
    """The path is ``$XDG_CONFIG_HOME/tux/config.toml`` when the var is set."""
    assert config_path() == config_home / "tux" / "config.toml"


def test_config_path_falls_back_to_dot_config(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With ``XDG_CONFIG_HOME`` unset the path is ``~/.config/tux/config.toml``."""
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert config_path() == tmp_path / ".config" / "tux" / "config.toml"


def test_load_config_returns_empty_when_file_absent(config_home) -> None:
    """A missing file yields no overrides rather than an error."""
    assert load_config() == {}


def test_load_config_reads_known_keys(config_home) -> None:
    """``endpoint`` and ``model`` are read from the file."""
    config = config_home / "tux" / "config.toml"
    config.parent.mkdir(parents=True)
    config.write_text('endpoint = "http://host:8080"\nmodel = "m.gguf"\n')
    assert load_config() == {"endpoint": "http://host:8080", "model": "m.gguf"}


def test_load_config_raises_clear_error_on_malformed_toml(config_home) -> None:
    """Invalid TOML surfaces a ``ConfigError`` naming the file, not a traceback."""
    config = config_home / "tux" / "config.toml"
    config.parent.mkdir(parents=True)
    config.write_text("endpoint = not valid = toml\n")
    with pytest.raises(ConfigError) as excinfo:
        load_config()
    assert str(config) in str(excinfo.value)


def test_set_value_creates_file_and_parents(config_home) -> None:
    """Setting a value creates missing parent dirs and a round-trippable file."""
    set_value("endpoint", "http://new:9000")
    config = config_home / "tux" / "config.toml"
    assert config.exists()
    with config.open("rb") as handle:
        assert tomllib.load(handle) == {"endpoint": "http://new:9000"}


def test_set_value_preserves_other_key(config_home) -> None:
    """Setting one key leaves a previously set key intact."""
    set_value("endpoint", "http://new:9000")
    set_value("model", "custom.gguf")
    assert load_config() == {"endpoint": "http://new:9000", "model": "custom.gguf"}


def test_set_value_overwrites_existing_key(config_home) -> None:
    """Setting a key again replaces its previous value."""
    set_value("model", "first.gguf")
    set_value("model", "second.gguf")
    assert load_config() == {"model": "second.gguf"}


def test_set_value_round_trips_values_needing_escaping(config_home) -> None:
    """Values with quotes/backslashes are quoted so they round-trip through tomllib."""
    tricky = 'a"b\\c'
    set_value("model", tricky)
    assert load_config() == {"model": tricky}


def test_set_value_rejects_unknown_key_without_writing(config_home) -> None:
    """An unknown key raises ``ConfigError`` naming the allowed keys and writes nothing."""
    with pytest.raises(ConfigError) as excinfo:
        set_value("timeout", "30")
    message = str(excinfo.value)
    assert "endpoint" in message and "model" in message
    assert not config_path().exists()


def test_resolved_settings_marks_source(config_home) -> None:
    """Each setting is tagged ``config`` or ``default`` per where its value came from."""
    set_value("endpoint", "http://configured:8080")
    defaults = {"endpoint": "http://default:8080", "model": "default.gguf"}
    assert resolved_settings(defaults) == [
        ("endpoint", "http://configured:8080", "config"),
        ("model", "default.gguf", "default"),
    ]
