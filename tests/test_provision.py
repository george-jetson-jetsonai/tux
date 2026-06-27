"""Tests for the hardware-aware provisioning brain, all external effects mocked.

No real hardware probe, Ollama install, model download, GPU, or network is
touched: a fake runtime records what it was asked to do and the probe / consent /
reachability seams are injected, so the whole flow runs offline.
"""

import pytest

from tux.config import ConfigError, load_config, set_value
from tux.provision import (
    FULL_TIER,
    LOOKUP_TIER,
    MIN_FULL_VRAM_MB,
    OLLAMA_ENDPOINT,
    HardwareInfo,
    OllamaRuntime,
    decide_tier,
    pin_variant,
    provision,
)


@pytest.fixture
def config_home(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Point ``XDG_CONFIG_HOME`` at a temp dir so the real home is never touched."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    return tmp_path


class FakeRuntime(OllamaRuntime):
    """An Ollama runtime that records calls instead of touching the system."""

    def __init__(self, *, installed: bool, models: list[str] | None = None) -> None:
        self.installed = installed
        self.models = list(models or [])
        self.install_calls = 0
        self.pulled: list[str] = []

    def is_installed(self) -> bool:
        return self.installed

    def install(self) -> None:
        self.install_calls += 1
        self.installed = True

    def has_model(self, model: str) -> bool:
        return model in self.models

    def pull(self, model: str) -> None:
        self.pulled.append(model)
        self.models.append(model)


def _cpu_only() -> HardwareInfo:
    """A modest CPU-only host: no GPU, low RAM."""
    return HardwareInfo(cpu_count=4, ram_mb=8 * 1024, gpu_vendor=None, vram_mb=0)


def _capable_gpu() -> HardwareInfo:
    """A host with a capable GPU comfortably above the VRAM bar."""
    return HardwareInfo(
        cpu_count=16, ram_mb=64 * 1024, gpu_vendor="NVIDIA", vram_mb=24 * 1024
    )


# --- tier decision --------------------------------------------------------


def test_cpu_only_selects_lookup_tier() -> None:
    """A CPU-only / low-RAM machine selects the small lookup-only tier."""
    decision = decide_tier(_cpu_only())
    assert decision.tier is LOOKUP_TIER
    assert any("no GPU" in reason for reason in decision.reasons)


def test_capable_gpu_selects_full_tier() -> None:
    """A machine with enough VRAM selects the larger full-capability tier."""
    decision = decide_tier(_capable_gpu())
    assert decision.tier is FULL_TIER
    assert any("VRAM" in reason for reason in decision.reasons)


def test_gpu_below_vram_bar_falls_back_to_lookup() -> None:
    """A GPU under the VRAM bar still selects the lookup tier, and says why."""
    weak = HardwareInfo(
        cpu_count=8, ram_mb=16 * 1024, gpu_vendor="NVIDIA", vram_mb=MIN_FULL_VRAM_MB - 1
    )
    decision = decide_tier(weak)
    assert decision.tier is LOOKUP_TIER
    assert any("VRAM" in reason for reason in decision.reasons)


# --- provisioning flow ----------------------------------------------------


def test_install_runs_before_pull_when_ollama_absent(config_home) -> None:
    """With no Ollama, the runtime is installed before the model is pulled."""
    runtime = FakeRuntime(installed=False)
    result = provision(
        probe=_capable_gpu,
        runtime=runtime,
        reachable=lambda endpoint: True,
        assume_yes=True,
    )
    assert runtime.install_calls == 1
    assert result.ollama_installed is True
    assert runtime.pulled == [FULL_TIER.model]


def test_consent_prompt_surfaces_model_and_size_before_pull(config_home) -> None:
    """The consent prompt names the model and its download size before any pull."""
    runtime = FakeRuntime(installed=True)
    seen: list[str] = []

    def confirm(prompt: str) -> bool:
        seen.append(prompt)
        return True

    provision(
        probe=_cpu_only,
        runtime=runtime,
        confirm=confirm,
        reachable=lambda endpoint: True,
    )
    assert len(seen) == 1
    assert LOOKUP_TIER.model in seen[0]
    assert LOOKUP_TIER.download_size in seen[0]
    assert runtime.pulled == [LOOKUP_TIER.model]


def test_declined_consent_defers_pull(config_home) -> None:
    """Declining consent pulls nothing and records the pull as deferred."""
    runtime = FakeRuntime(installed=True)
    result = provision(
        probe=_cpu_only,
        runtime=runtime,
        confirm=lambda prompt: False,
        reachable=lambda endpoint: True,
    )
    assert runtime.pulled == []
    assert result.model_deferred is True
    assert result.model_pulled is False


def test_config_points_at_ollama_after_provisioning(config_home) -> None:
    """Config records the local Ollama endpoint, the model, and the variant."""
    runtime = FakeRuntime(installed=True)
    result = provision(
        probe=_capable_gpu,
        runtime=runtime,
        reachable=lambda endpoint: True,
        assume_yes=True,
    )
    config = load_config()
    assert config["endpoint"] == OLLAMA_ENDPOINT == result.endpoint
    assert config["model"] == FULL_TIER.model
    assert config["variant"] == FULL_TIER.variant


def test_rerun_is_idempotent(config_home) -> None:
    """Re-running with Ollama and the model present reinstalls and re-pulls nothing."""
    first = FakeRuntime(installed=False)
    provision(
        probe=_cpu_only, runtime=first, reachable=lambda endpoint: True, assume_yes=True
    )
    config_after_first = load_config()

    second = FakeRuntime(installed=True, models=[LOOKUP_TIER.model])
    result = provision(
        probe=_cpu_only,
        runtime=second,
        reachable=lambda endpoint: True,
        assume_yes=True,
    )
    assert second.install_calls == 0
    assert second.pulled == []
    assert result.ollama_installed is False
    assert result.model_pulled is False
    assert load_config() == config_after_first


def test_unattended_defers_pull_without_prompting(config_home) -> None:
    """A non-interactive run never prompts and defers the pull; it never hangs."""
    runtime = FakeRuntime(installed=True)

    def confirm(prompt: str) -> bool:
        raise AssertionError("must not prompt in a non-interactive run")

    result = provision(
        probe=_cpu_only,
        runtime=runtime,
        confirm=confirm,
        reachable=lambda endpoint: True,
        interactive=False,
    )
    assert runtime.pulled == []
    assert result.model_deferred is True
    # Config is still recorded so a first run can complete the deferred pull.
    assert load_config()["model"] == LOOKUP_TIER.model


def test_unattended_with_assume_yes_pulls(config_home) -> None:
    """``assume_yes`` (preseeded consent) pulls without prompting, even unattended."""
    runtime = FakeRuntime(installed=True)
    result = provision(
        probe=_cpu_only,
        runtime=runtime,
        confirm=lambda prompt: pytest.fail("must not prompt"),
        reachable=lambda endpoint: True,
        interactive=False,
        assume_yes=True,
    )
    assert runtime.pulled == [LOOKUP_TIER.model]
    assert result.model_pulled is True


def test_records_signals_that_drove_the_decision(config_home) -> None:
    """The result carries the human-readable signals behind the tier pick."""
    result = provision(
        probe=_capable_gpu,
        runtime=FakeRuntime(installed=True, models=[FULL_TIER.model]),
        reachable=lambda endpoint: True,
    )
    assert result.decision is not None
    assert result.decision.reasons
    assert any("VRAM" in reason for reason in result.decision.reasons)


def test_existing_endpoint_bypasses_provisioning(config_home) -> None:
    """A user-set endpoint (no variant recorded) bypasses provisioning entirely."""
    set_value("endpoint", "http://my-own-host:8080")
    runtime = FakeRuntime(installed=False)
    result = provision(
        probe=lambda: pytest.fail("must not probe when bypassing"),
        runtime=runtime,
        reachable=lambda endpoint: True,
    )
    assert result.bypassed is True
    assert runtime.install_calls == 0
    assert runtime.pulled == []
    # The user's endpoint is left untouched and no variant is forced on.
    assert load_config() == {"endpoint": "http://my-own-host:8080"}


def test_endpoint_reachability_checked_and_surfaced(config_home) -> None:
    """After a pull the endpoint is reachability-checked and the result surfaced."""
    checked: list[str] = []

    def reachable(endpoint: str) -> bool:
        checked.append(endpoint)
        return False

    result = provision(
        probe=_cpu_only,
        runtime=FakeRuntime(installed=True, models=[LOOKUP_TIER.model]),
        reachable=reachable,
    )
    assert checked == [OLLAMA_ENDPOINT]
    assert result.endpoint_reachable is False


def test_variant_install_seam_invoked_with_chosen_variant(config_home) -> None:
    """The variant-package install seam is called with the chosen variant."""
    installed: list[str] = []
    provision(
        probe=_capable_gpu,
        runtime=FakeRuntime(installed=True, models=[FULL_TIER.model]),
        install_variant=installed.append,
        reachable=lambda endpoint: True,
    )
    assert installed == [FULL_TIER.variant]


# --- variant pinning (tux-lite package) -----------------------------------


def test_lite_pin_overrides_gpu_probe(config_home) -> None:
    """Pinning lite stays lite on a GPU-capable host: the probe never upgrades it."""
    runtime = FakeRuntime(installed=True, models=[LOOKUP_TIER.model])
    result = provision(
        pin="lite",
        probe=lambda: pytest.fail("a pinned install must not probe hardware"),
        runtime=runtime,
        reachable=lambda endpoint: True,
        assume_yes=True,
    )
    assert result.variant == "lite"
    assert result.model == LOOKUP_TIER.model
    assert result.tier is LOOKUP_TIER
    config = load_config()
    assert config["variant"] == "lite"
    assert config["model"] == LOOKUP_TIER.model


def test_pin_records_the_pin_in_the_decision(config_home) -> None:
    """A pinned run reports the pin as the reason behind the tier."""
    result = provision(
        pin="lite",
        runtime=FakeRuntime(installed=True, models=[LOOKUP_TIER.model]),
        reachable=lambda endpoint: True,
    )
    assert result.decision is not None
    assert any("pinned" in reason for reason in result.decision.reasons)


def test_pin_fills_the_install_variant_seam(config_home) -> None:
    """A pinned run defaults the seam to the real pin (not 8a's record-only no-op)."""
    runtime = FakeRuntime(installed=True, models=[LOOKUP_TIER.model])
    # No config exists yet; pin_variant writing the variant proves the seam ran.
    provision(pin="lite", runtime=runtime, reachable=lambda endpoint: True)
    assert load_config()["variant"] == "lite"


def test_unknown_pin_is_rejected(config_home) -> None:
    """An unknown pinned variant is rejected before anything is installed."""
    runtime = FakeRuntime(installed=False)
    with pytest.raises(ConfigError):
        provision(pin="medium", runtime=runtime, reachable=lambda endpoint: True)
    assert runtime.install_calls == 0


def test_pin_variant_seam_writes_config(config_home) -> None:
    """The concrete ``pin_variant`` seam records the variant in config."""
    pin_variant("lite")
    assert load_config()["variant"] == "lite"
