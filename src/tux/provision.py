"""Hardware-aware provisioning brain for a fresh tux install.

Provisioning probes the host's hardware, maps it to a model **tier** — a small
lookup-only model on CPU/low-RAM, a larger full-capability model on a capable GPU
— ensures the **Ollama** runtime is installed, pulls the tier's model (after
surfacing its download size and getting consent, never silently), checks the
local endpoint is reachable, and records the endpoint / model / variant in tux's
config so a fresh ``tux ask`` works with no further setup.

This is item 8a: it makes the variant **decision** and exposes the
variant-package install **seam**; building and gating the concrete
``tux-lite`` / ``tux-full`` packages is a separate sibling item, and the
``.deb`` / apt mechanics are 8b. Re-running is idempotent — an already-installed
runtime and an already-pulled model are left untouched and the config converges
to the same values.

Every external effect — the hardware probe, the Ollama runtime, the consent
prompt, the endpoint-reachability check, the config writer, and the
variant-package install — is an injectable seam, so the whole flow runs under the
test suite offline with no real install, download, or GPU.
"""

import os
import shutil
import subprocess
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from tux.config import ConfigError, load_config, set_value

#: OpenAI-compatible base URL Ollama serves locally. tux's model client appends
#: ``/v1/chat/completions`` to the stored endpoint, which matches Ollama's
#: OpenAI-compatible route, so the base (no ``/v1``) is what gets recorded.
OLLAMA_ENDPOINT = "http://localhost:11434"

#: Official one-line Ollama installer. Used only by the default runtime's
#: :meth:`OllamaRuntime.install`, which is mocked out in the test suite.
OLLAMA_INSTALL_URL = "https://ollama.com/install.sh"

#: A GPU clears the bar for the full tier only with at least this much VRAM (MB).
#: Kept deliberately simple and transparent; the chosen tier records the VRAM it
#: saw so the user can see and override the decision.
MIN_FULL_VRAM_MB = 8 * 1024


@dataclass(frozen=True)
class HardwareInfo:
    """A snapshot of the signals the tier decision is allowed to use.

    ``gpu_vendor`` is ``None`` when no GPU was detected, in which case
    ``vram_mb`` is ``0``.
    """

    cpu_count: int
    ram_mb: int
    gpu_vendor: str | None
    vram_mb: int


@dataclass(frozen=True)
class Tier:
    """A capability tier: the variant package and the Ollama model it runs."""

    name: str
    variant: str
    model: str
    download_size: str
    capability: str


#: Lookup-only tier for CPU-only / low-RAM hosts: a small instruction-tuned model
#: sized so common command lookups stay correct with bearable latency.
LOOKUP_TIER = Tier(
    name="lookup",
    variant="lite",
    model="qwen2.5-coder:3b",
    download_size="1.9 GB",
    capability="lookup-only",
)

#: Full-capability tier for hosts with a capable GPU: a larger model that backs
#: the richer stepwise and conversational surfaces.
FULL_TIER = Tier(
    name="full",
    variant="full",
    model="qwen2.5-coder:14b",
    download_size="9.0 GB",
    capability="full-capability",
)

#: Variant name → tier, for a pinned (packaged) install that forces a tier
#: regardless of what the hardware probe would otherwise pick. The variant
#: packages (e.g. tux-lite) provision with their tier pinned through this map so
#: a tux-lite install on a GPU-capable host stays lite.
TIERS = {LOOKUP_TIER.variant: LOOKUP_TIER, FULL_TIER.variant: FULL_TIER}


@dataclass(frozen=True)
class TierDecision:
    """The chosen tier plus the human-readable signals that drove the pick."""

    tier: Tier
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ProvisionResult:
    """The outcome of a provisioning run, for the caller to report.

    ``tier`` and ``decision`` are ``None`` only on a bypassed run (the user
    pointed tux at their own endpoint, so no decision was made).
    ``endpoint_reachable`` is ``None`` when reachability was not checked — a
    deferred pull or a bypass leaves it unknown.
    """

    tier: Tier | None
    decision: TierDecision | None
    ollama_installed: bool
    model_pulled: bool
    model_deferred: bool
    bypassed: bool
    endpoint_reachable: bool | None
    endpoint: str
    model: str
    variant: str


#: A probe returns the host's hardware snapshot.
Probe = Callable[[], HardwareInfo]

#: A confirm callable shows the consent prompt and returns the user's yes/no.
Confirm = Callable[[str], bool]

#: A config writer persists one ``key = value`` pair (mirrors ``config.set_value``).
ConfigWriter = Callable[[str, str], None]

#: A variant installer installs the chosen variant package. The default is a
#: no-op seam — the concrete ``tux-lite`` / ``tux-full`` packages are a separate
#: item; 8a only records the decision and marks where the install would happen.
VariantInstaller = Callable[[str], None]

#: A reachability check returns whether the endpoint base URL answers.
Reachable = Callable[[str], bool]


class OllamaRuntime:
    """The Ollama CLI as an injectable seam: presence, install, models, pull.

    The default methods shell out to the real ``ollama`` binary / installer; the
    test suite injects a fake instance so the flow runs with no real install or
    download.
    """

    def is_installed(self) -> bool:
        """Return whether the ``ollama`` binary is on ``PATH``."""
        return shutil.which("ollama") is not None

    def install(self) -> None:
        """Install the Ollama runtime via its official installer script.

        Raises:
            OSError: If the installer cannot be fetched or run.
            subprocess.CalledProcessError: If the installer exits non-zero.
        """
        subprocess.run(
            f"curl -fsSL {OLLAMA_INSTALL_URL} | sh",
            shell=True,
            check=True,
        )

    def has_model(self, model: str) -> bool:
        """Return whether ``model`` has already been pulled.

        Raises:
            subprocess.CalledProcessError: If ``ollama list`` exits non-zero.
        """
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True,
        )
        names = [line.split()[0] for line in result.stdout.splitlines()[1:] if line.split()]
        return model in names

    def pull(self, model: str) -> None:
        """Pull ``model`` via Ollama.

        Raises:
            subprocess.CalledProcessError: If the pull exits non-zero.
        """
        subprocess.run(["ollama", "pull", model], check=True)


def decide_tier(hardware: HardwareInfo) -> TierDecision:
    """Map probed hardware to a model tier, recording the signals that drove it.

    A capable GPU (a detected vendor with at least :data:`MIN_FULL_VRAM_MB` of
    VRAM) selects the full tier; everything else — no GPU, or a GPU below the
    VRAM bar — selects the lookup-only tier. The returned reasons make the pick
    transparent so the user can see why and override it.
    """
    if hardware.gpu_vendor and hardware.vram_mb >= MIN_FULL_VRAM_MB:
        reason = (
            f"{hardware.gpu_vendor} GPU with {hardware.vram_mb} MB VRAM "
            f"(≥ {MIN_FULL_VRAM_MB} MB)"
        )
        return TierDecision(FULL_TIER, (reason,))
    if not hardware.gpu_vendor:
        gpu_reason = "no GPU detected"
    else:
        gpu_reason = (
            f"{hardware.gpu_vendor} GPU with {hardware.vram_mb} MB VRAM "
            f"(< {MIN_FULL_VRAM_MB} MB)"
        )
    host_reason = f"{hardware.ram_mb} MB system RAM, {hardware.cpu_count} CPU(s)"
    return TierDecision(LOOKUP_TIER, (gpu_reason, host_reason))


def probe_hardware() -> HardwareInfo:
    """Return the host's hardware snapshot from simple, transparent sources."""
    return HardwareInfo(
        cpu_count=_probe_cpu_count(),
        ram_mb=_probe_ram_mb(),
        gpu_vendor=_probe_gpu_vendor(),
        vram_mb=_probe_vram_mb(),
    )


def _probe_cpu_count() -> int:
    """Return the usable CPU count, falling back to ``1`` when unknown."""
    return os.cpu_count() or 1


def _probe_ram_mb() -> int:
    """Return total system RAM in MB from ``/proc/meminfo`` (``0`` if unreadable).

    A missing or unparsable ``/proc/meminfo`` means the RAM signal is unknown,
    which the tier decision treats as low — the safe (lookup-only) direction.
    """
    try:
        meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
    except OSError:
        return 0
    for line in meminfo.splitlines():
        if line.startswith("MemTotal:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1]) // 1024  # MemTotal is reported in kB
    return 0


def _nvidia_vram_mb() -> int | None:
    """Return NVIDIA VRAM in MB via ``nvidia-smi``, or ``None`` when absent.

    A missing ``nvidia-smi`` (``FileNotFoundError``) or a non-zero exit means no
    usable NVIDIA GPU, which is reported as ``None`` rather than an error.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    first = result.stdout.strip().splitlines()
    if first and first[0].strip().isdigit():
        return int(first[0].strip())
    return None


def _probe_gpu_vendor() -> str | None:
    """Return the GPU vendor string, or ``None`` when no GPU is detected."""
    if _nvidia_vram_mb() is not None:
        return "NVIDIA"
    return None


def _probe_vram_mb() -> int:
    """Return detected VRAM in MB, or ``0`` when no GPU is detected."""
    return _nvidia_vram_mb() or 0


def _consent_prompt(tier: Tier) -> str:
    """Return the one-line consent prompt naming the model and its download size."""
    return (
        f"tux will download the {tier.capability} model '{tier.model}' "
        f"(~{tier.download_size}) via Ollama. Proceed? [y/N] "
    )


def confirm_pull(prompt: str) -> bool:
    """Read a yes/no answer for the model-pull consent prompt from stdin.

    Anything other than ``y``/``yes`` (including end-of-input) is treated as no,
    so a stray keystroke never triggers a multi-GB download.
    """
    try:
        answer = input(prompt).strip().lower()
    except EOFError:
        print()
        return False
    return answer in {"y", "yes"}


def endpoint_reachable(endpoint: str) -> bool:
    """Return whether the endpoint base URL answers an HTTP request.

    A network failure (``urllib.error.URLError`` / ``TimeoutError``, both
    ``OSError`` subclasses) means not reachable rather than an error, since this
    is only a post-pull readiness check.
    """
    try:
        with urllib.request.urlopen(f"{endpoint}/", timeout=5.0):
            return True
    except urllib.error.HTTPError:
        # An HTTP error response still proves the endpoint is answering.
        return True
    except OSError:
        return False


def _record_variant_only(variant: str) -> None:
    """No-op variant-install seam: record the decision without installing.

    8a only records the chosen variant; the generic (unpinned) provisioning path
    keeps this no-op so the escape hatch and existing behavior are untouched. A
    pinned, packaged install fills the seam with :func:`pin_variant` instead.
    """


def pin_variant(variant: str) -> None:
    """Pin the resolved variant into config: the concrete variant-package seam.

    Where :func:`_record_variant_only` is 8a's no-op placeholder, a real variant
    package pins its variant so a later hardware probe can never upgrade, say, a
    tux-lite install to full. It is idempotent — it re-asserts the same variant
    provisioning already records — so re-running converges on the pinned value.
    """
    set_value("variant", variant)


def provision(
    *,
    probe: Probe = probe_hardware,
    runtime: OllamaRuntime | None = None,
    confirm: Confirm = confirm_pull,
    writer: ConfigWriter = set_value,
    install_variant: VariantInstaller | None = None,
    reachable: Reachable = endpoint_reachable,
    interactive: bool = True,
    assume_yes: bool = False,
    pin: str | None = None,
) -> ProvisionResult:
    """Provision a working local model sized to the host and record it in config.

    The host is probed, mapped to a tier, the Ollama runtime is ensured, the
    tier's model is pulled (with consent), the endpoint is reachability-checked,
    and the endpoint / model / variant are written to config. The run is
    idempotent: an already-installed runtime and an already-pulled model are left
    untouched and the config converges to the same values.

    The escape hatch: if config already names an ``endpoint`` but no ``variant``,
    the user pointed tux at their own endpoint before provisioning ran, so
    provisioning bypasses entirely and changes nothing.

    Consent never blocks an unattended install. When the tier's model is absent:
    ``assume_yes`` pulls without prompting; otherwise an interactive run asks via
    ``confirm`` and a non-interactive run defers the consent and pull to first
    run (it never prompts and never pulls silently).

    Args:
        probe: Hardware probe seam; injected in tests to supply a fixed host.
        runtime: Ollama runtime seam; defaults to the real CLI-backed runtime.
        confirm: Consent prompt seam returning the user's yes/no.
        writer: Config writer seam; defaults to ``config.set_value``.
        install_variant: Variant-package install seam. When ``None`` (the
            default) it resolves to :func:`pin_variant` for a pinned, packaged
            install and to 8a's :func:`_record_variant_only` no-op otherwise.
        reachable: Endpoint reachability seam; injected in tests.
        interactive: Whether a prompt may be shown. False defers the pull.
        assume_yes: Treat consent as already granted (e.g. a preseeded install).
        pin: When set to a known variant name, force that variant's tier instead
            of probing hardware, so a variant package (e.g. tux-lite) stays on
            its tier on any host. ``None`` keeps the hardware-probed tiering.

    Returns:
        A :class:`ProvisionResult` describing what the run did.

    Raises:
        ConfigError: If ``pin`` names an unknown variant.
    """
    runtime = runtime if runtime is not None else OllamaRuntime()
    if install_variant is None:
        install_variant = pin_variant if pin is not None else _record_variant_only

    existing = load_config()
    if "endpoint" in existing and "variant" not in existing:
        return ProvisionResult(
            tier=None,
            decision=None,
            ollama_installed=False,
            model_pulled=False,
            model_deferred=False,
            bypassed=True,
            endpoint_reachable=None,
            endpoint=existing.get("endpoint", ""),
            model=existing.get("model", ""),
            variant=existing.get("variant", ""),
        )

    if pin is not None:
        if pin not in TIERS:
            allowed = ", ".join(sorted(TIERS))
            raise ConfigError(f"unknown variant {pin!r}; allowed variants are: {allowed}")
        tier = TIERS[pin]
        decision = TierDecision(
            tier, (f"variant pinned to {pin!r} for the tux-{pin} package",)
        )
    else:
        decision = decide_tier(probe())
        tier = decision.tier

    ollama_installed = False
    if not runtime.is_installed():
        runtime.install()
        ollama_installed = True

    model_pulled = False
    model_deferred = False
    if runtime.has_model(tier.model):
        pass  # already pulled — idempotent, no re-pull
    elif assume_yes:
        runtime.pull(tier.model)
        model_pulled = True
    elif interactive:
        if confirm(_consent_prompt(tier)):
            runtime.pull(tier.model)
            model_pulled = True
        else:
            model_deferred = True
    else:
        # Unattended without preseeded consent: never hang, never pull silently;
        # defer the consent and pull to first run.
        model_deferred = True

    install_variant(tier.variant)

    writer("endpoint", OLLAMA_ENDPOINT)
    writer("model", tier.model)
    writer("variant", tier.variant)

    # Reachability is only meaningful once the model is in place; a deferred pull
    # leaves it unknown for the first run to settle.
    reachable_now = None if model_deferred else reachable(OLLAMA_ENDPOINT)

    return ProvisionResult(
        tier=tier,
        decision=decision,
        ollama_installed=ollama_installed,
        model_pulled=model_pulled,
        model_deferred=model_deferred,
        bypassed=False,
        endpoint_reachable=reachable_now,
        endpoint=OLLAMA_ENDPOINT,
        model=tier.model,
        variant=tier.variant,
    )
