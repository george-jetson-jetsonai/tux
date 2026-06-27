# tux

A terminal-native Linux-learning assistant for Ubuntu. You ask in plain English;
tux **proposes** the terminal command — with a one-line explanation — and *you*
decide whether to run it. The terminal is the subject, not just the delivery:
tux keeps beginners in the terminal where they actually learn Linux, instead of
pulling them into a GUI.

tux runs against a **local model** (via [Ollama](https://ollama.com) on
`localhost`), so it is local-first and privacy-respecting — your questions and
your machine's state never leave the box.

```
$ tux ask "how do I see what's using port 8080?"

  ┌─ Find the process listening on port 8080 ───────────────┐
  │ command:     sudo lsof -i :8080                          │
  │ description: lists the process holding TCP port 8080     │
  └─────────────────────────────────────────────────────────┘

    ▸ run        dismiss
```

Nothing runs until you pick **run**. Dismiss is the default.

## Install

tux ships as a single self-contained `.deb` that **bundles its own Python
interpreter** — a fresh Ubuntu machine with no Python toolchain gets a working
`tux` on `PATH`. `pip`/`pipx`/`venv` are never touched.

```sh
sudo apt install ./tux_<version>_<arch>.deb     # or: sudo dpkg -i tux_*.deb
tux --version
tux --help
```

Installing the package does **not** silently download a multi-gigabyte model or
hang an unattended install on a prompt. Provision the local model when you're
ready:

```sh
tux provision
```

`tux provision` sizes a model to your hardware, ensures the Ollama runtime, asks
for consent before any download, and points tux's config at the local endpoint.
After it completes, `tux ask "..."` just works. See
[`packaging/README.md`](packaging/README.md) for the full install, first-run
provisioning, and removal story.

## Usage

```sh
tux ask "..."          # ask a question; tux proposes a command
tux ask --new "..."    # start a fresh conversation thread
tux config show        # show the effective endpoint and model
tux config set ...     # point tux at a different endpoint/model
tux provision          # hardware-aware local model setup
```

- **Propose, never auto-run.** Every command is staged behind an explicit
  run/dismiss choice; dismiss is the default. Each command you run is appended to
  a run log (`$XDG_STATE_HOME/tux/history.log`); dismissed proposals and command
  output are never logged.
- **Destructive-command flagging.** Before the run/dismiss choice, tux statically
  flags dangerous proposals (`rm -rf`, `dd`, `mkfs`, writes to block devices, …)
  with a short plain-English reason — so you choose with the risk visible.
- **Per-terminal context.** Follow-up questions in the same terminal share
  context (the thread is keyed to your shell); `tux ask --new` resets it.
- **Configurable endpoint.** Point tux at any OpenAI-compatible
  `/v1/chat/completions` endpoint via `tux config` if you'd rather not use the
  bundled Ollama default.

## What tux does (lookup-first)

This release is **lookup-first**, sized to a small, CPU-runnable model
(`qwen2.5-coder:3b`, ~1.9 GB):

- `tux ask` turns a plain-English question into a single proposed command.
- Run/dismiss staging and the run log keep you in control of what executes.
- Destructive-command flagging surfaces risk before you choose.
- Conversational questions still get a reply, with a short nudge back toward
  command lookup.

It runs comfortably without a GPU, which is the point: a small local model is
enough for everyday command lookup, and everything stays on your machine.

## Roadmap

1. **`tux history`** — a viewer for the run log: list and inspect past runs. The
   log is already written today; this is the reader (and log pruning/rotation).

2. **"tux is thinking" indicator** — a waiting cue while tux is blocked on the
   model, instead of silent dead air on the first token.

3. **An "explain / why" affordance** — go deeper on *why* a command is the right
   one, beyond the one-line description. The explanation is the product, so this
   earns its own surface.

4. **Inline edit of a step** — tweak a proposed command in place before running
   it, alongside run / dismiss / clarify (builds on tux-full's stepwise walk).

5. **tux-full** — a larger-model (`qwen2.5-coder:14b`, GPU-tier) variant that adds
   **multi-step guided plans**: tux answers with an ordered plan (a discovery step
   like `pwd` before the action), walked one step at a time with run / dismiss /
   **clarify**, re-planning the rest as it learns from each step's output. Includes
   an opt-in **run all remaining steps** to execute the rest of a vetted plan
   without confirming each step (still respecting destructive-command flags). This
   is the next major milestone and the foundation for item 5 below.

5.1. **Redo thread persistence under `~/.local/state/tux/threads/` for the stepwise
  setup** — the stepwise walk now feeds **captured command output** into `history`,
  and `state.save_thread` persists `history` verbatim, so step output now lands on
  disk in the per-shell-PID thread file (`<ppid>.json`) — even though item 5
  deliberately kept output out of the **run log**. Rework what the thread persists vs.
  what stays in-process only (e.g. carry captured output in-memory for the re-plan but
  strip/cap it before `save_thread`, or bound/redact it on write), so the on-disk
  thread doesn't quietly become an output log. Revisit the TTL/staleness model too if
  the persisted shape changes. Surfaced 2026-06-25; backlogged.

## Development

Requires Python 3.11+. The runtime is stdlib-only; the test toolchain is the only
extra dependency.

```sh
# Set up a development virtualenv with an editable install + dev deps
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"

# Run the test suite (runs fully offline — no GPU, no model, no network)
python3 -m pytest
```

Building the `.deb` (PyInstaller + dpkg-deb):

```sh
make deb            # default build
make deb-lite       # tux-lite variant (pins the lite tier at install time)
make clean          # remove build/packaging artifacts
```

Layout:

```
src/tux/        package source
  cli.py        argparse surface: ask / config / provision
  client.py     OpenAI-compatible streaming model client (injectable)
  safety.py     pure destructive-command detector
  provision.py  hardware probe → tier → Ollama → consented pull → config
  packaging.py  dpkg-deb tree + control/maintainer scripts
  state.py      per-shell-PID conversation thread persistence
tests/          offline unit suite (injectable seams; no real model/GPU)
packaging/      .deb build script and packaging docs
```

The model client, command runner, menu chooser, and every external effect
(probe, runtime install, downloads, config writes) sit behind injectable seams,
so the entire suite runs offline with no GPU and no real model.

## Acknowledgements

tux was developed with [Claude Code](https://claude.com/claude-code), Anthropic's
agentic coding tool.
