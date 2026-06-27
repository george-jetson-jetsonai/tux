# Packaging tux as a self-contained `.deb`

tux installs on a fresh Ubuntu machine **with no pre-existing Python toolchain**.
The `.deb` bundles its own Python interpreter, so you never touch
`pip`/`pipx`/`venv` — Python is an implementation detail hidden inside the
package.

## Headline install (download + install the `.deb`)

```sh
sudo apt install ./tux_<version>_<arch>.deb     # or: sudo dpkg -i tux_*.deb
tux --version
tux --help
```

That is the supported install path: download the `.deb`, install it, run `tux`.
Because the interpreter is bundled, the **no-Python-prerequisite guarantee** holds
regardless of how the `.deb` reaches the machine.

## First-run model provisioning

Installing the package does **not** silently download a multi-gigabyte model and
never hangs an unattended install on a prompt. Following Debian practice:

- The `postinst` asks a single **low-priority debconf** question, *"Provision
  tux's local model now?"*, defaulting to **no**. An unattended / non-interactive
  install takes the default and **defers** the model pull to first run.
- If you opt in (e.g. raise the debconf priority, or preseed
  `tux/provision-now=true`), the `postinst` runs `tux provision --yes` with
  consent recorded — it reads from `/dev/null` and never blocks on a TTY.
- Otherwise, run provisioning yourself whenever you like:

  ```sh
  tux provision
  ```

  This reuses tux's existing provisioning brain (8a): it sizes a model to your
  hardware, ensures the Ollama runtime, asks for consent before any download, and
  points tux's config at the local endpoint. After it completes, `tux ask "..."`
  works.

### Ollama runtime, and the no-Python-prerequisite guarantee

Provisioning installs the **Ollama** runtime with its **official install script**
(`https://ollama.com/install.sh`), which is a self-contained shell installer and
**introduces no Python toolchain**. We deliberately do not depend on an apt
`ollama` package (none is shipped in the Ubuntu archive) so the packaging story
does not undermine the bundled-interpreter guarantee. The package `Recommends:
curl`, used to fetch that script. Ollama and its pulled models are installed and
owned outside tux's package and are not managed by `dpkg`.

## Removal

```sh
sudo apt remove tux      # or: sudo dpkg -r tux
```

A plain `remove` deletes only the package's own files (the bundled `/usr/bin/tux`
binary). It does **not** touch:

- your per-user config and state under `$XDG_CONFIG_HOME/tux` (e.g.
  `~/.config/tux`), and
- the separately-installed Ollama runtime and any models you pulled.

`purge` additionally clears the package's own debconf answers; it still leaves
your per-user config and the Ollama runtime/models in place, since neither is
owned by this package.

## Building the `.deb`

The build is scripted and reproducible, and derives its version from
`tux.__version__` (never a hardcoded string):

```sh
make deb                       # -> ./dist/tux_<version>_<arch>.deb
# or:
python3 packaging/build_deb.py [--output-dir DIR] [--keep-build]
```

Build prerequisites (build host only — **not** required on the install target):

- **PyInstaller** — freezes tux and a Python interpreter into one executable.
- **dpkg-deb** — assembles the `.deb` (part of `dpkg`).
- **lintian** (optional) — validates the package; the build lints with
  `--fail-on error` when lintian is present.

The build (`packaging/build_deb.py`) bundles the binary, lays it out into a
`dpkg-deb` tree via `tux.packaging`, builds the `.deb`, and lints it. The heavy
binary build is intentionally kept out of the offline unit suite, which instead
asserts on the generated control metadata, debconf templates, maintainer
scripts, and build wiring directly (`tests/test_packaging.py`).

## Secondary route: pipx / PyPI (only if you already have Python)

If you *already* have a working Python and prefer it, tux can be installed from
source / PyPI with pipx:

```sh
pipx install tux
```

This is a secondary convenience, not the headline path — on Ubuntu 23.04+ a bare
`pip install` fails under PEP 668, and pipx itself is a chicken-and-egg
prerequisite, which is exactly why the `.deb` bundles the interpreter.

## Later distribution channel (out of scope here)

A gpg-signed **hosted apt repository** and a **Launchpad PPA** that would enable

```sh
sudo apt install tux
```

are a planned later distribution channel and are **out of scope** for this item.
Bundling the interpreter already satisfies the no-Python-prerequisite constraint,
so the directly-installable `.deb` above is the deliverable here.
