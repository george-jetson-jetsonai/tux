# Scripted build targets for tux. The .deb build bundles a Python interpreter
# (PyInstaller) so the package installs on a host with no Python; see
# packaging/README.md for the full story and prerequisites.

.PHONY: deb deb-lite clean

# Build the self-contained .deb into ./dist (the heavy, non-offline build step).
deb:
	python3 packaging/build_deb.py

# Build the tux-lite variant .deb (pins the lite tier at install time).
deb-lite:
	python3 packaging/build_deb.py --variant lite

# Remove build and packaging artifacts.
clean:
	rm -rf build dist src/*.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
