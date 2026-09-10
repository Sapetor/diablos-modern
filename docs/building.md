# Building DiaBloS as a Standalone App

DiaBloS can be packaged as a standalone app using PyInstaller. Users don't need Python installed.

## macOS (Quick Build)

```bash
# arm64 -- RECOMMENDED for releases (fast, working cursor)
source ~/.venvs/diablos-arm64/bin/activate
./tools/build.sh
# Output: dist/DiaBloS-arm64.app + dist/DiaBloS-1.0.0-arm64.dmg (72MB)

# x86_64 (Rosetta) -- fallback for older Intel Macs.
# Built from the x86_64 conda env under Rosetta (PyInstaller bundles the
# active interpreter, so the env MUST be x86_64 -- arm64 venvs produce arm64).
arch -x86_64 /bin/bash -c '
  source ~/opt/anaconda3/etc/profile.d/conda.sh
  conda activate diablos_x86
  ./tools/build.sh'
# Output: dist/DiaBloS-x86_64.app + dist/DiaBloS-1.0.0-x86_64.dmg (~117MB)

# Move DMG out of vault and clean up
mv dist/DiaBloS-*.dmg ~/Desktop/
rm -rf dist/DiaBloS-*.app build/
```

`tools/build.sh` runs three steps: sync block registry, PyInstaller build, DMG creation. App names include the architecture (`DiaBloS-arm64.app`, `DiaBloS-x86_64.app`) so both can coexist in `/Applications`.

Both `tools/build.sh` and `diablos.spec` read the version from `[project] version` in `pyproject.toml` -- the single source of truth. `modern_ui.__version__` reads it back at runtime from, in order: `_version.txt` (written into the bundle by `diablos.spec`, since a frozen app ships no `.dist-info`), `pyproject.toml` itself (dev checkout), installed distribution metadata, then a literal fallback. A frozen build therefore reports the real version rather than a hard-coded one. The DMG is named `DiaBloS-<version>-<arch>.dmg` and the bundle's `CFBundleShortVersionString` matches it, so bumping one number in `pyproject.toml` updates the window title, the About/bundle version, and the installer filename together.

> **macOS cursor bug — FIXED.** The native macOS style fails to draw the blinking text caret in any QLineEdit that has a stylesheet `background-color` (QTBUG-109450) — which is every input field in this app. It first bit the PyQt5 5.15 arm64 builds and is not fixed in Qt 6 either. The app therefore switches to the Fusion style on macOS (`_maybe_use_fusion_style` in `modern_ui/styles/qss_styles.py`); Fusion is stylesheet-aware and draws the caret itself. arm64 is also ~10x faster to start than the Rosetta x86_64 build, so it is the preferred release.

## Build Venvs

Two separate venvs are used because PyInstaller bundles the Python interpreter from the active venv:

| Env | Python | PyQt6 | Arch | Status |
|------|--------|-------|------|--------|
| conda env `diablos_x86` (`~/opt/anaconda3/envs/diablos_x86`) | 3.9 (Anaconda) | 6.10.x (`<6.11`; 6.11 needs 3.10+) | x86_64 | x86_64 release (build via `arch -x86_64`) |
| `~/.venvs/diablos-arm64/` | 3.12 (Homebrew) | 6.11.x | arm64 | **Recommended release** (Fusion cursor fix) |

Both envs need: `PyQt6 numpy scipy matplotlib pyqtgraph Pillow tqdm pyinstaller`
-- install them with `pip install -r requirements.txt pyinstaller` so the
Python-version marker on the PyQt6 pin is honoured. The PyQt6 macOS wheels are
`universal2`, so the same wheel serves both arches; the interpreter's
architecture is what decides the build.

!!! note "Conda envs: disable Anaconda's `qt.conf`"
    Both envs were re-provisioned for PyQt6 on 2026-09-10 (`pip uninstall
    PyQt5 PyQt5-Qt5 PyQt5-sip`, then `pip install -r requirements.txt
    pyinstaller`). In the conda env that alone was not enough: Anaconda's own
    `qt-main` (Qt 5.15) package ships `<env>/bin/qt.conf`, which Qt reads from
    the directory of the running executable and which points every Qt -- PyQt6
    included -- at conda's Qt5 plugin directory. The symptom is
    *"This application failed to start because no Qt platform plugin could be
    initialized"* with `QT_DEBUG_PLUGINS=1` reporting *"uses incompatible Qt
    library (5.15.0)"*. Fix: rename `<env>/bin/qt.conf` (kept as
    `qt.conf.disabled-for-pyqt6`); a conda update of `qt-main` may put it back.
The x86_64 env is a conda env (not a `~/.venvs/` venv) and must be built under
Rosetta. PyInstaller cannot cross-compile -- it bundles whatever interpreter is
active, so an arm64 interpreter always yields an arm64 app regardless of flags.

## Output

| Build | DMG | App Size | Sim Speed | Startup | Cursor |
|-------|-----|----------|-----------|---------|--------|
| arm64 | 72MB | 160MB | ~80K itr/s | ~2s | Works (Fusion fix) |
| x86_64 | 113MB | 319MB | ~44K itr/s | ~25s (Rosetta) | Works |

## Key Files

| File | Purpose |
|------|---------|
| `diablos.spec` | PyInstaller config -- hidden imports, data files, excludes, platform packaging |
| `tools/build.sh` | One-command build: sync registry, PyInstaller, DMG |
| `pyproject.toml` | `[project] version` -- single source of truth for the app version (read by the spec, build script, and `modern_ui/__init__.py`) |
| `tools/sync_block_registry.py` | Auto-scans `blocks/` and updates `_BLOCK_MODULES` in `block_loader.py`; `--check` fails instead of writing (run in CI's lint job) |
| `lib/app_paths.py` | Path resolver: `resource_path()` (read-only assets), `user_data_path()` (writable data) |
| `lib/block_loader.py` | `_BLOCK_MODULES` static registry for frozen mode |

## How It Works

- **Block discovery**: In dev mode, `block_loader.py` scans `blocks/` dynamically. In frozen mode, it uses `_BLOCK_MODULES`. The sync script (`tools/sync_block_registry.py`) keeps this list up to date -- run automatically by `tools/build.sh`, and enforced in CI with `python tools/sync_block_registry.py --check`. The scan skips helper modules that define no block: anything listed in `EXCLUDED_MODULES`, plus any module whose name starts with `_` or ends with `_base`.
- **Resource paths**: Read-only assets (icons, default configs, examples) use `resource_path()` which resolves to `sys._MEIPASS` when frozen. Writable data (logs, autosave, user configs) use `user_data_path()` which resolves to `~/Library/Application Support/DiaBloS/` on macOS.
- **Excluded packages**: `diablos.spec` excludes ~40 unused packages (torch, pandas, bokeh, selenium, etc.) to keep the bundle small. Only PyQt6, numpy, scipy, matplotlib, pyqtgraph, Pillow, and tqdm are included. `PyQt5`, `PySide2` and `PySide6` are excluded explicitly: a dev machine often has more than one Qt binding installed, and pyqtgraph/matplotlib probe for all of them at import time, which would otherwise drag a second Qt runtime (~100 MB) into the bundle and let the two fight over the platform plugin at startup.
- **Optional SymPy**: the symbolic features (LaTeX/MathML export via `lib/export/latex_exporter.py`, `lib/engine/symbolic_engine.py`, and each block's `symbolic_execute()`) import SymPy lazily and degrade to a warning when it is missing. SymPy is *not* in `requirements.txt`, so a default build environment produces a bundle where those features are unavailable. `diablos.spec` no longer hard-excludes it: install it in the build env (`pip install sympy`, or `pip install .[symbolic]`) and the spec picks it up automatically via `collect_submodules('sympy')`. Expect roughly +35-40 MB unpacked. The published releases are currently built **without** SymPy -- symbolic export is a niche feature and the release job installs only `requirements.txt`; add `sympy` there if you want it shipped.
- **macOS activation**: Frozen builds use ObjC runtime calls via ctypes to register as a foreground app (required for Finder/Dock launches).
- **Multiprocessing**: `multiprocessing.freeze_support()` is called at entry point to prevent duplicate process spawning.

## Frozen-Mode Path Handling

In frozen mode, the working directory is `/` (read-only). All file I/O must use writable paths:

| Data | Dev Mode Path | Frozen Mode Path |
|------|--------------|-----------------|
| Configs | `config/` | `~/Library/Application Support/DiaBloS/config/` |
| Autosave | `saves/` | `~/Library/Application Support/DiaBloS/saves/` |
| Logs | `diablos_modern.log` | `~/Library/Logs/DiaBloS/diablos_modern.log` |
| Examples | `examples/` | `(bundled in app)/examples/` |

## macOS Distribution Notes

- **Unsigned apps**: Blocked by Gatekeeper. Users must run `xattr -rd com.apple.quarantine /Applications/DiaBloS-x86_64.app` after copying from DMG.
- **Code signing**: `codesign --force --deep --sign - DiaBloS-x86_64.app` for ad-hoc signing. For proper distribution, use an Apple Developer account ($99/yr) for signing + notarization.
- **App icon**: Set `icon='path/to/icon.icns'` in `diablos.spec` BUNDLE section.
- **Build artifacts**: Always move DMGs out of the vault (`~/Desktop/` or similar) and run `rm -rf dist/ build/` after building. Never leave binaries inside the Obsidian vault.

## Windows Build (Windows 10/11)

PyInstaller can only build for the platform it runs on. To build the Windows installer:

```powershell
# 1. Install Python 3.9+ from python.org (check "Add to PATH")
# 2. Clone the repo
git clone git@github.com:Sapetor/diablos-modern.git
cd diablos-modern

# 3. Create venv and install dependencies
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt pyinstaller

# 4. Sync block registry and build
python tools/sync_block_registry.py
pyinstaller --noconfirm diablos.spec

# 5. Output: dist\DiaBloS\DiaBloS.exe (distribute the entire dist\DiaBloS folder)
```

Also works from WSL with a Windows Python, or from a GitHub Actions CI workflow.

### Windows Distribution Notes
- Unsigned `.exe` may trigger Windows Defender SmartScreen warnings -- users click "More info" then "Run anyway"
- Distributing as a folder (not `--onefile`) reduces false positives
- Code signing certificate ($200-400/yr) eliminates warnings

## Ubuntu/Linux Build

```bash
# 1. Install dependencies (plus the Qt6 runtime libraries PyInstaller has to
#    load while collecting PyQt6 -- see docs/getting-started/installation.md
#    for the full list; libxcb-cursor0 is the Qt6-only addition)
sudo apt install python3 python3-venv python3-pip \
  libgl1 libegl1 libdbus-1-3 libfontconfig1 libfreetype6 \
  libxkbcommon0 libxkbcommon-x11-0 libxcb-cursor0 \
  libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 \
  libxcb-render-util0 libxcb-shape0 libxcb-xfixes0 libxcb-xinerama0

# 2. Clone and setup
git clone git@github.com:Sapetor/diablos-modern.git
cd diablos-modern
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt pyinstaller

# 3. Build
python tools/sync_block_registry.py
pyinstaller --noconfirm diablos.spec

# 4. Output: dist/DiaBloS/DiaBloS (distribute the entire dist/DiaBloS folder)
```

## Releasing

Releases are cut from a tag; `.github/workflows/release.yml` does the building.

1. **Bump the version** in `pyproject.toml` (`[project] version`). This is the
   only place the version is written -- `modern_ui.__version__` (window title),
   `diablos.spec` (`CFBundleShortVersionString`) and `tools/build.sh` (DMG
   filename) all read it back.
2. **Update `CHANGELOG.md`** with the changes under the new version heading.
3. **Commit, tag and push:**

   ```bash
   git commit -am "release: v1.0.0"
   git push
   git tag v1.0.0
   git push --tags
   ```

4. **The workflow builds the artifacts.** Pushing a `v*` tag runs
   `Release`, which first runs the whole CI workflow (`jobs.tests` reuses
   `.github/workflows/ci.yml` through `workflow_call`, so lint + the 3.9/3.12
   test matrix must be green), and only then builds on `macos-latest` (arm64,
   via `tools/build.sh`), `windows-latest` (PyInstaller + a zipped
   `dist/DiaBloS` folder) and `ubuntu-latest` (PyInstaller + a `dist/DiaBloS`
   tarball). It publishes a GitHub release with `DiaBloS-<version>-arm64.dmg`,
   `DiaBloS-<version>-windows-x64.zip` and
   `DiaBloS-<version>-linux-x86_64.tar.gz` attached.

The tag must match the version in `pyproject.toml` (`v1.0.0` <-> `1.0.0`) --
nothing enforces this, so check it before tagging. All three artifacts are
unsigned; the generated release notes tell users how to get past Gatekeeper and
SmartScreen. The macOS x86_64 (Rosetta) build is still manual (see above).
