# Building DiaBloS as a Standalone App

DiaBloS can be packaged as a standalone app using PyInstaller. Users don't need Python installed.

## macOS (Quick Build)

```bash
# arm64 -- RECOMMENDED for releases (fast, working cursor)
source ~/.venvs/diablos-arm64/bin/activate
./tools/build.sh
# Output: dist/DiaBloS-arm64.app + dist/DiaBloS-arm64.dmg (72MB)

# x86_64 (Rosetta) -- fallback for older Intel Macs.
# Built from the x86_64 conda env under Rosetta (PyInstaller bundles the
# active interpreter, so the env MUST be x86_64 -- arm64 venvs produce arm64).
arch -x86_64 /bin/bash -c '
  source ~/opt/anaconda3/etc/profile.d/conda.sh
  conda activate diablos_x86
  ./tools/build.sh'
# Output: dist/DiaBloS-x86_64.app + dist/DiaBloS-x86_64.dmg (~117MB)

# Move DMG out of vault and clean up
mv dist/DiaBloS-*.dmg ~/Desktop/
rm -rf dist/DiaBloS-*.app build/
```

`tools/build.sh` runs three steps: sync block registry, PyInstaller build, DMG creation. App names include the architecture (`DiaBloS-arm64.app`, `DiaBloS-x86_64.app`) so both can coexist in `/Applications`.

> **arm64 cursor bug — FIXED.** PyQt5 5.15 (arm64) had a macOS bug where the text cursor was invisible in styled QLineEdit widgets (QTBUG-109450): the native `macintosh` style fails to draw the caret in any input with a stylesheet `background-color`. Fixed by switching the app to the Fusion style on macOS + Qt >= 5.10 (`_maybe_use_fusion_style` in `modern_ui/styles/qss_styles.py`); Fusion is stylesheet-aware and draws the caret itself. The fix is scoped so the x86_64/PyQt5-5.9 build (native style works there) keeps its native look. arm64 is now ~10x faster to start with a fully working cursor, so it is the preferred release.

## Build Venvs

Two separate venvs are used because PyInstaller bundles the Python interpreter from the active venv:

| Env | Python | PyQt5 | Arch | Status |
|------|--------|-------|------|--------|
| conda env `diablos_x86` (`~/opt/anaconda3/envs/diablos_x86`) | 3.9 (Anaconda) | 5.15.9 | x86_64 | x86_64 release (build via `arch -x86_64`) |
| `~/.venvs/diablos-arm64/` | 3.12 (Homebrew) | 5.15.11 | arm64 | **Recommended release** (Fusion cursor fix) |

Both envs need: `PyQt5 numpy scipy matplotlib pyqtgraph Pillow tqdm pyinstaller`.
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
| `tools/sync_block_registry.py` | Auto-scans `blocks/` and updates `_BLOCK_MODULES` in `block_loader.py` |
| `lib/app_paths.py` | Path resolver: `resource_path()` (read-only assets), `user_data_path()` (writable data) |
| `lib/block_loader.py` | `_BLOCK_MODULES` static registry for frozen mode |

## How It Works

- **Block discovery**: In dev mode, `block_loader.py` scans `blocks/` dynamically. In frozen mode, it uses `_BLOCK_MODULES`. The sync script (`tools/sync_block_registry.py`) keeps this list up to date -- run automatically by `tools/build.sh`.
- **Resource paths**: Read-only assets (icons, default configs, examples) use `resource_path()` which resolves to `sys._MEIPASS` when frozen. Writable data (logs, autosave, user configs) use `user_data_path()` which resolves to `~/Library/Application Support/DiaBloS/` on macOS.
- **Excluded packages**: `diablos.spec` excludes ~40 unused packages (torch, pandas, bokeh, selenium, etc.) to keep the bundle small. Only PyQt5, numpy, scipy, matplotlib, pyqtgraph, Pillow, and tqdm are included.
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
pip install PyQt5 numpy scipy matplotlib pyqtgraph Pillow tqdm pyinstaller

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
# 1. Install dependencies
sudo apt install python3 python3-venv python3-pip

# 2. Clone and setup
git clone git@github.com:Sapetor/diablos-modern.git
cd diablos-modern
python3 -m venv .venv
source .venv/bin/activate
pip install PyQt5 numpy scipy matplotlib pyqtgraph Pillow tqdm pyinstaller

# 3. Build
python tools/sync_block_registry.py
pyinstaller --noconfirm diablos.spec

# 4. Output: dist/DiaBloS/DiaBloS (distribute the entire dist/DiaBloS folder)
```
