# Building DiaBloS as a Standalone App

DiaBloS can be packaged as a standalone app using PyInstaller. Users don't need Python installed.

## macOS (Quick Build)

```bash
# x86_64 (Rosetta) -- RECOMMENDED for releases (fully functional)
source ~/.venvs/evaluate-diablos/bin/activate
./tools/build.sh
# Output: dist/DiaBloS-x86_64.app + dist/DiaBloS-x86_64.dmg (113MB)

# Move DMG out of vault and clean up
mv dist/DiaBloS-x86_64.dmg ~/Desktop/
rm -rf dist/DiaBloS-x86_64.app build/
```

`tools/build.sh` runs three steps: sync block registry, PyInstaller build, DMG creation. App names include the architecture (`DiaBloS-arm64.app`, `DiaBloS-x86_64.app`) so both can coexist in `/Applications`.

> **Why x86_64?** PyQt5 5.15 (the only version available for arm64) has an unfixable macOS bug where the text cursor is invisible in styled QLineEdit widgets (QTBUG-109450). PyQt5 5.9 (x86_64 via Anaconda/Rosetta) works correctly. The arm64 build is ~10x faster to start but has no visible cursor in the property editor. Use x86_64 for releases until a PyQt6 migration.

## Build Venvs

Two separate venvs are used because PyInstaller bundles the Python interpreter from the active venv:

| Venv | Python | PyQt5 | Arch | Status |
|------|--------|-------|------|--------|
| `~/.venvs/evaluate-diablos/` | 3.9 (Anaconda) | 5.9.7 | x86_64 | **Release build** |
| `~/.venvs/diablos-arm64/` | 3.12 (Homebrew) | 5.15.11 | arm64 | Dev/testing only (cursor bug) |

Both venvs need: `PyQt5 numpy scipy matplotlib pyqtgraph Pillow tqdm pyinstaller`

## Output

| Build | DMG | App Size | Sim Speed | Startup | Cursor |
|-------|-----|----------|-----------|---------|--------|
| x86_64 | 113MB | 319MB | ~44K itr/s | ~25s (Rosetta) | Works |
| arm64 | 72MB | 160MB | ~80K itr/s | ~2s | **Invisible** |

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
