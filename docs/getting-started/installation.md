# Installation

There are two ways to get DiaBloS Modern: download a prebuilt app, or run it
from a source checkout. Start with the prebuilt app unless you intend to modify
the code.

## Prebuilt apps (recommended)

Every tagged release publishes three artifacts on the
[GitHub Releases page](https://github.com/Sapetor/diablos-modern/releases),
built by `.github/workflows/release.yml` from a green CI run:

| Platform | Artifact | How to run it |
|---|---|---|
| macOS (Apple Silicon) | `DiaBloS-<version>-arm64.dmg` | Open the DMG, drag the app to Applications |
| Windows (x64) | `DiaBloS-<version>-windows-x64.zip` | Extract the whole `DiaBloS` folder, run `DiaBloS.exe` |
| Linux (x86_64) | `DiaBloS-<version>-linux-x86_64.tar.gz` | `tar -xzf …`, then run `./DiaBloS/DiaBloS` |

No Python installation is needed — the interpreter and every dependency are
bundled.

!!! warning "The builds are unsigned"
    They are produced by a public CI workflow with no code-signing certificate,
    so both desktop platforms will object the first time.

    **macOS** — after dragging the app to Applications, clear the quarantine
    flag once:

    ```bash
    xattr -rd com.apple.quarantine /Applications/DiaBloS-arm64.app
    ```

    **Windows** — SmartScreen shows "Windows protected your PC". Choose
    **More info**, then **Run anyway**.

    Extract the Windows zip fully before running: `DiaBloS.exe` needs the rest
    of the `DiaBloS` folder next to it, and running it from inside the zip
    viewer will fail.

There is no Intel-macOS or ARM-Linux artifact in the automated release. Build
one yourself from source — see [Building & Packaging](../building.md).

## From source

DiaBloS needs **Python 3.9 or newer** and a GUI-capable environment. CI tests
3.9 and 3.12; 3.9 is the baseline, so the code avoids 3.10+-only syntax.

=== "venv"

    ```bash
    git clone https://github.com/Sapetor/diablos-modern.git
    cd diablos-modern

    python -m venv .venv
    source .venv/bin/activate        # Windows: .venv\Scripts\activate
    pip install -r requirements.txt

    python diablos_modern.py
    ```

=== "conda"

    ```bash
    git clone https://github.com/Sapetor/diablos-modern.git
    cd diablos-modern

    conda create -n diablos python=3.12
    conda activate diablos
    pip install -r requirements.txt

    python diablos_modern.py
    ```

    Install the requirements with `pip` inside the conda environment rather than
    from conda channels — the Qt build on some channels is too old (or is a
    PyQt5 build) and will not start the app.

Runtime dependencies (`requirements.txt`):

```
numpy>=1.20.0,<3.0
matplotlib>=3.5.0
tqdm>=4.60.0
pyqtgraph>=0.13.0
PyQt6>=6.7,<6.11; python_version < "3.10"
PyQt6>=6.7; python_version >= "3.10"
scipy>=1.6.0,<2.0
Pillow>=8.0.0          # GIF animation export
```

!!! note "Why PyQt6 is pinned differently on Python 3.9"
    The GUI runs on **PyQt6**. PyQt6 6.11 dropped Python 3.9 (its wheels are
    `cp310-abi3`), but 3.9 is still the supported baseline here, so a 3.9
    install resolves to the last 3.9-compatible series (6.10.x / 6.9.x) while
    3.10+ gets the current release. Nothing in the code depends on the newer
    series; the split exists only so `pip install -r requirements.txt` keeps
    working on 3.9. It collapses to a plain `PyQt6>=6.7` once the baseline
    moves to 3.10.

Optional external tool: **ffmpeg** for MP4 animation export from the field
scopes (`brew install ffmpeg`, `apt install ffmpeg`). GIF export needs only
Pillow, which is already a requirement.

### Linux system libraries

On a bare Ubuntu or a container, PyQt6 needs its Qt platform plugin libraries.
Qt 6 links a slightly wider set than Qt 5 did — `libxcb-cursor0` is the classic
addition, and its absence is the usual cause of *"could not load the Qt platform
plugin xcb"*. This is the same list the CI jobs install:

```bash
sudo apt-get install -y --no-install-recommends \
  libgl1 libegl1 libdbus-1-3 libfontconfig1 libfreetype6 \
  libxkbcommon0 libxkbcommon-x11-0 libxcb-cursor0 \
  libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 \
  libxcb-render-util0 libxcb-shape0 libxcb-xfixes0 libxcb-xinerama0
```

### Development install

```bash
pip install -r requirements-dev.txt

QT_QPA_PLATFORM=offscreen pytest      # the suite, headless
ruff check .                          # lint gate
ruff format --check .                 # format gate
```

`tests/conftest.py` forces `QT_QPA_PLATFORM=offscreen` and `MPLBACKEND=Agg` at
import time, so the suite is headless however you launch it. See the
[Developer Guide](../DEVELOPER_GUIDE.md).

## Headless and CLI use

DiaBloS has two subcommands that never build a GUI, so they work over SSH, in a
container and in CI:

```bash
# Simulate a diagram and write every Scope trace to CSV (or .npz)
python diablos_modern.py run model.diablos -o out.csv

# Override the diagram's stored settings for this run
python diablos_modern.py run model.diablos -o out.npz --time 30 --dt 0.005
python diablos_modern.py run model.diablos --solver interpreter
python diablos_modern.py run model.diablos --no-zero-crossing

# Export the diagram as a standalone numpy + scipy script
python diablos_modern.py export-python model.diablos -o model.py
```

Both accept `-q` / `--quiet` to suppress the summary line. Only `run` and
`export-python` are headless — every other invocation starts the GUI.

Opening the GUI on a file:

```bash
python diablos_modern.py path/to/model.diablos
```

If you are running the GUI itself in a headless environment (a test harness,
for example), set `QT_QPA_PLATFORM=offscreen`.

## Choosing the interface language

DiaBloS ships English and Spanish. Pick one from **View ▸ Language**:

- **System default** follows your OS locale (the packaged default).
- **English**
- **Español**

The choice is applied immediately and remembered across sessions. Windows that
are already open keep the language they were built with until you reopen them.

Adding another language is one JSON file and no code — see
[Language & Translations](../user-guide/localization.md).

## Where DiaBloS keeps your files

In a source checkout, user data lives in the project folder. In a packaged
build it goes to the platform's user-data directory:

| Platform | Location |
|---|---|
| macOS | `~/Library/Application Support/DiaBloS/` |
| Windows | `%APPDATA%\DiaBloS\` |
| Linux | `~/.local/share/DiaBloS/` |

That folder holds your writable `config/`, `user_preferences.json` and your
`library/` of saved blocks.

## Next steps

- [Quick start](quickstart.md) — build and run your first diagram
- [User Manual](../USER_MANUAL.md)
- [Building & Packaging](../building.md) — make your own binaries
