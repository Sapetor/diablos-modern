"""
Path resolution for both development and PyInstaller-bundled modes.

Two path types:
  resource_path()  — Read-only bundled assets (default configs, locales, examples).
  user_data_path() — Writable user data (autosave, recent files, modified configs).

Three ways DiaBloS runs, and where each kind of path resolves:

  =================  ==========================  ===========================
  mode               resource_path()             user_data_path()
  =================  ==========================  ===========================
  source checkout    repository root             repository root
  (incl. ``pip install -e .``)
  installed wheel    ``diablos/resources/``      platform data dir (below)
  PyInstaller        ``sys._MEIPASS``            platform data dir
  =================  ==========================  ===========================

In a checkout the data folders (``config/``, ``locales/``, ``examples/``) live
at the repository root; ``pyproject.toml`` maps them into the wheel as
``diablos/resources/``, so the relative paths callers pass are the same in
every mode.

**Every runtime write must go through** :func:`user_data_path`,
:func:`user_saves_path` or :func:`user_logs_path` — never through a bare
relative literal (``open("saves/x.dat", "w")``) and never through
:func:`resource_path`. In a packaged build both the bundle (``sys._MEIPASS``)
and the process CWD are read-only (a macOS ``.app`` launched from Finder starts
with ``cwd == "/"``), so a relative write fails with
``[Errno 30] Read-only file system``. ``tests/unit/test_frozen_writes.py``
enforces this: it greps for relative-literal writes and it re-runs the real
autosave / config / export / run-history writers with ``sys.frozen`` faked and
a read-only CWD.
"""

import os
import sys

#: Name of the writable folder holding autosaves, data exports and run history.
#: Kept as a constant so nothing has to spell the literal ``"saves"`` again.
SAVES_DIR_NAME = "saves"

# QSettings org/app for UI preferences. Single source of truth shared by every
# call site (main_window.py first-run flag, modern_palette.py collapsed flags).
# KEEP these values — they name the existing on-disk store; changing them would
# orphan already-written settings.
SETTINGS_ORG = "DiaBloS"
SETTINGS_APP = "DiaBloS"


# src/diablos/core/app_paths.py: the package directory is one level up and,
# in a source checkout, the repository root three levels above that.
_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CHECKOUT_ROOT = os.path.dirname(os.path.dirname(_PACKAGE_DIR))


def checkout_root():
    """The repository root when running from a source checkout, else None."""
    root = _CHECKOUT_ROOT
    if os.path.isfile(os.path.join(root, "pyproject.toml")) and os.path.isdir(
        os.path.join(root, "config")
    ):
        return root
    return None


def get_base_path() -> str:
    """Return the base path for resolving bundled resource files (read-only)."""
    if getattr(sys, "frozen", False):
        return sys._MEIPASS
    return checkout_root() or os.path.join(_PACKAGE_DIR, "resources")


def resource_path(relative_path: str) -> str:
    """Resolve a path to a read-only bundled resource."""
    return os.path.join(get_base_path(), relative_path)


def locales_path(filename: str = "") -> str:
    """Resolve the bundled ``locales/`` directory (or a file inside it).

    Translation catalogs are read-only bundled resources, so they resolve the
    same way icons and default configs do. ``locales`` is listed in
    ``diablos.spec`` ``datas`` so frozen builds ship them too.
    """
    if filename:
        return resource_path(os.path.join("locales", filename))
    return resource_path("locales")


def get_user_data_dir() -> str:
    """Return a writable directory for user data (configs, autosave, etc.).

    Frozen (PyInstaller) and installed wheel:
      macOS:   ~/Library/Application Support/DiaBloS/
      Windows: %APPDATA%/DiaBloS/
      Linux:   ~/.local/share/DiaBloS/
    Source checkout: the repository root (same as get_base_path).
    """
    if not getattr(sys, "frozen", False):
        root = checkout_root()
        if root:
            return root

    if sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    elif sys.platform == "win32":
        base = os.environ.get("APPDATA", os.path.expanduser("~"))
    else:
        base = os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))

    data_dir = os.path.join(base, "DiaBloS")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def user_data_path(relative_path: str) -> str:
    """Resolve a path to a writable user data file."""
    full = os.path.join(get_user_data_dir(), relative_path)
    parent = os.path.dirname(full)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return full


def user_saves_dir(create: bool = True) -> str:
    """Return the writable ``saves/`` folder (autosaves, exports, run history).

    Dev: ``<project root>/saves``. Frozen: ``<user data dir>/saves``, i.e.
    ``~/Library/Application Support/DiaBloS/saves`` on macOS,
    ``%APPDATA%/DiaBloS/saves`` on Windows and
    ``~/.local/share/DiaBloS/saves`` elsewhere.

    This exists so no caller ever writes to the *relative* ``saves/`` again:
    the CWD of a packaged app is read-only (and often ``/``), which is what
    produced ``[Errno 30] Read-only file system: 'saves'``.
    """
    path = os.path.join(get_user_data_dir(), SAVES_DIR_NAME)
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def user_saves_path(relative_path: str = "") -> str:
    """Resolve a writable path inside :func:`user_saves_dir`.

    ``user_saves_path()`` returns the directory itself; passing a relative name
    returns a file inside it, with the parent directory created.
    """
    base = user_saves_dir(create=True)
    if not relative_path:
        return base
    full = os.path.join(base, relative_path)
    parent = os.path.dirname(full)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return full


def user_logs_dir() -> str:
    """Return the writable directory for log files.

    macOS keeps logs in the conventional ``~/Library/Logs/DiaBloS`` when frozen;
    every other frozen platform uses ``<user data dir>/logs``. In dev the logs
    stay in the project root, where the repo's ``.gitignore`` already expects
    them.
    """
    if not getattr(sys, "frozen", False):
        return get_user_data_dir()
    if sys.platform == "darwin":
        log_dir = os.path.expanduser("~/Library/Logs/DiaBloS")
    else:
        log_dir = os.path.join(get_user_data_dir(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def user_logs_path(filename: str) -> str:
    """Resolve a writable log-file path (absolute input is passed through)."""
    if os.path.isabs(filename):
        return filename
    return os.path.join(user_logs_dir(), filename)


def is_writable_dir(path: str) -> bool:
    """True when ``path`` is an existing directory the process may write to."""
    return bool(path) and os.path.isdir(path) and os.access(path, os.W_OK)


def writable_dir_or_saves(preferred: str) -> str:
    """Return ``preferred`` when writable, else the user ``saves/`` folder.

    Used for file-dialog starting directories: a frozen build points them at
    the bundled ``examples/`` folder, which is inside the read-only app bundle,
    so a *Save* dialog opening there hands the user a path they cannot write.
    """
    if is_writable_dir(preferred):
        return preferred
    try:
        return user_saves_dir(create=True)
    except OSError:
        return os.path.expanduser("~")


def ui_settings():
    """Return the shared ``QSettings`` store for UI preferences.

    Single accessor for the ``SETTINGS_ORG``/``SETTINGS_APP`` pair so every UI
    call site reads and writes the same store. ``QSettings`` is imported lazily
    to keep this module free of a Qt import at module scope.

    ``DIABLOS_SETTINGS_INI`` (a file path) swaps the native store -- the
    registry on Windows -- for a private INI file. ``tests/conftest.py`` sets it
    so the suite neither reads nor overwrites the developer's real preferences.
    """
    from PyQt6.QtCore import QSettings

    ini = os.environ.get("DIABLOS_SETTINGS_INI")
    if ini:
        return QSettings(ini, QSettings.Format.IniFormat)
    return QSettings(SETTINGS_ORG, SETTINGS_APP)
