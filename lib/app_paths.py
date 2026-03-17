"""
Path resolution for both development and PyInstaller-bundled modes.

Two path types:
  resource_path()  — Read-only bundled assets (icons, default configs, examples).
                     Resolves to sys._MEIPASS when frozen, project root in dev.
  user_data_path() — Writable user data (autosave, recent files, modified configs).
                     Resolves to a platform-appropriate directory when frozen,
                     project root in dev.
"""

import os
import sys


def get_base_path() -> str:
    """Return the base path for resolving bundled resource files (read-only)."""
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resource_path(relative_path: str) -> str:
    """Resolve a path to a read-only bundled resource."""
    return os.path.join(get_base_path(), relative_path)


def get_user_data_dir() -> str:
    """Return a writable directory for user data (configs, autosave, etc.).

    Frozen (PyInstaller):
      macOS:   ~/Library/Application Support/DiaBloS/
      Windows: %APPDATA%/DiaBloS/
      Linux:   ~/.local/share/DiaBloS/
    Development: project root (same as get_base_path).
    """
    if not getattr(sys, 'frozen', False):
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if sys.platform == 'darwin':
        base = os.path.expanduser('~/Library/Application Support')
    elif sys.platform == 'win32':
        base = os.environ.get('APPDATA', os.path.expanduser('~'))
    else:
        base = os.environ.get('XDG_DATA_HOME', os.path.expanduser('~/.local/share'))

    data_dir = os.path.join(base, 'DiaBloS')
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def user_data_path(relative_path: str) -> str:
    """Resolve a path to a writable user data file."""
    full = os.path.join(get_user_data_dir(), relative_path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    return full
