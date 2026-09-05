"""
Modern UI package for DiaBloS
Provides modern theming, widgets, and styling components.

Version resolution
------------------
``[project] version`` in ``pyproject.toml`` is the single source of truth (it is
also parsed by ``diablos.spec``, ``tools/build.sh`` and the release workflow).
This module reads it back through, in order:

1. ``_version.txt`` bundled next to the frozen app's resources — written by
   ``diablos.spec`` at build time, because a PyInstaller bundle ships no
   ``.dist-info`` for ``importlib.metadata`` to find;
2. ``pyproject.toml`` itself, found relative to this file — the plain dev
   checkout, which is usually not ``pip install``-ed;
3. installed distribution metadata, for a real ``pip install``;
4. ``_FALLBACK_VERSION`` as a last resort.
"""

import os
import re
import sys
from importlib.metadata import version as _package_version

#: Last-resort literal, used only when every source above fails. Keep in sync
#: with ``[project] version`` in pyproject.toml, which is the single source of
#: truth; nothing in a normal dev checkout or frozen build should reach it.
_FALLBACK_VERSION = "1.0.0"

#: Plain-text file the PyInstaller spec drops at the root of the bundle.
_VERSION_FILENAME = "_version.txt"

_VERSION_RE = re.compile(r'^version\s*=\s*["\'](.+?)["\']')


def _version_from_bundle():
    """Read the version PyInstaller bundled at ``sys._MEIPASS/_version.txt``."""
    base = getattr(sys, "_MEIPASS", None)
    if not base:
        return None
    try:
        with open(os.path.join(base, _VERSION_FILENAME), "r", encoding="utf-8") as fh:
            return fh.read().strip() or None
    except OSError:
        return None


def _version_from_pyproject():
    """Parse ``[project] version`` out of the checkout's pyproject.toml.

    Regex-parsed rather than via ``tomllib`` so this works on the 3.9 baseline
    (tomllib is 3.11+); ``diablos.spec`` parses the same key the same way.
    """
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pyproject.toml"
    )
    try:
        with open(path, "r", encoding="utf-8") as fh:
            in_project = False
            for line in fh:
                stripped = line.strip()
                if stripped.startswith("["):
                    in_project = stripped == "[project]"
                    continue
                if in_project:
                    match = _VERSION_RE.match(stripped)
                    if match:
                        return match.group(1)
    except OSError:
        pass
    return None


def _version_from_metadata():
    """Read the version from installed distribution metadata."""
    try:
        return _package_version("diablos-modern")
    except Exception:
        # PackageNotFoundError in the normal case; a broad guard keeps a quirky
        # frozen-import environment from breaking application startup.
        return None


def resolve_version():
    """Return the application version from the first source that yields one."""
    for source in (_version_from_bundle, _version_from_pyproject, _version_from_metadata):
        try:
            found = source()
        except Exception:
            found = None
        if found:
            return found
    return _FALLBACK_VERSION


__version__ = resolve_version()

__author__ = "DiaBloS Development Team"
