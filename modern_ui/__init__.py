"""
Modern UI package for DiaBloS
Provides modern theming, widgets, and styling components.
"""

from importlib.metadata import version as _package_version

#: Used when the distribution metadata is unavailable — a plain dev checkout
#: that was never ``pip install``-ed, or a PyInstaller-frozen bundle (which
#: ships no ``.dist-info``). Keep in sync with ``[project] version`` in
#: pyproject.toml, which is the single source of truth.
_FALLBACK_VERSION = "1.0.0"

try:
    __version__ = _package_version("diablos-modern")
except Exception:
    # PackageNotFoundError in the normal case; a broad guard keeps a quirky
    # frozen-import environment from breaking application startup.
    __version__ = _FALLBACK_VERSION

__author__ = "DiaBloS Development Team"
