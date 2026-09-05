"""Version resolution for ``modern_ui.__version__``.

``[project] version`` in pyproject.toml is the single source of truth; the spec,
tools/build.sh and the release workflow all parse it. ``modern_ui/__init__.py``
has to read it back at runtime, including in a PyInstaller bundle that ships no
``.dist-info``. This used to be a hard-coded ``_FALLBACK_VERSION = "1.0.0"``
literal, so every frozen build reported 1.0.0 forever.
"""

import os
import re

import pytest

import modern_ui

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _pyproject_version():
    """Independently parse the version out of pyproject.toml."""
    path = os.path.join(_PROJECT_ROOT, "pyproject.toml")
    with open(path, "r", encoding="utf-8") as fh:
        in_project = False
        for line in fh:
            stripped = line.strip()
            if stripped.startswith("["):
                in_project = stripped == "[project]"
                continue
            if in_project:
                match = re.match(r'^version\s*=\s*["\'](.+?)["\']', stripped)
                if match:
                    return match.group(1)
    return None


@pytest.mark.unit
class TestVersionResolution:
    def test_dev_checkout_reports_the_pyproject_version(self):
        """In a plain checkout the version must come from pyproject.toml."""
        expected = _pyproject_version()
        assert expected, "could not parse [project] version from pyproject.toml"
        assert modern_ui.__version__ == expected

    def test_pyproject_source_matches_the_file(self):
        assert modern_ui._version_from_pyproject() == _pyproject_version()

    def test_no_bundle_version_outside_a_frozen_build(self):
        """``sys._MEIPASS`` only exists under PyInstaller."""
        assert modern_ui._version_from_bundle() is None

    def test_bundle_version_wins_when_frozen(self, tmp_path, monkeypatch):
        """A frozen build reads ``_version.txt`` from ``sys._MEIPASS``.

        This is the case the old hard-coded fallback got wrong, and the one no
        dev checkout exercises, so simulate _MEIPASS.
        """
        (tmp_path / "_version.txt").write_text("9.8.7\n", encoding="utf-8")
        monkeypatch.setattr(modern_ui.sys, "_MEIPASS", str(tmp_path), raising=False)

        assert modern_ui._version_from_bundle() == "9.8.7"
        assert modern_ui.resolve_version() == "9.8.7"

    def test_falls_back_to_the_literal_when_every_source_fails(self, monkeypatch):
        monkeypatch.setattr(modern_ui, "_version_from_bundle", lambda: None)
        monkeypatch.setattr(modern_ui, "_version_from_pyproject", lambda: None)
        monkeypatch.setattr(modern_ui, "_version_from_metadata", lambda: None)

        assert modern_ui.resolve_version() == modern_ui._FALLBACK_VERSION

    def test_a_raising_source_does_not_break_startup(self, monkeypatch):
        """Version lookup runs at import time; it must never raise."""

        def _boom():
            raise RuntimeError("unreadable")

        monkeypatch.setattr(modern_ui, "_version_from_bundle", _boom)

        assert modern_ui.resolve_version() == _pyproject_version()

    def test_fallback_literal_is_kept_in_sync_with_pyproject(self):
        """The literal is documentation as much as code -- keep it honest."""
        assert modern_ui._FALLBACK_VERSION == _pyproject_version()
