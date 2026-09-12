"""The licence notices must exist and must travel with the binaries.

v1.1.0 shipped three PyInstaller bundles containing PyQt6 (GPL-3.0-only) and
Qt 6 (LGPL v3) with **no licence text at all** -- simultaneously out of
compliance with MIT (the copyright notice must accompany every copy), GPL v3
(the licence and the corresponding-source offer must accompany the combined
work) and LGPL v3 s4(b) (a copy of both the GPL and the LGPL).

These tests pin the fix so a future edit to ``diablos.spec`` cannot silently
drop the notices again. They read the spec as text rather than importing it,
because it is a PyInstaller script that only runs under ``pyinstaller``.
"""

import ast
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Files the bundle must carry, and a phrase that proves the right one landed.
REQUIRED_NOTICES = {
    "LICENSE": "MIT License",
    "THIRD_PARTY_LICENSES.md": "Corresponding source",
    "licenses/GPL-3.0.txt": "GNU GENERAL PUBLIC LICENSE",
    "licenses/LGPL-3.0.txt": "GNU LESSER GENERAL PUBLIC LICENSE",
}


def _spec_datas():
    """The ``datas`` list from ``diablos.spec``, as (source, destination) pairs."""
    spec = (REPO_ROOT / "diablos.spec").read_text()
    match = re.search(r"^datas = (\[.*?^\])", spec, re.S | re.M)
    assert match, "could not find the datas list in diablos.spec"
    # The list holds one non-literal entry (VERSION_FILE); make it parseable.
    literal = match.group(1).replace("VERSION_FILE", "'VERSION_FILE'")
    return [tuple(entry) for entry in ast.literal_eval(literal)]


class TestNoticesExistInTheRepo:
    @pytest.mark.parametrize("relative_path, phrase", sorted(REQUIRED_NOTICES.items()))
    def test_file_is_present_and_is_the_right_licence(self, relative_path, phrase):
        path = REPO_ROOT / relative_path
        assert path.is_file(), f"{relative_path} is missing"
        assert phrase in path.read_text(), f"{relative_path} does not look like the right licence"

    def test_mit_grant_names_both_copyright_holders(self):
        """Upstream DiaBloS is MnRojas'; MIT requires its notice to be retained."""
        text = (REPO_ROOT / "LICENSE").read_text()
        assert "MnRojas" in text
        assert "DiaBloS Modern contributors" in text

    def test_third_party_notice_covers_the_gpl_binding(self):
        """PyQt6 being GPL-3.0-only is the whole reason a bundle is GPL."""
        text = (REPO_ROOT / "THIRD_PARTY_LICENSES.md").read_text()
        assert "GPL-3.0-only" in text
        assert "PyQt6" in text
        assert "LGPL v3" in text


class TestBundleShipsTheNotices:
    @pytest.mark.parametrize("relative_path", sorted(REQUIRED_NOTICES))
    def test_every_notice_is_in_the_spec_datas(self, relative_path):
        sources = {source for source, _destination in _spec_datas()}
        # licenses/*.txt travel as the whole directory.
        assert relative_path in sources or relative_path.split("/")[0] in sources

    def test_notices_land_next_to_the_executable(self):
        destinations = {source: dest for source, dest in _spec_datas()}
        assert destinations["LICENSE"] == "."
        assert destinations["THIRD_PARTY_LICENSES.md"] == "."
        assert destinations["licenses"] == "licenses"


class TestAboutDialogCarriesTheNotice:
    """LGPL v3 s4(c) wants the Qt notice among the notices shown at run time."""

    def test_about_text_names_the_licences_and_the_file(self):
        source = (REPO_ROOT / "modern_ui/builders/menu_builder.py").read_text()
        about = source[source.index("def _show_about") :]
        about = about[: about.index("\n    def ", 1)] if "\n    def " in about[1:] else about
        for expected in ("MIT", "LGPL v3", "GPL v3", "THIRD_PARTY_LICENSES.md"):
            assert expected in about, f"the About dialog no longer mentions {expected}"
