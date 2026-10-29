"""Opening a diagram in the real window keeps its project-local custom blocks.

The window used to rebuild the diagram first and register the project's
``blocks/`` folder afterwards, so on a cold open every custom block (and its
connections) was dropped -- and the missing-type warning, which compared the
file with the already-refreshed palette, reported nothing.
"""

import json

import pytest
from PyQt6.QtWidgets import QMessageBox

import lib.user_blocks as ub
from tests.unit.test_missing_and_project_blocks import write_project


@pytest.fixture
def userdata(tmp_path, monkeypatch):
    import lib.app_paths

    monkeypatch.delenv(ub.BLOCKS_ENV_VAR, raising=False)
    home = tmp_path / "userdata"
    home.mkdir()
    monkeypatch.setattr(ub, "get_user_data_dir", lambda: str(home))
    monkeypatch.setattr(lib.app_paths, "get_user_data_dir", lambda: str(home))
    monkeypatch.setattr(ub, "_SESSION_TRUSTED", set(), raising=False)
    ub.purge_user_modules()
    yield home
    ub.purge_user_modules()


@pytest.fixture
def window(qapp, userdata):
    """A fresh window per test: its palette must not know the project blocks."""
    from modern_ui.main_window import ModernDiaBloSWindow

    w = ModernDiaBloSWindow()
    yield w
    w.dsim.model.dirty = False
    w.close()


@pytest.fixture
def warnings_shown(monkeypatch):
    shown = []

    def _warning(parent, title, text, *args, **kwargs):
        shown.append((title, text))
        return QMessageBox.StandardButton.Ok

    monkeypatch.setattr("modern_ui.main_window.QMessageBox.warning", _warning)
    return shown


@pytest.fixture
def click_trust(monkeypatch):
    """Make the trust prompt answer "Trust and Load"."""

    def _clicked(box):
        return next(
            b for b in box.buttons() if box.buttonRole(b) == QMessageBox.ButtonRole.AcceptRole
        )

    monkeypatch.setattr(QMessageBox, "clickedButton", _clicked)


def live_blocks(window):
    return sorted(b.block_fn for b in window.dsim.blocks_list)


def live_placeholders(window):
    from lib.simulation.missing_block import find_missing_blocks

    return find_missing_blocks(window.dsim.blocks_list)


@pytest.mark.qt
class TestColdOpen:
    def test_trusted_project_keeps_every_block_and_connection(
        self, window, tmp_path, warnings_shown
    ):
        path = write_project(tmp_path / "p", trusted=True)
        assert not any(mb.block_fn == "ReviewTripler" for mb in window.dsim.menu_blocks)
        window.open_example(path)
        assert live_blocks(window) == ["Constant", "ReviewTripler", "Scope"]
        assert live_placeholders(window) == []
        assert len(window.dsim.line_list) == 2
        assert warnings_shown == []

    def test_open_dialog_path_registers_blocks_before_loading(
        self, window, tmp_path, monkeypatch, warnings_shown
    ):
        path = write_project(tmp_path / "p", trusted=True)
        monkeypatch.setattr(window.diagram_service, "choose_open_file", lambda: path)
        window.open_diagram()
        assert live_blocks(window) == ["Constant", "ReviewTripler", "Scope"]
        assert live_placeholders(window) == []
        assert window._current_diagram_path() == path

    def test_trusting_at_the_prompt_loads_them_on_first_open(
        self, window, tmp_path, click_trust, warnings_shown
    ):
        path = write_project(tmp_path / "p", trusted=False)
        window.open_example(path)
        assert ub.is_project_blocks_trusted(ub.project_blocks_dir(path))
        assert live_placeholders(window) == []
        assert "ReviewTripler" in live_blocks(window)
        assert warnings_shown == []


@pytest.mark.qt
class TestDeclinedTrust:
    def test_save_keeps_the_custom_block_records(self, window, tmp_path, warnings_shown):
        path = write_project(tmp_path / "p", trusted=False)
        window.open_example(path)  # the stubbed prompt declines
        assert [b.missing_block_type for b in live_placeholders(window)] == ["ReviewTripler"]
        out = str(tmp_path / "resaved.diablos")
        assert window.diagram_service.save_diagram(out)
        with open(out, encoding="utf-8") as fh:
            saved = json.load(fh)
        with open(path, encoding="utf-8") as fh:
            original = json.load(fh)
        assert sorted(b["block_fn"] for b in saved["blocks_data"]) == [
            "Constant",
            "ReviewTripler",
            "Scope",
        ]
        tripler = next(b for b in saved["blocks_data"] if b["block_fn"] == "ReviewTripler")
        before = next(b for b in original["blocks_data"] if b["block_fn"] == "ReviewTripler")
        assert tripler["params"] == before["params"]
        assert len(saved["lines_data"]) == 2

    def test_missing_type_warning_lists_the_type(self, window, tmp_path, warnings_shown):
        path = write_project(tmp_path / "p", trusted=False)
        window.open_example(path)
        assert len(warnings_shown) == 1
        assert "ReviewTripler" in warnings_shown[0][1]

    def test_trusting_later_and_reloading_restores_the_block(
        self, window, tmp_path, warnings_shown
    ):
        path = write_project(tmp_path / "p", trusted=False)
        window.open_example(path)
        ub.trust_project_blocks(ub.project_blocks_dir(path))
        window.reload_user_blocks(quiet=True)
        assert live_placeholders(window) == []
        assert "ReviewTripler" in live_blocks(window)
