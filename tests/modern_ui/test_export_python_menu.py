"""Wiring test for File > Export > Export as Python Script.

The generator itself is unit-tested in tests/unit/test_python_codegen.py; this
checks the feature is reachable from the GUI and that both outcomes behave:
a supported diagram writes a compilable script, an unsupported one warns and
writes nothing.
"""

import os

import pytest
from PyQt5.QtWidgets import QMenu


class _FakeBlock:
    def __init__(self, name, block_fn, params=None, in_ports=1, out_ports=1, username=""):
        self.name = name
        self.block_fn = block_fn
        self.params = params or {}
        self.in_ports = in_ports
        self.out_ports = out_ports
        self.username = username or name


class _FakeLine:
    def __init__(self, srcblock, srcport, dstblock, dstport):
        self.srcblock = srcblock
        self.srcport = srcport
        self.dstblock = dstblock
        self.dstport = dstport


def _step_gain_scope():
    blocks = [
        _FakeBlock("step0", "Step", {"value": 1.0, "delay": 0.0, "type": "up"}, 0, 1, "u"),
        _FakeBlock("gain1", "Gain", {"gain": 2.0}, 1, 1, "K"),
        _FakeBlock("scope2", "Scope", {"labels": "y"}, 1, 0, "scope"),
    ]
    lines = [
        _FakeLine("step0", 0, "gain1", 0),
        _FakeLine("gain1", 0, "scope2", 0),
    ]
    return blocks, lines


def _export_submenu(window):
    for act in window.menuBar().actions():
        if "File" not in act.text():
            continue
        for sub in act.menu().actions():
            if "xport" in sub.text() and isinstance(sub.menu(), QMenu):
                return sub.menu()
    return None


@pytest.mark.unit
def test_export_python_action_is_wired(qapp):
    from modern_ui.main_window import ModernDiaBloSWindow

    window = ModernDiaBloSWindow()
    try:
        assert hasattr(window, "export_python_script")
        export_menu = _export_submenu(window)
        assert export_menu is not None, "File > Export submenu not found"
        labels = [a.text() for a in export_menu.actions()]
        assert any("Python Script" in label for label in labels), labels
    finally:
        window.close()


@pytest.mark.unit
def test_empty_diagram_informs_and_does_not_prompt(qapp, monkeypatch):
    import PyQt5.QtWidgets as qtw
    from modern_ui.main_window import ModernDiaBloSWindow
    import modern_ui.tools.file_dialogs as file_dialogs

    window = ModernDiaBloSWindow()
    try:
        infos = []
        monkeypatch.setattr(qtw.QMessageBox, "information", lambda *a, **k: infos.append(a))
        monkeypatch.setattr(
            file_dialogs,
            "ask_save_path",
            lambda *a, **k: pytest.fail("must not prompt for an empty diagram"),
        )
        window.dsim.blocks_list = []
        window.export_python_script()
        assert len(infos) == 1
    finally:
        window.close()


@pytest.mark.unit
def test_writes_a_compilable_script(qapp, monkeypatch, tmp_path):
    from modern_ui.main_window import ModernDiaBloSWindow
    import modern_ui.tools.file_dialogs as file_dialogs

    window = ModernDiaBloSWindow()
    try:
        target = str(tmp_path / "model.py")
        monkeypatch.setattr(file_dialogs, "ask_save_path", lambda *a, **k: target)
        window.dsim.blocks_list, window.dsim.line_list = _step_gain_scope()

        window.export_python_script()

        assert os.path.exists(target)
        compile(open(target).read(), target, "exec")
    finally:
        window.close()


@pytest.mark.unit
def test_unsupported_blocks_warn_and_write_nothing(qapp, monkeypatch, tmp_path):
    import PyQt5.QtWidgets as qtw
    from modern_ui.main_window import ModernDiaBloSWindow
    import modern_ui.tools.file_dialogs as file_dialogs

    window = ModernDiaBloSWindow()
    try:
        target = str(tmp_path / "model.py")
        monkeypatch.setattr(file_dialogs, "ask_save_path", lambda *a, **k: target)
        warnings = []
        monkeypatch.setattr(qtw.QMessageBox, "warning", lambda *a, **k: warnings.append(a))

        blocks, lines = _step_gain_scope()
        blocks.append(_FakeBlock("noise3", "Noise", {}, 0, 1))
        window.dsim.blocks_list, window.dsim.line_list = blocks, lines

        window.export_python_script()

        assert len(warnings) == 1
        assert "noise3" in warnings[0][2], warnings[0]
        assert not os.path.exists(target)
    finally:
        window.close()
