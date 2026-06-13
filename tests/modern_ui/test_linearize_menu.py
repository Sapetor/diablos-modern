"""Integration wiring test for the Linearize & Analyze feature.

The controller / results window / dialog are unit-tested separately; this checks
that the feature is reachable from the GUI: the Analysis menu exists with a
Linearize action, the window method exists, and the empty-diagram path is safe.
"""

import pytest
from PyQt5.QtWidgets import QMenu


@pytest.mark.unit
def test_analysis_menu_and_method_wired(qapp, monkeypatch):
    from modern_ui.main_window import ModernDiaBloSWindow
    w = ModernDiaBloSWindow()
    try:
        assert hasattr(w, "linearize_and_analyze")

        # Top-level "Analysis" menu exists and contains a "Linearize" action.
        analysis_menu = None
        for act in w.menuBar().actions():
            if "Analysis" in act.text():
                analysis_menu = act.menu()
                break
        assert isinstance(analysis_menu, QMenu), "Analysis menu not found in menubar"
        assert any("Linearize" in a.text() for a in analysis_menu.actions())

        # Empty diagram: should show an info box (monkeypatched) and not crash.
        import PyQt5.QtWidgets as qtw
        calls = []
        monkeypatch.setattr(qtw.QMessageBox, "information",
                            lambda *a, **k: calls.append(a))
        w.dsim.blocks_list = []
        w.linearize_and_analyze()
        assert calls, "empty-diagram path should inform the user"
    finally:
        w.close()
