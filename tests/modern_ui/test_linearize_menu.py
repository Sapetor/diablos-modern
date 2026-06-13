"""Integration wiring tests for the Analysis menu features.

The controllers / windows / dialogs are unit-tested separately; this checks the
features are reachable from the GUI: the Analysis menu exists with Linearize and
Monte Carlo actions, the window methods exist, and the empty-diagram paths are safe.
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

        # Monte Carlo is wired too.
        assert hasattr(w, "run_monte_carlo")
        assert any("Monte Carlo" in a.text() for a in analysis_menu.actions())

        # Empty diagram: both features show an info box (monkeypatched), no crash.
        import PyQt5.QtWidgets as qtw
        calls = []
        monkeypatch.setattr(qtw.QMessageBox, "information",
                            lambda *a, **k: calls.append(a))
        w.dsim.blocks_list = []
        w.linearize_and_analyze()
        w.run_monte_carlo()
        assert len(calls) == 2, "empty-diagram paths should inform the user"
    finally:
        w.close()
