"""How a stiff run reaches the user: a status line and a property-editor row.

The engine records the verdict (``lib/engine/solver_diagnostics.py``); these are
the two places it is surfaced. Both must be silent when the heuristic did not
fire and neither may ever raise -- a diagnostic that breaks the end of a good
run is worse than no diagnostic.
"""

import types

import pytest
from PyQt6.QtWidgets import QLabel

from modern_ui.controllers.simulation_controller import SimulationController
from modern_ui.widgets.property_editor import PropertyEditor


def _dsim_with(diagnostics, solver_method="RK45", fallback_reason=None):
    """A DSim stub whose engine reports the given solver diagnostics."""
    engine = types.SimpleNamespace(
        get_solver_diagnostics=lambda: dict(diagnostics),
        get_compile_fallback_reason=lambda: fallback_reason,
    )
    return types.SimpleNamespace(
        engine=engine,
        blocks_list=[],
        line_list=[],
        solver_method=solver_method,
        sim_dt=0.01,
        sim_time=1.0,
        rtol=1e-9,
        atol=1e-12,
        zero_crossing=True,
        scope_plotter=None,
        current_filepath=None,
    )


SUSPECTED = {
    "method_used": "RK45",
    "stiffness_suspected": True,
    "stiffness": {
        "suspected": True,
        "work_ratio": 53.6,
        "stiffness_index": 150.0,
        "eig_ratio": 9e6,
        "suggested_method": "LSODA",
    },
}

NOT_SUSPECTED = {
    "method_used": "RK45",
    "stiffness_suspected": False,
    "stiffness": {
        "suspected": False,
        "work_ratio": 0.3,
        "stiffness_index": 0.6,
        "eig_ratio": 1.0,
        "suggested_method": "LSODA",
    },
}


@pytest.mark.qt
class TestControllerStatusMessage:
    def _statuses(self, dsim):
        seen = []
        ctrl = SimulationController(dsim)
        ctrl.status_changed.connect(seen.append)
        ctrl._report_stiffness()
        return seen

    def test_suspected_run_suggests_an_implicit_solver(self, qapp):
        seen = self._statuses(_dsim_with(SUSPECTED))
        assert len(seen) == 1
        assert "LSODA" in seen[0] and "RK45" in seen[0]

    def test_clean_run_says_nothing(self, qapp):
        assert self._statuses(_dsim_with(NOT_SUSPECTED)) == []

    def test_no_diagnostics_says_nothing(self, qapp):
        assert self._statuses(_dsim_with({})) == []

    def test_interpreter_run_says_nothing(self, qapp):
        # No engine at all (or one without the accessor): nothing to report.
        dsim = types.SimpleNamespace(engine=None)
        seen = []
        ctrl = SimulationController(dsim)
        ctrl.status_changed.connect(seen.append)
        ctrl._report_stiffness()
        assert seen == []

    def test_a_raising_engine_is_swallowed(self, qapp):
        def _boom():
            raise RuntimeError("engine gone")

        dsim = types.SimpleNamespace(engine=types.SimpleNamespace(get_solver_diagnostics=_boom))
        ctrl = SimulationController(dsim)
        seen = []
        ctrl.status_changed.connect(seen.append)
        ctrl._report_stiffness()  # must not raise
        assert seen == []


@pytest.mark.qt
class TestPropertyEditorStiffnessRows:
    def _editor(self, dsim):
        editor = PropertyEditor()
        editor._dsim = dsim
        return editor

    def test_suspected_run_names_the_suggested_method(self, qapp):
        editor = self._editor(_dsim_with(SUSPECTED))
        try:
            rows = dict(editor._last_run_rows())
            assert len(rows) == 2
            assert "LSODA" in " ".join(rows.values())
        finally:
            editor.deleteLater()

    def test_clean_run_still_reports_the_verdict(self, qapp):
        editor = self._editor(_dsim_with(NOT_SUSPECTED))
        try:
            values = [v for _k, v in editor._last_run_rows()]
            assert values, "a completed probe should still say 'not detected'"
            assert "LSODA" not in values[0]
        finally:
            editor.deleteLater()

    def test_no_diagnostics_hides_the_rows(self, qapp):
        editor = self._editor(_dsim_with({}))
        try:
            assert editor._last_run_rows() == []
        finally:
            editor.deleteLater()

    def test_missing_engine_hides_the_rows(self, qapp):
        editor = self._editor(types.SimpleNamespace(engine=None))
        try:
            assert editor._last_run_rows() == []
        finally:
            editor.deleteLater()

    def test_interpreter_fallback_names_the_offending_block(self, qapp):
        editor = self._editor(
            _dsim_with({}, fallback_reason="zoh3 (ZeroOrderHold): it has a discrete sample time")
        )
        try:
            values = [v for _k, v in editor._last_run_rows()]
            assert values == ["zoh3 (ZeroOrderHold): it has a discrete sample time"]
        finally:
            editor.deleteLater()

    def test_auto_method_is_shown_with_what_it_runs(self, qapp):
        """The inspector must not leave 'auto' unexplained."""
        editor = self._editor(_dsim_with({}, solver_method="auto"))
        try:
            editor.set_diagram_context(editor._dsim, None)
            texts = [lbl.text() for lbl in editor.findChildren(QLabel)]
            assert any("auto" in t and "LSODA" in t for t in texts)
        finally:
            editor.deleteLater()
