"""Pressing Play used to pop the Simulation-settings modal on *every* run.

``DSim.execution_init`` unconditionally called ``execution_init_time()``, which
constructs a ``SimulationDialog`` and ``exec()``s it. The whole suite missed it
because every test that reached ``execution_init`` stubbed
``execution_init_time`` out, and there was no other way to reach the dialog --
so it doubled as the settings editor.

Now:

  * Play runs immediately with the settings stored in the diagram;
  * Simulation > Simulation Settings... (Ctrl+E, or the toolbar gear) opens the
    same dialog pre-filled, and applies on OK;
  * an opt-in "Ask before every run" preference (QSettings
    ``simulation/ask_before_run``, default off) brings the old flow back.

The tests spy on the ``SimulationDialog`` *class* -- construction is the
observable that matters, since a constructed-but-unshown dialog would still be
the bug on a real display.
"""

import pytest
from PyQt6.QtCore import QPoint
from PyQt6.QtWidgets import QDialog

import lib.lib as lib_lib
from lib.dialogs import SimulationDialog

pytestmark = [pytest.mark.regression, pytest.mark.qt]


# ---------------------------------------------------------------------------
# Spies
# ---------------------------------------------------------------------------


def _spy(monkeypatch, result=QDialog.DialogCode.Rejected, mutate=None):
    """Replace ``lib.lib.SimulationDialog`` with a recording subclass.

    Returns the list its instances are appended to. ``mutate`` (if given) is
    called with the dialog just before ``exec`` returns ``result``, so a test
    can simulate the user editing a field.
    """
    seen = []

    class _Spy(SimulationDialog):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            seen.append(self)

        def exec(self):
            if mutate is not None:
                mutate(self)
            return result

    monkeypatch.setattr(lib_lib, "SimulationDialog", _Spy)
    return seen


def _build_diagram(dsim):
    """A minimal runnable diagram: Step -> Scope."""
    menu = {b.fn_name: b for b in dsim.menu_blocks}
    step = dsim.add_block(menu["step"], QPoint(100, 100))
    scope = dsim.add_block(menu["scope"], QPoint(400, 100))
    dsim.add_line((step.name, 0, step.out_coords[0]), (scope.name, 0, scope.in_coords[0]))
    dsim.sim_time, dsim.sim_dt = 0.05, 0.01
    return dsim


@pytest.fixture
def dsim(qapp, tmp_path, monkeypatch):
    from lib.lib import DSim

    monkeypatch.chdir(tmp_path)
    sim = DSim()
    monkeypatch.setattr(sim, "save", lambda *a, **k: 0)
    _build_diagram(sim)
    yield sim
    sim.execution_initialized = False


# ---------------------------------------------------------------------------
# The preference itself
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAskBeforeRunPreference:
    def test_defaults_to_off(self, monkeypatch):
        from PyQt6.QtCore import QSettings

        import lib.sim_prefs as sim_prefs

        store = QSettings("DiaBloS-test", "sim-prefs-default")
        store.clear()
        monkeypatch.setattr(sim_prefs, "ui_settings", lambda: store)
        assert sim_prefs.ask_before_run() is False

    @pytest.mark.parametrize("flag", [True, False])
    def test_round_trips_through_qsettings(self, monkeypatch, flag):
        from PyQt6.QtCore import QSettings

        import lib.sim_prefs as sim_prefs

        store = QSettings("DiaBloS-test", "sim-prefs-roundtrip")
        store.clear()
        monkeypatch.setattr(sim_prefs, "ui_settings", lambda: store)
        sim_prefs.set_ask_before_run(flag)
        assert sim_prefs.ask_before_run() is flag

    def test_a_stored_false_reads_back_as_false(self, monkeypatch):
        """The INI backend hands booleans back as the string "false"."""
        import lib.sim_prefs as sim_prefs

        monkeypatch.setattr(
            sim_prefs, "ui_settings", lambda: type("S", (), {"value": lambda s, k, d: "false"})()
        )
        assert sim_prefs.ask_before_run() is False


# ---------------------------------------------------------------------------
# DSim.execution_init
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExecutionInitDoesNotAsk:
    def test_no_dialog_by_default(self, dsim, monkeypatch):
        seen = _spy(monkeypatch)
        dsim.ask_before_run = False
        assert dsim.execution_init() is True
        assert seen == []

    def test_uses_the_stored_duration(self, dsim, monkeypatch):
        _spy(monkeypatch)
        dsim.ask_before_run = False
        dsim.sim_time = 0.07
        assert dsim.execution_init() is True
        assert dsim.execution_time == 0.07

    def test_dialog_when_the_preference_is_on(self, dsim, monkeypatch):
        seen = _spy(monkeypatch, result=QDialog.DialogCode.Accepted)
        dsim.ask_before_run = True
        assert dsim.execution_init() is True
        assert len(seen) == 1

    def test_cancelling_the_ask_path_aborts_the_run(self, dsim, monkeypatch):
        _spy(monkeypatch, result=QDialog.DialogCode.Rejected)
        dsim.ask_before_run = True
        assert dsim.execution_init() is False
        assert dsim.execution_initialized is False

    def test_explicit_ask_false_beats_the_preference(self, dsim, monkeypatch):
        """The headless/analysis callers must never be able to pop a dialog."""
        seen = _spy(monkeypatch, result=QDialog.DialogCode.Accepted)
        dsim.ask_before_run = True
        assert dsim.execution_init(ask=False) is True
        assert seen == []

    def test_explicit_ask_true_forces_the_dialog(self, dsim, monkeypatch):
        seen = _spy(monkeypatch, result=QDialog.DialogCode.Accepted)
        dsim.ask_before_run = False
        assert dsim.execution_init(ask=True) is True
        assert len(seen) == 1


@pytest.mark.unit
class TestExecutionInitTimeStillWorks:
    """It is the *ask* path now, but its contract is unchanged."""

    def test_returns_the_duration_on_accept(self, dsim, monkeypatch):
        _spy(
            monkeypatch,
            result=QDialog.DialogCode.Accepted,
            mutate=lambda d: d.sim_time_input.setText("3.25"),
        )
        assert dsim.execution_init_time() == 3.25
        assert dsim.sim_time == 3.25

    def test_returns_minus_one_on_cancel(self, dsim, monkeypatch):
        _spy(monkeypatch, result=QDialog.DialogCode.Rejected)
        assert dsim.execution_init_time() == -1


# ---------------------------------------------------------------------------
# apply_sim_settings / dirty flag
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestApplySimSettings:
    def test_a_persisted_change_reports_dirty(self, dsim):
        dsim.dirty = False
        assert dsim.apply_sim_settings({"sim_time": dsim.sim_time + 1.0}) is True

    def test_a_session_only_change_does_not(self, dsim):
        """real_time / dynamic_plot are run modes, not diagram data."""
        assert dsim.apply_sim_settings({"real_time": not dsim.real_time}) is False
        assert dsim.apply_sim_settings({"dynamic_plot": not dsim.dynamic_plot}) is False

    def test_unchanged_values_report_no_change(self, dsim):
        values = {key: getattr(dsim, key) for key in dsim._SIM_SETTINGS}
        assert dsim.apply_sim_settings(values) is False

    def test_every_solver_field_round_trips(self, dsim):
        dsim.apply_sim_settings(
            {
                "sim_time": 12.0,
                "sim_dt": 0.002,
                "plot_trange": 250,
                "solver_method": "LSODA",
                "rtol": 1e-6,
                "atol": 1e-8,
                "zero_crossing": False,
                "real_time": True,
                "dynamic_plot": True,
            }
        )
        assert (dsim.sim_time, dsim.sim_dt, dsim.plot_trange) == (12.0, 0.002, 250)
        assert dsim.solver_method == "LSODA"
        assert (dsim.rtol, dsim.atol) == (1e-6, 1e-8)
        assert dsim.zero_crossing is False
        assert dsim.real_time is True and dsim.dynamic_plot is True

    def test_the_ask_flag_is_persisted_not_dirtying(self, dsim, monkeypatch):
        written = {}
        monkeypatch.setattr(lib_lib, "set_ask_before_run", lambda v: written.__setitem__("v", v))
        dsim.ask_before_run = False
        assert dsim.apply_sim_settings({"ask_before_run": True}) is False
        assert dsim.ask_before_run is True
        assert written == {"v": True}


# ---------------------------------------------------------------------------
# The window: Play, the menu action and the toolbar gear
# ---------------------------------------------------------------------------


@pytest.fixture
def armed_window(window, tmp_path, monkeypatch):
    """The real main window with a runnable Step -> Scope diagram.

    The batch run itself is stubbed out: these tests are about what happens
    *before* the solver, and a worker thread would outlive the test.
    """
    from modern_ui.controllers.simulation_controller import SimulationController

    monkeypatch.chdir(tmp_path)
    dsim = window.dsim
    dsim.clear_all()
    _build_diagram(dsim)
    monkeypatch.setattr(dsim, "save", lambda *a, **k: 0)
    monkeypatch.setattr(window, "_auto_save", lambda *a, **k: None)
    monkeypatch.setattr(SimulationController, "run_batch", lambda self: None)
    monkeypatch.setattr(dsim, "ask_before_run", False, raising=False)
    yield window
    dsim.execution_initialized = False
    dsim.clear_all()


class TestPlayFromTheWindow:
    def test_play_does_not_open_the_dialog(self, armed_window, monkeypatch):
        seen = _spy(monkeypatch)
        armed_window.start_simulation()
        assert seen == [], "pressing Play must not pop the settings modal"
        assert armed_window.dsim.execution_initialized is True

    def test_play_opens_the_dialog_when_the_preference_is_on(self, armed_window, monkeypatch):
        seen = _spy(monkeypatch, result=QDialog.DialogCode.Accepted)
        armed_window.dsim.ask_before_run = True
        armed_window.start_simulation()
        assert len(seen) == 1


class TestSimulationSettingsAction:
    def test_the_menu_action_exists(self, armed_window):
        action = getattr(armed_window, "simulation_settings_action", None)
        assert action is not None
        assert action.shortcut().toString() == "Ctrl+E"

    def test_the_toolbar_has_a_gear_next_to_the_transport(self, armed_window):
        assert armed_window.toolbar.transport.settings_btn is not None

    def test_the_toolbar_still_fits(self, armed_window):
        """The gear must not push the trailing tools into the overflow menu."""
        assert armed_window.toolbar.sizeHint().width() <= armed_window.minimumWidth()

    def test_the_dialog_opens_pre_filled_with_the_live_values(self, armed_window, monkeypatch):
        dsim = armed_window.dsim
        dsim.apply_sim_settings(
            {"sim_time": 9.0, "sim_dt": 0.004, "solver_method": "BDF", "zero_crossing": False}
        )
        seen = _spy(monkeypatch)
        armed_window.open_simulation_settings()

        assert len(seen) == 1
        dlg = seen[0]
        assert float(dlg.sim_time_input.text()) == 9.0
        assert float(dlg.sampling_time_input.text()) == 0.004
        assert dlg.solver_method_combo.currentText() == "BDF"
        assert dlg.zero_crossing_checkbox.isChecked() is False
        assert dlg.ask_before_run_checkbox.isChecked() is dsim.ask_before_run

    def test_accepting_applies_and_marks_the_diagram_dirty(self, armed_window, monkeypatch):
        dsim = armed_window.dsim
        dsim.dirty = False
        _spy(
            monkeypatch,
            result=QDialog.DialogCode.Accepted,
            mutate=lambda d: d.sim_time_input.setText("42.0"),
        )
        assert armed_window.open_simulation_settings() is True
        assert dsim.sim_time == 42.0
        assert dsim.dirty is True

    def test_accepting_an_unchanged_dialog_leaves_the_flag_alone(self, armed_window, monkeypatch):
        dsim = armed_window.dsim
        dsim.dirty = False
        _spy(monkeypatch, result=QDialog.DialogCode.Accepted)
        assert armed_window.open_simulation_settings() is True
        assert dsim.dirty is False

    def test_cancelling_changes_nothing(self, armed_window, monkeypatch):
        dsim = armed_window.dsim
        dsim.dirty = False
        before = dsim.sim_time
        _spy(
            monkeypatch,
            result=QDialog.DialogCode.Rejected,
            mutate=lambda d: d.sim_time_input.setText("999.0"),
        )
        assert armed_window.open_simulation_settings() is False
        assert dsim.sim_time == before
        assert dsim.dirty is False

    def test_turning_the_preference_on_through_the_dialog_arms_the_ask_path(
        self, armed_window, monkeypatch
    ):
        dsim = armed_window.dsim
        written = {}
        monkeypatch.setattr(lib_lib, "set_ask_before_run", lambda v: written.__setitem__("v", v))
        _spy(
            monkeypatch,
            result=QDialog.DialogCode.Accepted,
            mutate=lambda d: d.ask_before_run_checkbox.setChecked(True),
        )
        armed_window.open_simulation_settings()
        assert dsim.ask_before_run is True
        assert written == {"v": True}

        seen = _spy(monkeypatch, result=QDialog.DialogCode.Accepted)
        armed_window.start_simulation()
        assert len(seen) == 1
