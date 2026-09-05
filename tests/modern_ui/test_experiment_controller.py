"""Smoke + regression tests for :class:`ExperimentController`.

Regression: the Monte-Carlo / parameter-sweep workers used to be handed the
*live* ``window.dsim``. Both runners drive full simulations (rewriting
blocks_list / line_list / timeline / execution_initialized) from a worker
thread while the window's 60 FPS tick reads the same objects to paint, so the
diagram now goes to the worker as an isolated copy. The tuning panel's 50 ms
debounce -- which writes block params and re-simulates on the GUI thread -- is
also disarmed before the worker starts.

The dialogs and workers are stubbed; no experiment is actually run.
"""

import types

import pytest
from PyQt5.QtCore import QPoint


@pytest.fixture(scope="module")
def window(qapp):
    from modern_ui.main_window import ModernDiaBloSWindow

    w = ModernDiaBloSWindow()
    yield w
    w.dsim.dirty = False
    w.close()


@pytest.fixture
def diagram(window):
    """Give the window a two-block diagram, removed again afterwards."""
    dsim = window.dsim
    menu = {b.fn_name: b for b in dsim.menu_blocks}
    step = dsim.add_block(menu["step"], QPoint(100, 100))
    scope = dsim.add_block(menu["scope"], QPoint(300, 100))
    dsim.add_line((step.name, 0, step.out_coords[0]), (scope.name, 0, scope.in_coords[0]))
    yield dsim
    dsim.clear_all()
    dsim.dirty = False


class _FakeWorker:
    """Stands in for MonteCarloWorker / ParameterSweepWorker."""

    instances = []

    def __init__(self, dsim, selection, parent=None):
        self.dsim = dsim
        self.selection = selection
        self.started = False
        _FakeWorker.instances.append(self)

    # Signal-shaped attributes the controller connects to.
    class _Sig:
        def connect(self, *_a, **_k):
            return None

    progress = _Sig()
    finished = _Sig()
    failed = _Sig()

    def start(self):
        self.started = True

    def isRunning(self):
        return False

    def cancel(self):
        return None

    def wait(self, _ms=None):
        return True

    def deleteLater(self):
        return None


@pytest.fixture
def fake_workers(monkeypatch):
    _FakeWorker.instances = []
    import modern_ui.widgets.monte_carlo_worker as mc_mod
    import modern_ui.widgets.parameter_sweep_worker as sw_mod
    import modern_ui.widgets.monte_carlo_dialog as mc_dlg
    import modern_ui.widgets.parameter_sweep_dialog as sw_dlg

    monkeypatch.setattr(mc_mod, "MonteCarloWorker", _FakeWorker)
    monkeypatch.setattr(sw_mod, "ParameterSweepWorker", _FakeWorker)

    class _Dlg:
        def __init__(self, dsim, parent=None):
            self.dsim = dsim

        def exec_(self):
            from PyQt5.QtWidgets import QDialog

            return QDialog.Accepted

        def get_selection(self):
            return self.selection

    class _MCDialog(_Dlg):
        selection = {"n_runs": 3, "master_seed": 7, "sim_time": 0.1, "sim_dt": 0.01}

    class _SweepDialog(_Dlg):
        selection = {
            "axes": [{"block": "step0", "param": "value", "values": [1.0, 2.0]}],
            "sim_time": 0.1,
            "sim_dt": 0.01,
        }

    monkeypatch.setattr(mc_dlg, "MonteCarloDialog", _MCDialog)
    monkeypatch.setattr(sw_dlg, "ParameterSweepDialog", _SweepDialog)
    monkeypatch.setattr(sw_dlg, "sweepable_blocks", lambda dsim: [object()])
    return _FakeWorker


@pytest.mark.qt
class TestIsolatedDiagram:
    def test_isolated_dsim_is_a_copy(self, window, diagram):
        clone = window.experiment_controller._isolated_dsim("Monte Carlo")
        assert clone is not None
        assert clone is not window.dsim
        assert clone.blocks_list is not window.dsim.blocks_list
        assert [b.name for b in clone.blocks_list] == [b.name for b in window.dsim.blocks_list]

    def test_isolated_dsim_reports_failure_instead_of_falling_back(self, window, monkeypatch):
        monkeypatch.setattr(
            window.dsim,
            "clone_for_analysis",
            lambda: (_ for _ in ()).throw(RuntimeError("nope")),
        )
        assert window.experiment_controller._isolated_dsim("Monte Carlo") is None

    def test_monte_carlo_worker_gets_the_copy_not_the_live_dsim(
        self, window, diagram, fake_workers
    ):
        window.tuning_controller.store_sim_params(1.0, 0.01)
        assert window.tuning_controller.is_active is True

        window.experiment_controller.run_monte_carlo()

        assert len(fake_workers.instances) == 1
        worker = fake_workers.instances[0]
        assert worker.started is True
        assert worker.dsim is not window.dsim
        assert [b.name for b in worker.dsim.blocks_list] == [
            b.name for b in window.dsim.blocks_list
        ]
        # The tuning debounce must not fire while an experiment runs.
        assert window.tuning_controller.is_active is False

        window._mc_worker = None

    def test_parameter_sweep_worker_gets_the_copy(self, window, diagram, fake_workers):
        window.tuning_controller.store_sim_params(1.0, 0.01)
        window.experiment_controller.run_parameter_sweep()

        assert len(fake_workers.instances) == 1
        worker = fake_workers.instances[0]
        assert worker.started is True
        assert worker.dsim is not window.dsim
        assert window.tuning_controller.is_active is False

        window._sweep_worker = None


@pytest.mark.qt
class TestGuards:
    def test_empty_diagram_does_not_start_a_worker(self, window, fake_workers):
        window.dsim.clear_all()
        window.experiment_controller.run_monte_carlo()
        window.experiment_controller.run_parameter_sweep()
        assert fake_workers.instances == []

    def test_second_run_is_refused_while_one_is_active(self, window, diagram, fake_workers):
        window._mc_worker = types.SimpleNamespace()
        try:
            window.experiment_controller.run_monte_carlo()
            assert fake_workers.instances == []
        finally:
            window._mc_worker = None
