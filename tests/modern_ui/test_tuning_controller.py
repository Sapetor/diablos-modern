"""Smoke tests for :class:`TuningController`.

The controller had zero coverage while owning a 50 ms debounce that writes
block params and re-simulates on the GUI thread -- the same debounce that had
to be disarmed before an experiment worker starts (see
``tests/modern_ui/test_experiment_controller.py``).

The re-simulation itself is stubbed; nothing here runs the engine.
"""

import numpy as np
import pytest


class _StubPlotter:
    plotty = None

    def pyqtPlotScope(self):
        self.plotted = True


class _StubDSim:
    def __init__(self):
        self.blocks_list = []
        self.timeline = np.linspace(0.0, 1.0, 3)
        self.calls = []
        self.result = (True, "")

    def run_tuning_simulation(self, sim_time, sim_dt, **kwargs):
        self.calls.append((sim_time, sim_dt))
        return self.result


def _block(name, params):
    class _B:
        pass

    b = _B()
    b.name = name
    b.block_fn = "Gain"
    b.params = dict(params)
    return b


@pytest.fixture
def controller(qapp):
    from modern_ui.controllers.tuning_controller import TuningController

    dsim = _StubDSim()
    ctrl = TuningController(dsim, _StubPlotter())
    ctrl._update_plots = lambda: None  # no Qt plot windows in tests
    return ctrl


@pytest.mark.qt
class TestArming:
    def test_starts_inactive(self, controller):
        assert controller.is_active is False

    def test_store_sim_params_arms_it(self, controller):
        controller.store_sim_params(2.0, 0.01)
        assert controller.is_active is True

    def test_deactivate_disarms_and_stops_the_debounce(self, controller):
        controller.store_sim_params(2.0, 0.01)
        controller.dsim.blocks_list = [_block("gain0", {"gain": 1.0})]
        controller.on_param_changed("gain0", "gain", 3.0)
        assert controller._debounce.isActive() is True

        controller.deactivate()
        assert controller.is_active is False
        assert controller._debounce.isActive() is False
        assert controller._pending_changes == {}

    def test_param_change_while_disarmed_is_reported_not_applied(self, controller):
        messages = []
        controller.set_status_callback(messages.append)
        controller.dsim.blocks_list = [_block("gain0", {"gain": 1.0})]

        controller.on_param_changed("gain0", "gain", 3.0)

        assert controller._pending_changes == {}
        assert controller.dsim.blocks_list[0].params["gain"] == 1.0
        assert any("Run simulation first" in m for m in messages)


@pytest.mark.qt
class TestExecuteTuning:
    def test_applies_pending_changes_and_resimulates(self, controller):
        block = _block("gain0", {"gain": 1.0})
        controller.dsim.blocks_list = [block]
        controller.store_sim_params(2.0, 0.05)

        controller.on_param_changed("gain0", "gain", 4.0)
        controller._execute_tuning()

        assert block.params["gain"] == 4.0
        assert controller.dsim.calls == [(2.0, 0.05)]
        assert controller._pending_changes == {}

    def test_indexed_list_param_updates_one_element(self, controller):
        block = _block("tf0", {"den": [1.0, 2.0, 3.0]})
        controller.dsim.blocks_list = [block]
        controller.store_sim_params(1.0, 0.01)

        controller.on_param_changed("tf0", "den[1]", 9.0)
        controller._execute_tuning()

        assert list(block.params["den"]) == [1.0, 9.0, 3.0]

    def test_non_indexable_param_is_skipped_with_a_message(self, controller):
        messages = []
        controller.set_status_callback(messages.append)
        block = _block("gain0", {"gain": 1.0})
        controller.dsim.blocks_list = [block]
        controller.store_sim_params(1.0, 0.01)

        controller.on_param_changed("gain0", "gain[2]", 5.0)
        controller._execute_tuning()

        assert block.params["gain"] == 1.0
        assert any("skipped" in m for m in messages)

    def test_failed_resim_is_reported(self, controller):
        messages = []
        controller.set_status_callback(messages.append)
        controller.dsim.result = (False, "diverged")
        controller.dsim.blocks_list = [_block("gain0", {"gain": 1.0})]
        controller.store_sim_params(1.0, 0.01)

        controller.on_param_changed("gain0", "gain", 2.0)
        controller._execute_tuning()

        assert any("diverged" in m for m in messages)

    def test_no_pending_changes_is_a_noop(self, controller):
        controller.store_sim_params(1.0, 0.01)
        controller._execute_tuning()
        assert controller.dsim.calls == []
