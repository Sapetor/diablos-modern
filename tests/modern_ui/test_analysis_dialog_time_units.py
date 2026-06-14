"""Tests that the analysis dialogs label their seconds-valued fields with units.

The Monte-Carlo and parameter-sweep run dialogs expose a simulation-time and a
step-size (dt) spin box, both measured in seconds. Each carries a " s" suffix so
the unit is visible inline. The non-time fields (run count, master seed, sweep
ranges) must stay bare. The dialogs are constructed but never exec_()'d.
"""

import pytest


@pytest.fixture(autouse=True)
def _qt(qapp):
    """Use the session QApplication; never build our own (keeps theme_manager)."""
    return qapp


class _StubDSim:
    """Minimal stand-in for DSim: only the attributes the dialogs read."""

    def __init__(self):
        self.sim_time = 7.5
        self.sim_dt = 0.02
        self.blocks_list = []  # no sweepable blocks needed for a suffix check


@pytest.mark.unit
class TestAnalysisDialogTimeUnits:
    def test_monte_carlo_time_fields_have_seconds_suffix(self):
        from modern_ui.widgets.monte_carlo_dialog import MonteCarloDialog

        dialog = MonteCarloDialog(_StubDSim())

        # Seconds-valued fields are labelled with a unit.
        assert dialog.sim_time_spin.suffix().strip() != ""
        assert dialog.sim_dt_spin.suffix().strip() != ""
        assert dialog.sim_time_spin.suffix() == " s"
        assert dialog.sim_dt_spin.suffix() == " s"

        # Non-time fields stay bare.
        assert dialog.n_runs_spin.suffix() == ""
        assert dialog.master_seed_spin.suffix() == ""

    def test_parameter_sweep_time_fields_have_seconds_suffix(self):
        from modern_ui.widgets.parameter_sweep_dialog import ParameterSweepDialog

        dialog = ParameterSweepDialog(_StubDSim())

        # Seconds-valued fields are labelled with a unit.
        assert dialog.sim_time_spin.suffix().strip() != ""
        assert dialog.sim_dt_spin.suffix().strip() != ""
        assert dialog.sim_time_spin.suffix() == " s"
        assert dialog.sim_dt_spin.suffix() == " s"
