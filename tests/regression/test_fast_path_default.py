"""Play must reach the compiled fast solver, and the settings must round-trip.

``real_time`` defaulted to True, and ``SimulationController.start()`` sends a
real-time run down the interpreted step loop, which is paced to the wall clock --
a 200 s diagram then takes 200 s however trivial it is.  The default was masked
because the Simulation-settings dialog never initialised its "Run in real-time" /
"Enable Dynamic Plotting" checkboxes from the live values, so it opened them
unchecked and accepting it silently forced both to False.  Now that the boxes
round-trip, the default decides the routing, so it has to be off.
"""

import pytest

pytestmark = pytest.mark.regression


@pytest.mark.unit
@pytest.mark.qt
class TestFastPathIsTheDefault:
    def test_engine_does_not_default_to_real_time(self, simulation_engine):
        """real_time=True routes Play to the paced interpreter, not the solver."""
        assert simulation_engine.real_time is False

    def test_dsim_does_not_default_to_real_time(self):
        from lib.lib import DSim

        assert DSim().real_time is False


@pytest.mark.qt
class TestSimulationDialogRoundTrips:
    """Accepting the dialog must return the values it was opened with."""

    @pytest.mark.parametrize("flag", [True, False])
    def test_real_time_round_trips(self, qapp, flag):
        from lib.dialogs import SimulationDialog

        dlg = SimulationDialog(10.0, 0.01, 100, real_time=flag)
        assert dlg.get_values()["real_time"] is flag

    @pytest.mark.parametrize("flag", [True, False])
    def test_dynamic_plot_round_trips(self, qapp, flag):
        from lib.dialogs import SimulationDialog

        dlg = SimulationDialog(10.0, 0.01, 100, dynamic_plot=flag)
        assert dlg.get_values()["dynamic_plot"] is flag
