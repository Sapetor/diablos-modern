"""
Tests for ScopePlotter's previous-run stash (lib/plotting/scope_plotter.py).

Two successive pyqtPlotScope calls with fresh timelines must thread the first
run's flattened scope data into the second SignalPlot as ``previous_run``,
while re-plotting the same run ("Plot Again" / the interpreter double-plot)
must NOT rotate the stash. Uses a stub dsim (no engine) so pyqtPlotScope runs
against plain block objects.

Run offscreen:
    $env:QT_QPA_PLATFORM="offscreen"
"""

import numpy as np
import pytest

from lib.plotting.scope_plotter import ScopePlotter


class _StubScopeBlock:
    def __init__(self, name, labels, vector):
        self.block_fn = "Scope"
        self.name = name
        self.params = {"vec_labels": labels, "vector": vector, "vec_dim": 1}
        self.exec_params = {"vec_labels": labels, "vector": vector, "vec_dim": 1}


class _StubDSim:
    """Minimal dsim: no ``engine`` attribute, so blocks_list is used."""

    def __init__(self, n_samples=11):
        self.blocks_list = []
        self.timeline = np.linspace(0.0, 1.0, n_samples)
        self.sim_dt = 0.1
        self.plotty = None

    def get_neighbors(self, name):
        return [], []

    def new_run(self, vector):
        """Simulate a fresh run: new timeline object + new scope data."""
        self.timeline = np.linspace(0.0, 1.0, len(self.timeline))
        for block in self.blocks_list:
            block.exec_params["vector"] = vector


@pytest.mark.unit
class TestScopePlotterPrevRun:
    def test_two_runs_thread_previous_through(self, qapp):
        dsim = _StubDSim()
        run1 = np.arange(11.0)
        dsim.blocks_list.append(_StubScopeBlock("Scope0", "sig", run1))
        plotter = ScopePlotter(dsim)

        # First run ever: no held data
        plotter.pyqtPlotScope()
        assert plotter.plotty is not None
        assert plotter.plotty.previous_run is None
        assert not plotter.plotty.prev_run_checkbox.isEnabled()

        # Second run (new timeline object): previous run threads through
        run2 = np.arange(11.0) * 2
        dsim.new_run(run2)
        plotter.pyqtPlotScope()
        prev = plotter.plotty.previous_run
        assert prev is not None
        assert prev["labels"] == ["sig"]
        np.testing.assert_allclose(prev["vectors"][0], run1)
        assert prev["step_modes"] == [False]
        assert plotter.plotty.prev_run_checkbox.isEnabled()
        assert plotter.plotty.prev_curves[0] is not None

        # The stash must pin the timeline object it identified the run by, so
        # its id() can never be recycled for a later run's timeline.
        assert plotter._current_run["timeline_obj"] is dsim.timeline

        plotter._close_plotty()

    def test_replot_same_run_does_not_rotate(self, qapp):
        dsim = _StubDSim()
        run1 = np.arange(11.0)
        dsim.blocks_list.append(_StubScopeBlock("Scope0", "sig", run1))
        plotter = ScopePlotter(dsim)

        plotter.pyqtPlotScope()
        # "Plot Again" / interpreter double-plot: same timeline object
        plotter.pyqtPlotScope()
        assert plotter.plotty.previous_run is None

        # A real second run still sees run1 (not the replot) as previous
        run2 = np.arange(11.0) * 2
        dsim.new_run(run2)
        plotter.pyqtPlotScope()
        np.testing.assert_allclose(plotter.plotty.previous_run["vectors"][0], run1)
        # And replotting run 2 keeps run1 held
        plotter.pyqtPlotScope()
        np.testing.assert_allclose(plotter.plotty.previous_run["vectors"][0], run1)

        plotter._close_plotty()

    def test_stash_run_deep_copies(self, qapp):
        dsim = _StubDSim()
        plotter = ScopePlotter(dsim)
        t = dsim.timeline.astype(float)
        vec = np.arange(11.0)

        plotter._stash_run(t, ["sig"], [vec], [False])
        vec[:] = -1.0  # engine reuses/clears vectors between runs

        dsim.timeline = np.linspace(0.0, 1.0, 11)  # new run identity
        prev = plotter._stash_run(dsim.timeline.astype(float), ["sig"], [vec], [False])
        np.testing.assert_allclose(prev["vectors"][0], np.arange(11.0))

    def test_multi_signal_scope_stashes_split_columns(self, qapp):
        """The vec_dim>1 interleaved-reshape branch must stash per-signal data."""
        dsim = _StubDSim()
        sig_a = np.arange(11.0)
        sig_b = np.arange(11.0) * 2
        interleaved = np.column_stack([sig_a, sig_b]).ravel()
        block = _StubScopeBlock("Scope0", "a, b", interleaved)
        block.params["vec_dim"] = 2
        block.exec_params["vec_dim"] = 2
        dsim.blocks_list.append(block)
        plotter = ScopePlotter(dsim)

        plotter.pyqtPlotScope()
        dsim.new_run(np.column_stack([sig_a * 5, sig_b * 5]).ravel())
        plotter.pyqtPlotScope()

        prev = plotter.plotty.previous_run
        assert prev["labels"] == ["a", "b"]
        np.testing.assert_allclose(prev["vectors"][0], sig_a)
        np.testing.assert_allclose(prev["vectors"][1], sig_b)
        assert prev["step_modes"] == [False, False]

        plotter._close_plotty()

    def test_reset_held_runs_clears_stash(self, qapp):
        """After the diagram is replaced, the next run must have no previous run."""
        dsim = _StubDSim()
        dsim.blocks_list.append(_StubScopeBlock("Scope0", "sig", np.arange(11.0)))
        plotter = ScopePlotter(dsim)

        plotter.pyqtPlotScope()
        dsim.new_run(np.arange(11.0) * 2)
        plotter.pyqtPlotScope()
        assert plotter.plotty.previous_run is not None

        # Diagram replaced (File > New / Open): the stash belongs to the old
        # diagram and must be dropped.
        plotter.reset_held_runs()
        assert plotter._prev_run is None
        assert plotter._current_run is None

        dsim.new_run(np.arange(11.0) * 3)
        plotter.pyqtPlotScope()
        assert plotter.plotty.previous_run is None
        assert not plotter.plotty.prev_run_checkbox.isEnabled()

        plotter._close_plotty()

    def test_dsim_clear_all_resets_held_runs(self, qapp):
        """DSim.clear_all (File > New) must drop the held-run stash."""
        from lib.lib import DSim

        dsim = DSim()
        dsim.scope_plotter._prev_run = {"sig": 1}
        dsim.scope_plotter._current_run = {"sig": 2}

        dsim.clear_all()

        assert dsim.scope_plotter._prev_run is None
        assert dsim.scope_plotter._current_run is None

    def test_dsim_deserialize_resets_held_runs(self, qapp):
        """Loading a diagram (File > Open) must drop the held-run stash."""
        from lib.lib import DSim

        dsim = DSim()
        dsim.scope_plotter._prev_run = {"sig": 1}
        dsim.scope_plotter._current_run = {"sig": 2}

        dsim.deserialize({"sim_data": {}, "blocks_data": [], "lines_data": []})

        assert dsim.scope_plotter._prev_run is None
        assert dsim.scope_plotter._current_run is None
