"""Unit tests for the phases of ``DSim._interpreter_step`` (lib/lib.py).

The step is a fixed sequence over small methods: ``_advance_clock``,
``_publish_memory_outputs``, ``_run_hierarchy_passes`` /
``_execute_ready_block``, ``_is_end_of_run`` and ``_finish_run``. The
end-to-end numerics are pinned by the regression suites (RK45 interpreter,
simulation horizon, feedthrough memory, discrete sampling); these tests cover
each phase in isolation against a mocked engine.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from lib.lib import DSim


def _dsim(rk45=False, sim_dt=0.1, execution_time=1.0, blocks=()):
    """A DSim shell with a mocked engine: no GUI, no real blocks."""
    d = DSim.__new__(DSim)
    d.engine = MagicMock()
    d.engine.time_step = 0.0
    d.engine.rk_counter = 0
    d.engine.rk45_len = rk45
    d.engine.memory_blocks = set()
    d.engine.max_hier = 0
    d.engine.execution_initialized = True
    d.engine.active_blocks_list = list(blocks)
    d.engine.execute_block.return_value = {0: np.array([1.0])}
    d.blocks_list = list(blocks)
    d.sim_dt = sim_dt
    d.execution_time = execution_time
    d.execution_pause = False
    d._timeline_list = []
    d._defer_gui_plots = False
    d.dynamic_plot = False
    d.pbar = MagicMock()
    d.export_data = MagicMock()
    d._record_run_history = MagicMock()
    d.pyqtPlotScope = MagicMock()
    d.dynamic_pyqtPlotScope = MagicMock()
    return d


def _block(name, hierarchy=0, in_ports=1, out_ports=1, b_type=0, **kw):
    b = SimpleNamespace(
        name=name,
        hierarchy=hierarchy,
        in_ports=in_ports,
        out_ports=out_ports,
        b_type=b_type,
        data_received=kw.pop("data_received", in_ports),
        computed_data=False,
        block_instance=None,
        effective_sample_time=kw.pop("effective_sample_time", 0),
        params={},
        exec_params={},
        should_execute=kw.pop("should_execute", lambda t: True),
        get_held_output=lambda p: f"held{p}",
        schedule_next_execution=MagicMock(),
    )
    for k, v in kw.items():
        setattr(b, k, v)
    return b


@pytest.mark.unit
class TestAdvanceClock:
    def test_euler_advances_one_step_and_records_sample(self):
        d = _dsim(sim_dt=0.1)
        assert d._advance_clock(interactive=False) is True
        assert np.isclose(d.time_step, 0.1)
        assert d._timeline_list == [d.time_step]
        d.pbar.update.assert_not_called()

    def test_rk4_cycle_advances_exactly_one_step(self):
        d = _dsim(rk45=True, sim_dt=0.1)
        increments, samples = [], []
        for counter in range(4):
            d.rk_counter = counter
            t0 = d.time_step
            samples.append(d._advance_clock(interactive=True))
            increments.append(d.time_step - t0)
        assert np.allclose(increments, [0.0, 0.05, 0.0, 0.05])
        assert samples == [True, False, False, False]
        assert len(d._timeline_list) == 1
        assert d.pbar.update.call_count == 1

    def test_rk_counter_wraps_modulo_four(self):
        d = _dsim(rk45=True)
        d.rk_counter = 4
        assert d._advance_clock(interactive=False) is True  # 4 % 4 == 0 -> sample
        assert d.rk_counter == 0


@pytest.mark.unit
class TestEndOfRun:
    def test_stops_after_last_sample_inside_horizon(self):
        d = _dsim(sim_dt=0.1, execution_time=1.0)
        d.time_step = 0.9
        assert not d._is_end_of_run(sample_recorded=True)
        d.time_step = 1.0
        assert d._is_end_of_run(sample_recorded=True)

    def test_sub_step_uses_runaway_guard_only(self):
        d = _dsim(sim_dt=0.1, execution_time=1.0)
        d.time_step = 1.05  # past the horizon, but no sample: keep going
        assert not d._is_end_of_run(sample_recorded=False)
        d.time_step = 1.2
        assert d._is_end_of_run(sample_recorded=False)


@pytest.mark.unit
class TestHasEnoughInputs:
    def test_counts_required_ports(self):
        assert not DSim._has_enough_inputs(_block("b", in_ports=2, data_received=1))
        assert DSim._has_enough_inputs(_block("b", in_ports=2, data_received=2))

    def test_optional_inputs_are_not_required(self):
        b = _block("b", in_ports=2, data_received=1)
        b.block_instance = SimpleNamespace(optional_inputs=[1])
        assert DSim._has_enough_inputs(b)

    def test_sources_always_ready(self):
        assert DSim._has_enough_inputs(_block("s", in_ports=0, data_received=0))


@pytest.mark.unit
class TestBlockFailed:
    def test_none_stops_run(self):
        d = _dsim()
        assert d._block_failed(None) is True
        assert d.execution_initialized is False
        assert d.error_msg == "Block returned None"
        d.engine.reset_memblocks.assert_called_once()

    def test_error_dict_reports_message(self):
        d = _dsim()
        assert d._block_failed({"E": True, "error": "boom"}) is True
        assert d.error_msg == "boom"

    def test_normal_output_passes(self):
        d = _dsim()
        assert d._block_failed({0: 1.0}) is False
        assert d._block_failed({"E": False, 0: 1.0}) is False
        assert d.execution_initialized is True


@pytest.mark.unit
class TestPublishMemoryOutputs:
    def test_non_memory_blocks_are_skipped(self):
        g = _block("g")
        d = _dsim(blocks=[g])
        assert d._publish_memory_outputs([g]) == {}
        d.engine.execute_block.assert_not_called()

    def test_memory_block_runs_output_only_and_propagates(self):
        m = _block("m", b_type=1)
        d = _dsim(blocks=[m])
        d.memory_blocks = {"m"}
        out = {0: np.array([2.0])}
        d.engine.execute_block.return_value = out
        assert d._publish_memory_outputs([m]) == {}  # continuous: nothing to hold
        d.engine.execute_block.assert_called_once_with(m, output_only=True)
        d.engine.propagate_outputs.assert_called_once_with(m, out)
        m.schedule_next_execution.assert_not_called()

    def test_discrete_memory_output_is_kept_for_holding(self):
        m = _block("m", b_type=1, effective_sample_time=0.5)
        d = _dsim(blocks=[m])
        d.memory_blocks = {"m"}
        out = {0: np.array([2.0])}
        d.engine.execute_block.return_value = out
        assert d._publish_memory_outputs([m]) == {"m": out}

    def test_off_sample_discrete_block_propagates_held_outputs(self):
        m = _block("m", b_type=1, out_ports=2, should_execute=lambda t: False)
        sink = _block("s", b_type=3, should_execute=lambda t: False)
        d = _dsim(blocks=[m, sink])
        d.memory_blocks = {"m", "s"}
        d._publish_memory_outputs([m, sink])
        d.engine.execute_block.assert_not_called()
        d.engine.propagate_outputs.assert_called_once_with(m, {0: "held0", 1: "held1"})

    def test_rk4_skip_flag_written_to_both_param_dicts(self):
        b = _block("b")
        d = _dsim(rk45=True, blocks=[b])
        d.rk_counter = 2
        d._publish_memory_outputs([b])
        assert b.params["_skip_"] is True and b.exec_params["_skip_"] is True
        d.rk_counter = 0
        d._publish_memory_outputs([b])
        assert b.params["_skip_"] is False and b.exec_params["_skip_"] is False

    def test_failure_returns_none(self):
        m = _block("m", b_type=1)
        d = _dsim(blocks=[m])
        d.memory_blocks = {"m"}
        d.engine.execute_block.return_value = {"E": True, "error": "bad"}
        assert d._publish_memory_outputs([m]) is None
        assert d.error_msg == "bad"

    def test_exception_is_caught_and_stops_run(self):
        m = _block("m", b_type=1)
        d = _dsim(blocks=[m])
        d.memory_blocks = {"m"}
        d.engine.execute_block.side_effect = RuntimeError("kaboom")
        assert d._publish_memory_outputs([m]) is None
        assert "kaboom" in d.error_msg
        assert d.execution_initialized is False


@pytest.mark.unit
class TestExecuteReadyBlock:
    def test_algebraic_block_executes_and_propagates(self):
        g = _block("g")
        d = _dsim(blocks=[g])
        out = {0: np.array([3.0])}
        d.engine.execute_block.return_value = out
        assert d._execute_ready_block(g, {}) is True
        d.engine.execute_block.assert_called_once_with(g)
        d.engine.update_global_list.assert_called_once_with("g", h_value=0)
        assert g.computed_data is True
        d.engine.propagate_outputs.assert_called_once_with(g, out)
        d.engine.sync_integrator_output.assert_not_called()

    def test_sink_does_not_propagate(self):
        s = _block("s", b_type=3)
        d = _dsim(blocks=[s])
        d._execute_ready_block(s, {})
        d.engine.propagate_outputs.assert_not_called()
        assert s.computed_data is True

    def test_strictly_proper_memory_block_syncs_and_stays_silent(self):
        m = _block("m", b_type=1)
        d = _dsim(blocks=[m])
        d.memory_blocks = {"m"}
        d._execute_ready_block(m, {})
        d.engine.sync_integrator_output.assert_called_once_with(m)
        d.engine.propagate_outputs.assert_not_called()

    def test_feedthrough_memory_block_refreshes_without_counting(self):
        m = _block("m", b_type=2)
        d = _dsim(blocks=[m])
        d.memory_blocks = {"m"}
        out = {0: np.array([1.0])}
        d.engine.execute_block.return_value = out
        d._execute_ready_block(m, {})
        d.engine.propagate_outputs.assert_called_once_with(m, out, count=False)

    def test_discrete_block_stamps_held_outputs_and_reschedules(self):
        z = _block("z", effective_sample_time=0.5)
        d = _dsim(blocks=[z])
        d.time_step = 0.5
        out = {0: np.array([1.0])}
        pre = {0: np.array([0.0])}
        d.engine.execute_block.return_value = out
        d._execute_ready_block(z, {"z": pre})
        d.engine.stamp_held_outputs.assert_called_once_with(z, out, pre)
        z.schedule_next_execution.assert_called_once_with(0.5)

    def test_off_sample_block_is_marked_computed_with_held_outputs(self):
        z = _block("z", should_execute=lambda t: False)
        d = _dsim(blocks=[z])
        assert d._execute_ready_block(z, {}) is True
        d.engine.execute_block.assert_not_called()
        assert z.computed_data is True
        d.engine.update_global_list.assert_called_once_with("z", h_value=0)
        d.engine.propagate_outputs.assert_called_once_with(z, {0: "held0"})

    def test_failure_returns_false_before_marking_computed(self):
        g = _block("g")
        d = _dsim(blocks=[g])
        d.engine.execute_block.return_value = None
        assert d._execute_ready_block(g, {}) is False
        assert g.computed_data is False
        d.engine.propagate_outputs.assert_not_called()


@pytest.mark.unit
class TestRunHierarchyPasses:
    def _wire(self, d, wiring):
        """propagate_outputs delivers one input to each consumer in ``wiring``."""

        def propagate(block, out_value, count=True):
            if count:
                for consumer in wiring.get(block.name, ()):
                    consumer.data_received += 1

        d.engine.propagate_outputs.side_effect = propagate

    def test_within_level_reordering(self):
        # b consumes a but precedes it in the list; the inner repeat picks it up.
        a = _block("a", in_ports=0, data_received=0)
        b = _block("b", in_ports=1, data_received=0)
        d = _dsim(blocks=[b, a])
        self._wire(d, {"a": [b]})
        assert d._run_hierarchy_passes([b, a], {}) is True
        assert [c.args[0].name for c in d.engine.execute_block.call_args_list] == ["a", "b"]

    def test_cross_level_repass_reaches_memory_block(self):
        # mem is pinned to hierarchy 0 but its state update needs gain (hier 1).
        src = _block("src", in_ports=0, data_received=0)
        mem = _block("mem", b_type=1, in_ports=1, data_received=0)
        gain = _block("gain", hierarchy=1, in_ports=1, data_received=0)
        d = _dsim(blocks=[src, mem, gain])
        d.memory_blocks = {"mem"}
        d.max_hier = 1
        self._wire(d, {"src": [gain], "gain": [mem]})
        assert d._run_hierarchy_passes([src, mem, gain], {}) is True
        assert [c.args[0].name for c in d.engine.execute_block.call_args_list] == [
            "src",
            "gain",
            "mem",
        ]
        assert all(b.computed_data for b in (src, mem, gain))

    def test_each_block_fires_at_most_once(self):
        a = _block("a", in_ports=0, data_received=0)
        d = _dsim(blocks=[a])
        d.max_hier = 2
        assert d._run_hierarchy_passes([a], {}) is True
        assert d.engine.execute_block.call_count == 1

    def test_failure_aborts_passes(self):
        a = _block("a", in_ports=0, data_received=0)
        b = _block("b", in_ports=0, data_received=0)
        d = _dsim(blocks=[a, b])
        d.engine.execute_block.return_value = None
        assert d._run_hierarchy_passes([a, b], {}) is False
        assert d.engine.execute_block.call_count == 1


@pytest.mark.unit
class TestFinishRun:
    def test_headless_freezes_timeline_and_rearms_blocks(self):
        d = _dsim()
        d._timeline_list = [0.0, 0.1, 0.2]
        d._finish_run(interactive=False)
        assert np.array_equal(d.timeline, [0.0, 0.1, 0.2])
        assert d.execution_initialized is False
        d.engine.reset_memblocks.assert_called_once()
        d.export_data.assert_not_called()
        d.pyqtPlotScope.assert_not_called()
        d.pbar.close.assert_not_called()

    def test_interactive_exports_records_and_plots(self):
        d = _dsim()
        d._finish_run(interactive=True)
        d.pbar.close.assert_called_once()
        d.export_data.assert_called_once()
        d._record_run_history.assert_called_once()
        d.pyqtPlotScope.assert_called_once()

    def test_interactive_skips_plot_when_deferred_or_dynamic(self):
        d = _dsim()
        d._defer_gui_plots = True
        d._finish_run(interactive=True)
        d.pyqtPlotScope.assert_not_called()
        d = _dsim()
        d.dynamic_plot = True
        d._finish_run(interactive=True)
        d.pyqtPlotScope.assert_not_called()

    def test_run_history_failure_is_swallowed(self):
        d = _dsim()
        d._record_run_history.side_effect = RuntimeError("no inspector")
        d._finish_run(interactive=True)
        d.pyqtPlotScope.assert_called_once()


@pytest.mark.unit
class TestInterpreterStep:
    def test_paused_does_nothing(self):
        d = _dsim()
        d.execution_pause = True
        d._interpreter_step(interactive=False)
        assert d.time_step == 0.0
        d.engine.reset_execution_data.assert_not_called()

    def test_finished_run_only_advances_clock(self):
        d = _dsim(sim_dt=0.1)
        d.execution_initialized = False
        d._interpreter_step(interactive=False)
        assert np.isclose(d.time_step, 0.1)
        assert d._timeline_list == []
        d.engine.execute_block.assert_not_called()

    def test_normal_step_runs_blocks_and_advances_counter(self):
        a = _block("a", in_ports=0, data_received=0)
        d = _dsim(sim_dt=0.1, execution_time=1.0, blocks=[a])
        d._interpreter_step(interactive=False)
        d.engine.reset_execution_data.assert_called_once()
        d.engine.execute_block.assert_called_once_with(a)
        assert len(d._timeline_list) == 1
        assert d.rk_counter == 1
        assert d.execution_initialized is True

    def test_block_failure_stops_run_without_advancing_counter(self):
        a = _block("a", in_ports=0, data_received=0)
        d = _dsim(blocks=[a])
        d.engine.execute_block.return_value = {"E": True, "error": "nope"}
        d._interpreter_step(interactive=False)
        assert d.rk_counter == 0
        assert d.execution_initialized is False
        assert d.error_msg == "nope"

    def test_last_sample_finishes_run(self):
        a = _block("a", in_ports=0, data_received=0)
        d = _dsim(sim_dt=0.5, execution_time=1.0, blocks=[a])
        d.time_step = 0.5
        d._interpreter_step(interactive=False)  # records t=1.0, the last grid point
        assert d.execution_initialized is False
        assert np.allclose(d.timeline, [1.0])
        d.engine.reset_memblocks.assert_called_once()

    def test_unexpected_exception_is_reported(self):
        a = _block("a", in_ports=0, data_received=0)
        d = _dsim(blocks=[a])
        d.engine.execute_block.side_effect = ValueError("surprise")
        d._interpreter_step(interactive=False)
        assert d.execution_initialized is False
        assert "surprise" in d.error_msg
