"""Engine-side fixes from the 2026-09-05 audit (sections 1-3).

Four independent behaviours are pinned here:

* ``SimulationEngine._replay_has_feedthrough`` -- the post-solve replay's
  execution-order classification must read *resolved* params (or the compiled
  D matrix), not raw ``block.params``, where a workspace-variable-parameterised
  transfer function still holds variable *names*.
* the replay's Scope history -- a preallocated ``(num_steps, vec_dim)`` array
  rather than a grown Python list, with the shape/dtype the ScopePlotter reads.
* ``SimulationEngine.propagate_outputs`` -- now driven by a cached adjacency
  map; it must deliver exactly what the previous per-block/per-line scan did,
  and the cache must not survive a re-initialization.
* ``SimulationEngine.count_rk45_integrators`` -- must resolve method aliases
  the same way ``blocks.integrator`` does, or a diagram saved with the legacy
  "RK45" spelling silently loses the interpreter's 4-sub-step schedule.
"""

import os

import numpy as np
import pytest
from PyQt5.QtCore import QPoint, QRect
from PyQt5.QtGui import QColor

from lib.engine import graph_analysis
from lib.engine.simulation_engine import SimulationEngine
from lib.simulation.block import DBlock
from lib.simulation.connection import DLine

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXAMPLES = os.path.join(REPO_ROOT, "examples")


@pytest.fixture(autouse=True)
def _require_qapp(qapp):
    """DBlock builds QFont/QPixmap in its constructor, so every test here
    needs a live QApplication."""
    return qapp


class _MockModel:
    def __init__(self, blocks=None, lines=None):
        self.blocks_list = blocks or []
        self.line_list = lines or []
        self.variables = {}


def _block(block_fn, sid, in_ports, out_ports, params, b_type=2, block_class=None):
    return DBlock(
        block_fn=block_fn,
        sid=sid,
        coords=QRect(0, 0, 50, 40),
        color=QColor(150, 150, 150),
        in_ports=in_ports,
        out_ports=out_ports,
        params=params,
        username="",
        b_type=b_type,
        block_class=block_class,
    )


def _line(sid, src, srcport, dst, dstport):
    return DLine(
        sid=sid,
        srcblock=src,
        srcport=srcport,
        dstblock=dst,
        dstport=dstport,
        points=[QPoint(0, 0), QPoint(100, 0)],
    )


@pytest.fixture
def workspace_tf_coeffs():
    """Workspace variables for a strictly proper TF, restoring the singleton.

    The two names are deliberately the *same length*: the pre-fix code compared
    ``len(den) > len(num)`` on the raw strings, so equal-length names made a
    strictly proper 1/(s+1) look like a feedthrough block.
    """
    from lib.workspace import WorkspaceManager

    prev_instance = WorkspaceManager._instance
    WorkspaceManager._instance = None

    wm = WorkspaceManager()
    wm.variables = {"tf_num": [1.0], "tf_den": [1.0, 1.0]}

    yield wm

    WorkspaceManager._instance = prev_instance


@pytest.mark.unit
class TestReplayFeedthroughClassification:
    """Section 1: the replay must classify a workspace-variable TF like a
    literal one (it read raw ``params`` while the compiler used the resolved
    ones, so the two could disagree on execution order)."""

    def _resolve(self, block):
        from lib.workspace import WorkspaceManager

        block.exec_params = WorkspaceManager().resolve_params(block.params)
        return block

    def test_strictly_proper_tf_with_workspace_vars_matches_literal(self, workspace_tf_coeffs):
        literal = _block("TranFn", 0, 1, 1, {"numerator": [1.0], "denominator": [1.0, 1.0]})
        variable = self._resolve(
            _block("TranFn", 1, 1, 1, {"numerator": "tf_num", "denominator": "tf_den"})
        )

        # Same string length, so the old raw-params comparison said "feedthrough".
        assert len("tf_num") == len("tf_den")

        assert SimulationEngine._replay_has_feedthrough(literal, {}) is False
        assert SimulationEngine._replay_has_feedthrough(variable, {}) is False

    def test_proper_tf_is_feedthrough_with_and_without_workspace_vars(self):
        from lib.workspace import WorkspaceManager

        prev_instance = WorkspaceManager._instance
        WorkspaceManager._instance = None
        try:
            wm = WorkspaceManager()
            wm.variables = {"tf_num": [2.0, 1.0], "tf_den": [1.0, 1.0]}
            literal = _block(
                "TranFn", 0, 1, 1, {"numerator": [2.0, 1.0], "denominator": [1.0, 1.0]}
            )
            variable = _block("TranFn", 1, 1, 1, {"numerator": "tf_num", "denominator": "tf_den"})
            variable.exec_params = wm.resolve_params(variable.params)

            assert SimulationEngine._replay_has_feedthrough(literal, {}) is True
            assert SimulationEngine._replay_has_feedthrough(variable, {}) is True
        finally:
            WorkspaceManager._instance = prev_instance

    def test_compiled_block_matrices_win_over_params(self):
        """The compiled D matrix is authoritative when the compiler built one."""
        block = _block("TranFn", 0, 1, 1, {"numerator": [1.0], "denominator": [1.0, 1.0]})
        d0 = {block.name: (np.zeros((1, 1)), np.zeros((1, 1)), np.ones((1, 1)), np.zeros((1, 1)))}
        d_nonzero = {
            block.name: (np.zeros((1, 1)), np.zeros((1, 1)), np.ones((1, 1)), np.ones((1, 1)))
        }
        assert SimulationEngine._replay_has_feedthrough(block, d0) is False
        assert SimulationEngine._replay_has_feedthrough(block, d_nonzero) is True

    def test_integrator_is_never_feedthrough_and_gain_always_is(self):
        integrator = _block("Integrator", 0, 1, 1, {"init_conds": 0.0})
        gain = _block("Gain", 1, 1, 1, {"gain": 2.0})
        assert SimulationEngine._replay_has_feedthrough(integrator, {}) is False
        assert SimulationEngine._replay_has_feedthrough(gain, {}) is True


def _step_tf_scope(numerator, denominator):
    """Step -> TranFn -> Scope, with the TF coefficients supplied by caller."""
    from blocks.scope import ScopeBlock
    from blocks.step import StepBlock
    from blocks.transfer_function import TransferFunctionBlock

    step = _block(
        "Step",
        0,
        0,
        1,
        {"value": 1.0, "delay": 0.0, "type": "up"},
        b_type=0,
        block_class=StepBlock,
    )
    tf = _block(
        "TranFn",
        1,
        1,
        1,
        {"numerator": numerator, "denominator": denominator, "init_conds": 0.0},
        b_type=1,
        block_class=TransferFunctionBlock,
    )
    scope = _block("Scope", 2, 1, 0, {}, b_type=1, block_class=ScopeBlock)
    step.hierarchy, tf.hierarchy, scope.hierarchy = 0, 1, 2
    blocks = [step, tf, scope]
    lines = [
        _line(0, step.name, 0, tf.name, 0),
        _line(1, tf.name, 0, scope.name, 0),
    ]
    return blocks, lines


@pytest.mark.unit
class TestCompiledRunWithWorkspaceVariableTF:
    """End-to-end: a workspace-variable TF must produce the same recorded
    trace as the same diagram written with literal coefficients."""

    def _run(self, blocks, lines, t_end=2.0, dt=0.01):
        engine = SimulationEngine(_MockModel(blocks, lines))
        engine.update_sim_params(t_end, dt)
        assert engine.run_compiled_simulation(blocks, lines, (0.0, t_end), dt)
        scope = blocks[-1]
        return np.asarray(scope.exec_params["vector"], dtype=float)

    def test_traces_match_and_scope_history_is_preallocated(self, qapp, workspace_tf_coeffs):
        literal = self._run(*_step_tf_scope([1.0], [1.0, 1.0]))
        variable = self._run(*_step_tf_scope("tf_num", "tf_den"))

        # Shape/dtype contract the ScopePlotter reads: (samples, channels).
        assert literal.ndim == 2 and literal.shape[1] == 1
        assert literal.shape[0] == len(np.arange(0, 2.0 + 0.01, 0.01))
        np.testing.assert_allclose(variable, literal, rtol=1e-8, atol=1e-10)

        # Sanity: it really is the 1/(s+1) step response.
        assert literal[0, 0] == pytest.approx(0.0, abs=1e-9)
        assert literal[-1, 0] == pytest.approx(1 - np.exp(-2.0), abs=1e-3)


@pytest.mark.unit
class TestScopeHistoryBuffer:
    def test_history_is_an_array_not_a_list(self, qapp):
        blocks, lines = _step_tf_scope([1.0], [1.0, 1.0])
        engine = SimulationEngine(_MockModel(blocks, lines))
        engine.update_sim_params(0.5, 0.01)
        assert engine.run_compiled_simulation(blocks, lines, (0.0, 0.5), 0.01)

        vector = blocks[-1].exec_params["vector"]
        assert isinstance(vector, np.ndarray)
        assert vector.dtype == np.float64
        assert vector.shape == (len(engine.timeline), blocks[-1].exec_params["vec_dim"])


def _legacy_propagation(engine, block_name, out_value):
    """The pre-cache algorithm, kept here as the equivalence oracle."""
    children = graph_analysis.get_outputs(block_name, engine._active_line_source())
    target_blocks = (
        engine.active_blocks_list
        if len(engine.active_blocks_list) > 0
        else engine.model.blocks_list
    )
    deliveries = []
    for mblock in target_blocks:
        is_child, tuple_list = graph_analysis.is_child_of(mblock.name, children)
        if not is_child:
            continue
        for tuple_child in tuple_list:
            if tuple_child["srcport"] not in out_value:
                continue
            deliveries.append(
                (mblock.name, tuple_child["dstport"], out_value[tuple_child["srcport"]])
            )
    return deliveries


@pytest.mark.unit
class TestPropagationAdjacency:
    """Section 3: propagate_outputs uses a precomputed adjacency map instead of
    rescanning every block and every line per delivery."""

    def _engine_with_diagram(self):
        src = _block("Step", 0, 0, 2, {}, b_type=0)
        a = _block("Gain", 1, 1, 1, {}, b_type=2)
        b = _block("Sum", 2, 2, 1, {}, b_type=2)
        blocks = [src, a, b]
        lines = [
            _line(0, src.name, 0, a.name, 0),
            _line(1, src.name, 1, b.name, 0),
            _line(2, a.name, 0, b.name, 1),
        ]
        engine = SimulationEngine(_MockModel(blocks, lines))
        engine.active_blocks_list = blocks
        engine.active_line_list = lines
        return engine, blocks, lines

    def test_adjacency_matches_the_legacy_scan(self, qapp):
        engine, blocks, _lines = self._engine_with_diagram()
        out_value = {0: 1.0, 1: 2.0}

        for block in blocks:
            expected = _legacy_propagation(engine, block.name, out_value)
            actual = [
                (dst.name, dstport, out_value[srcport])
                for dst, srcport, dstport in engine._propagation_targets(block.name)
                if srcport in out_value
            ]
            assert sorted(actual) == sorted(expected), block.name

    def test_propagate_outputs_delivers_and_counts_once(self, qapp):
        engine, blocks, _lines = self._engine_with_diagram()
        src, gain, summer = blocks

        engine.propagate_outputs(src, {0: 5.0, 1: 7.0})
        assert gain.input_queue == {0: 5.0}
        assert summer.input_queue == {0: 7.0}
        assert (gain.data_received, summer.data_received, src.data_sent) == (1, 1, 2)

        # count=False refreshes a value without re-counting the arrival.
        engine.propagate_outputs(src, {0: 6.0, 1: 8.0}, count=False)
        assert (gain.input_queue[0], summer.input_queue[0]) == (6.0, 8.0)
        assert (gain.data_received, summer.data_received) == (1, 1)

    def test_missing_output_port_is_skipped_not_raised(self, qapp):
        engine, blocks, _lines = self._engine_with_diagram()
        src, gain, summer = blocks
        engine.propagate_outputs(src, {0: 5.0})  # port 1 absent
        assert gain.input_queue == {0: 5.0}
        assert summer.input_queue == {}

    def test_cache_is_invalidated_when_the_diagram_changes(self, qapp):
        engine, blocks, lines = self._engine_with_diagram()
        src = blocks[0]
        assert len(engine._propagation_targets(src.name)) == 2

        # Rewire in place, then invalidate as a re-initialization would.
        lines.pop()
        engine.invalidate_propagation_cache()
        assert len(engine._propagation_targets(src.name)) == 2
        assert engine._propagation_targets(blocks[1].name) == ()

    def test_lines_to_unknown_blocks_are_dropped(self, qapp):
        blocks = [_block("Gain", 0, 1, 1, {}, b_type=2)]
        lines = [_line(0, "ghost", 0, blocks[0].name, 0), _line(1, blocks[0].name, 0, "ghost", 0)]
        adjacency = SimulationEngine._build_propagation_adjacency(blocks, lines)
        assert list(adjacency) == ["ghost"]


@pytest.mark.unit
class TestPropagationOnAnExampleDiagram:
    """The adjacency must agree with the legacy scan on a real diagram.

    The diagram is only *loaded* (not simulated): propagation is a pure
    function of the block and line lists, and running a full DSim here leaves
    Qt plotting state behind that destabilises later GUI tests.
    """

    @pytest.mark.parametrize(
        "diagram_name", ["c01_tank_feedback.diablos", "c05_mass_spring_state_space.diablos"]
    )
    def test_example_diagram_targets_match_the_legacy_scan(self, qapp, diagram_name):
        from lib import cli

        dsim, _params = cli.load_diagram(os.path.join(EXAMPLES, diagram_name))
        engine = dsim.engine
        engine.active_blocks_list = list(dsim.blocks_list)
        engine.active_line_list = list(dsim.line_list)
        engine.invalidate_propagation_cache()
        assert engine.active_blocks_list, "example produced no blocks"
        assert engine.active_line_list, "example produced no connections"

        out_value = {0: 1.0, 1: 2.0, 2: 3.0}
        deliveries = 0
        for block in engine.active_blocks_list:
            expected = sorted(_legacy_propagation(engine, block.name, out_value))
            actual = sorted(
                (dst.name, dstport, out_value[srcport])
                for dst, srcport, dstport in engine._propagation_targets(block.name)
                if srcport in out_value
            )
            assert actual == expected, block.name
            deliveries += len(actual)
        assert deliveries >= len(engine.active_line_list) - 1


@pytest.mark.unit
class TestCountRk45Integrators:
    """The interpreter's 4-sub-step schedule must key off the *resolved*
    integrator method, so a diagram saved with the legacy "RK45" spelling of
    fixed-step RK4 still gets it."""

    def _engine_with_integrator(self, method):
        block = _block("Integrator", 0, 1, 1, {"method": method, "init_conds": 0.0}, b_type=1)
        engine = SimulationEngine(_MockModel([block], []))
        engine.active_blocks_list = [block]
        engine.active_line_list = []
        return engine

    @pytest.mark.parametrize("method", ["RK4", "RK45"])
    def test_both_spellings_of_rk4_are_detected(self, qapp, method):
        from blocks.integrator import resolve_method

        assert resolve_method(method) == "RK4"
        assert self._engine_with_integrator(method).count_rk45_integrators() is True

    @pytest.mark.parametrize("method", ["FWD_EULER", "SOLVE_IVP", "TUSTIN"])
    def test_other_methods_are_not_detected(self, qapp, method):
        assert self._engine_with_integrator(method).count_rk45_integrators() is False
