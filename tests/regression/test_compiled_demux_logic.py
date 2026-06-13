"""
Regression test for the ``compile-demux-logic`` feature.

The compiled fast path (``SystemCompiler`` / ``run_compiled_simulation``)
gates on ``check_compilability``: if ANY block in the (flattened) diagram is
not in ``COMPILABLE_BLOCKS`` it falls back to the slow interpreter, so a
single Demux or logic gate forfeited the speedup for the whole diagram.

This adds compiled executors for:

  * ``Demux``           — splits its single vector input into N consecutive
                          sub-vectors of length ``output_shape`` each, matching
                          ``blocks/demux.py``. Port 0 is stored at the block
                          signal key; secondary ports use the
                          ``"{name}_out{i}"`` convention.
  * ``LogicalOperator`` — applies the configured boolean op
                          (AND/OR/NAND/NOR/XOR/NOT) element-wise over its
                          inputs (nonzero = True), matching
                          ``blocks/logical_operator.py``.

The tests below build a diagram containing both blocks and assert:

  1. ``check_compilability`` now returns True (no interpreter fallback), and
  2. the Scope trace produced by the COMPILED fast path agrees, sample for
     sample, with the trace produced by the INTERPRETER for every logical
     operator — proving the compiled executors reproduce the block semantics.

A second test feeds the logic output into an Integrator so the new executors
run inside the compiled ODE right-hand-side (not only the post-solve replay).
"""

from pathlib import Path

import numpy as np
import pytest
from PyQt5.QtCore import QRect, QPoint
from PyQt5.QtGui import QColor


EXAMPLE = "test_demux_logic.diablos"
EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"


def _load_example():
    """Load the Demux+Logic example and return the configured DSim."""
    from lib.lib import DSim
    from lib.workspace import WorkspaceManager

    WorkspaceManager._instance = None
    dsim = DSim()
    data = dsim.file_service.load(filepath=str(EXAMPLES_DIR / EXAMPLE))
    assert data is not None, f"Failed to load {EXAMPLE}"
    dsim.file_service.apply_loaded_data(data)
    return dsim


def _scope_vector(dsim):
    """Return the (n_samples, vec_dim) ndarray captured by the diagram's Scope."""
    for b in dsim.engine.active_blocks_list:
        if b.block_fn != "Scope":
            continue
        params = getattr(b, "exec_params", b.params)
        vec = params.get("vector")
        if vec is None:
            continue
        vec_dim = params.get("vec_dim", 1)
        return np.asarray(vec).reshape(-1, vec_dim)
    raise AssertionError("No Scope with a captured vector found")


def _run(operator, use_fast):
    """Load the example, force the logic operator, run, return the Scope trace."""
    dsim = _load_example()
    for b in dsim.model.blocks_list:
        if b.block_fn == "LogicalOperator":
            b.params["operator"] = operator
    dsim.use_fast_solver = use_fast
    ok, err = dsim.run_tuning_simulation(dsim.sim_time, dsim.sim_dt)
    assert ok, f"Simulation (operator={operator}, use_fast={use_fast}) failed: {err}"
    return _scope_vector(dsim)


@pytest.mark.regression
class TestCompiledDemuxLogic:
    def test_diagram_is_compilable_no_fallback(self, qapp):
        """The Demux + LogicalOperator diagram must compile (no interpreter
        fallback). Before this feature, either block forced the slow path."""
        from lib.engine.system_compiler import SystemCompiler

        dsim = _load_example()
        assert dsim.engine.initialize_execution(
            dsim.model.blocks_list, dsim.model.line_list
        )
        compiler = SystemCompiler()
        assert "Demux" in compiler.COMPILABLE_BLOCKS
        assert "LogicalOperator" in compiler.COMPILABLE_BLOCKS
        assert compiler.check_compilability(dsim.engine.active_blocks_list), (
            "Demux + LogicalOperator diagram should be compilable now; "
            "check_compilability returned False (would fall back to interpreter)."
        )

    @pytest.mark.parametrize("operator", ["AND", "OR", "NAND", "NOR", "XOR"])
    def test_compiled_matches_interpreter(self, qapp, operator):
        """
        Constant [1, 0, 1] -> Demux(3) -> LogicalOperator -> Scope, with each
        demux output also wired to the Scope.

        The compiled fast path and the interpreter must produce identical Scope
        traces: this exercises the new Demux split semantics (ports 0/1/2) and
        every boolean operator, proving the compiled executors match
        blocks/demux.py and blocks/logical_operator.py.
        """
        compiled = _run(operator, use_fast=True)
        interp = _run(operator, use_fast=False)

        assert compiled.shape == interp.shape, (
            f"[{operator}] shape mismatch: compiled={compiled.shape} "
            f"interpreter={interp.shape}"
        )
        max_diff = float(np.max(np.abs(compiled - interp)))
        assert max_diff < 1e-9, (
            f"[{operator}] compiled vs interpreter Scope traces disagree by "
            f"max {max_diff:.3e}. compiled[0]={compiled[0]}, "
            f"interpreter[0]={interp[0]}"
        )

    def test_demux_split_values_are_correct(self, qapp):
        """The three demux columns of the Scope must be the split components of
        the [1, 0, 1] constant (port 0 -> 1, port 1 -> 0, port 2 -> 1)."""
        compiled = _run("XOR", use_fast=True)
        # Scope layout: [demux_out0, demux_out1, demux_out2, logic_out]
        np.testing.assert_allclose(compiled[0, 0:3], [1.0, 0.0, 1.0])
        # XOR over (1, 0, 1) is odd-parity over two True -> False == 0.0
        assert np.isclose(compiled[0, 3], 0.0)

    def test_logic_output_feeds_compiled_ode_rhs(self, qapp):
        """
        Constant [1, 1] -> Demux(2) -> AND -> Integrator.

        Here the Demux and LogicalOperator executors run inside the compiled
        ODE right-hand side (the Integrator's derivative), not only the
        post-solve replay. AND(1, 1) = 1, so the integrator output must be ~t.
        """
        from lib.simulation.block import DBlock
        from lib.simulation.connection import DLine
        from lib.engine.system_compiler import SystemCompiler
        from lib.workspace import WorkspaceManager
        from scipy.integrate import solve_ivp

        def mk(fn, sid, inp, outp, params, b_type=2):
            return DBlock(
                block_fn=fn, sid=sid, coords=QRect(0, 0, 50, 40),
                color=QColor(150, 150, 150), in_ports=inp, out_ports=outp,
                params=params, username="", b_type=b_type,
            )

        def ln(sid, src, sp, dst, dp):
            return DLine(sid=sid, srcblock=src, srcport=sp, dstblock=dst,
                         dstport=dp, points=[QPoint(0, 0), QPoint(1, 0)])

        blocks = [
            mk("Constant", 0, 0, 1, {"value": [1.0, 1.0]}, b_type=0),
            mk("Demux", 0, 1, 2, {"output_shape": 1}),
            mk("LogicalOperator", 0, 2, 1, {"operator": "AND", "_inputs_": 2}),
            mk("Integrator", 0, 1, 1, {"init_conds": 0.0}, b_type=1),
        ]
        lines = [
            ln(0, "constant0", 0, "demux0", 0),
            ln(1, "demux0", 0, "logicaloperator0", 0),
            ln(2, "demux0", 1, "logicaloperator0", 1),
            ln(3, "logicaloperator0", 0, "integrator0", 0),
        ]

        WorkspaceManager._instance = None
        wm = WorkspaceManager()
        for b in blocks:
            b.exec_params = wm.resolve_params(b.params)
            b.exec_params.update(
                {k: v for k, v in b.params.items() if k.startswith("_")}
            )
            b.exec_params["dtime"] = 0.01

        compiler = SystemCompiler()
        assert compiler.check_compilability(blocks)
        model_func, y0, state_map, _ = compiler.compile_system(blocks, blocks, lines)
        sol = solve_ivp(
            model_func, (0, 2.0), y0, method="RK45",
            t_eval=np.linspace(0, 2, 50), rtol=1e-8, atol=1e-10,
        )
        assert sol.success, f"solve_ivp failed: {sol.message}"
        start, _ = state_map["integrator0"]
        assert np.isclose(sol.y[start, -1], 2.0, atol=1e-4), (
            f"AND(1,1)=1 integrated over [0,2] should be ~2.0, "
            f"got {sol.y[start, -1]}"
        )
