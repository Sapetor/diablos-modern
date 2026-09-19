"""Compiled-vs-interpreted equivalence for the PID block in a feedback loop.

The PID is a *feedthrough* state block (its output depends on the current error),
so the compiled path executes it with the algebraic middle group while the plant
in the loop (a strictly-proper TranFn, D=0) is pre-populated and executes last.
This file builds the canonical closed loop that exercises that path:

    Step(setpoint=1) --> PID(setpoint, measurement) --> TranFn plant 1/(s+1) --> Scope
                          ^------------------- feedback ------------------|

and runs it through BOTH the interpreter (block.execute() time-step loop) and the
compiled solver (SystemCompiler + solve_ivp), reading the Scope trace from each.

Result of building this test: the two paths **genuinely diverge** in the
transient, so the trajectory-equivalence assertion is marked xfail (see the
reason string on ``test_pid_loop_trajectory_equivalence`` for the full analysis).
The equivalence that *does* hold -- both paths reach the same steady state -- is
pinned by ``test_pid_loop_reaches_shared_steady_state``, which passes.

Divergence summary (measured while writing this test; compiled path verified
against two closed-form analytic solutions, so the compiled trajectory is the
reference and the interpreter is the inaccurate one):

  * The compiled path integrates the *continuous* ODE and matches the analytic
    closed-loop response to machine precision (pure-P loop: (2/3)(1-e^{-3t});
    PI+integrator loop: 1+(t-1)e^{-t}).
  * The interpreter diverges by ~0.10 (abs) in the transient of the full PID
    loop at the default dt=0.01. A separate dtime-clobber bug (the engine
    re-stamped every block's exec_params['dtime'] with its default 0.01 during
    run_tuning_simulation, so interpreter state blocks integrated at 0.01 s per
    step regardless of sim_dt) has since been fixed by syncing engine.sim_dt
    before initialize_execution, so the interpreter is no longer pinned to
    dt=0.01. This test runs at dt=0.01, where that clobber was masked anyway.
  * At sim_dt=0.01 the transient still differs by up to ~0.10 (RMS ~0.015)
    because of the one-sample feedback delay inherent to the interpreter's
    memory-block loop, versus the compiled path's algebraic loop resolution.
    That is transient-only and first order in dt; the steady state agrees. The
    difference is what keeps the trajectory xfail at this test's fixed dt=0.01.
    (A second cause -- the derivative branch filtering a finite difference
    seeded from the first error sample, which deleted the step response of the
    D term entirely -- has been fixed: blocks/pid.py now carries the same
    filtered-error state the compiled kernel does, x_d' = N(e - x_d) from zero.
    That alone took the divergence from ~0.26 to ~0.10 and restored first-order
    convergence of the interpreted loop; see tests/validation/test_closed_loop.py.)
"""

import numpy as np
import pytest


def _defaults(block):
    return {
        k: v["default"] if isinstance(v, dict) and "default" in v else v
        for k, v in block.params.items()
    }


def _build_pid_loop(dsim, kp, ki, kd):
    """Populate ``dsim`` with Step -> PID -> TranFn(1/(s+1)) -> Scope + feedback."""
    from PyQt6.QtCore import QRect, QPoint
    from blocks.step import StepBlock
    from blocks.pid import PIDBlock
    from blocks.transfer_function import TransferFunctionBlock
    from blocks.scope import ScopeBlock
    from lib.simulation.block import DBlock
    from lib.simulation.connection import DLine

    step_b = StepBlock()
    step = DBlock(
        "Step",
        1,
        coords=QRect(0, 0, 50, 50),
        color="blue",
        in_ports=0,
        out_ports=1,
        b_type=step_b.b_type,
        params=_defaults(step_b),
        block_class=StepBlock,
        category=step_b.category,
    )
    step.params["value"] = 1.0
    step.params["delay"] = 0.0

    pid_b = PIDBlock()
    pid = DBlock(
        "PID",
        1,
        coords=QRect(100, 0, 50, 50),
        color="magenta",
        in_ports=2,
        out_ports=1,
        b_type=2,
        params=_defaults(pid_b),
        block_class=PIDBlock,
        category=pid_b.category,
    )
    pid.params["Kp"] = kp
    pid.params["Ki"] = ki
    pid.params["Kd"] = kd

    plant_b = TransferFunctionBlock()
    plant = DBlock(
        "TranFn",
        1,
        coords=QRect(200, 0, 50, 50),
        color="green",
        in_ports=1,
        out_ports=1,
        b_type=plant_b.b_type,
        params=_defaults(plant_b),
        block_class=TransferFunctionBlock,
        category=plant_b.category,
    )
    plant.params["numerator"] = [1.0]
    plant.params["denominator"] = [1.0, 1.0]

    scope_b = ScopeBlock()
    scope = DBlock(
        "Scope",
        1,
        coords=QRect(300, 0, 50, 50),
        color="red",
        in_ports=1,
        out_ports=0,
        b_type=scope_b.b_type,
        params=_defaults(scope_b),
        block_class=ScopeBlock,
        category=scope_b.category,
    )
    scope.params["labels"] = "y"

    lines = [
        DLine(
            sid=0,
            srcblock=step.name,
            srcport=0,
            dstblock=pid.name,
            dstport=0,
            points=[QPoint(0, 0), QPoint(1, 1)],
        ),
        DLine(
            sid=1,
            srcblock=pid.name,
            srcport=0,
            dstblock=plant.name,
            dstport=0,
            points=[QPoint(1, 1), QPoint(2, 2)],
        ),
        DLine(
            sid=2,
            srcblock=plant.name,
            srcport=0,
            dstblock=pid.name,
            dstport=1,
            points=[QPoint(2, 2), QPoint(1, 1)],
        ),
        DLine(
            sid=3,
            srcblock=plant.name,
            srcport=0,
            dstblock=scope.name,
            dstport=0,
            points=[QPoint(2, 2), QPoint(3, 3)],
        ),
    ]

    dsim.model.blocks_list[:] = [step, pid, plant, scope]
    dsim.model.line_list[:] = lines
    dsim.blocks_list = dsim.model.blocks_list
    dsim.line_list = dsim.model.line_list
    dsim.connections_list = dsim.line_list


def _scope_trace(dsim):
    """Return the Scope block's (n_samples, vec_dim) trace, or None."""
    for b in dsim.engine.active_blocks_list:
        if b.block_fn != "Scope":
            continue
        params = getattr(b, "exec_params", b.params)
        vec = params.get("vector")
        if vec is None:
            return None
        return np.asarray(vec, dtype=float).reshape(-1, params.get("vec_dim", 1))
    return None


def _run(kp, ki, kd, fast, dt, t_end):
    """Build and run the PID loop; return (timeline, scope_trace)."""
    from lib.lib import DSim
    from lib.workspace import WorkspaceManager

    WorkspaceManager._instance = None
    dsim = DSim()
    _build_pid_loop(dsim, kp, ki, kd)
    dsim.use_fast_solver = fast
    ok, err = dsim.run_tuning_simulation(t_end, dt)
    assert ok, "run_tuning_simulation(fast=%s) failed: %r" % (fast, err)
    return np.asarray(dsim.engine.timeline, dtype=float), _scope_trace(dsim)


# Full PID gains: proportional + integral + filtered-derivative are all active,
# so the compiled path routes the P/I/D feedthrough through the algebraic middle
# group while the D=0 plant is pre-populated and integrated last.
KP, KI, KD = 2.0, 1.0, 0.5
# The interpreter's fixed-dtime discretization is only self-consistent at 0.01
# (see module docstring), so the comparison is done at the default step.
DT = 0.01


@pytest.mark.regression
class TestPIDCompiledVsInterpreted:
    @pytest.mark.xfail(
        strict=False,
        reason=(
            "Interpreter and compiled paths diverge in the transient of a PID "
            "feedback loop (max ~0.10 abs, RMS ~0.015 at dt=0.01). The compiled "
            "path matches the analytic continuous closed-loop response to machine "
            "precision; the interpreter is the inaccurate one, from the one-sample "
            "feedback delay of the memory-block loop, which is first order in dt "
            "and so does not vanish at this test's fixed dt=0.01. (Two other "
            "causes -- state blocks pinned to dtime=0.01 regardless of sim_dt, and "
            "a derivative branch that never saw the reference step -- have since "
            "been fixed.) See module docstring."
        ),
    )
    def test_pid_loop_trajectory_equivalence(self, qapp):
        """The two paths should track the same PID-loop trajectory (they do not)."""
        tl_i, y_i = _run(KP, KI, KD, fast=False, dt=DT, t_end=5.0)
        tl_c, y_c = _run(KP, KI, KD, fast=True, dt=DT, t_end=5.0)
        assert y_i is not None and y_c is not None, "missing scope trace"

        # Compare at the interpreter's sample times (both run on the same dt grid,
        # so index i is time i*dt in each); trim to the shorter run.
        n = min(len(y_i), len(y_c))
        interp = y_i[:n, 0]
        compiled = y_c[:n, 0]

        # Tolerance appropriate to ODE-solver-vs-fixed-step differences.
        assert np.allclose(interp, compiled, rtol=1e-2, atol=1e-2), (
            "PID-loop trajectories diverge: max|delta|=%.4f"
            % float(np.max(np.abs(interp - compiled)))
        )

    def test_pid_loop_reaches_shared_steady_state(self, qapp):
        """Both paths settle the PID loop to the setpoint (steady state agrees).

        This is the compiled-vs-interpreted equivalence that genuinely holds for
        the full PID loop: given enough settling time, the integral action drives
        the plant output of both engines to the setpoint of 1.0. Runs at the
        default dt=0.01, where the interpreter's state blocks are self-consistent.
        """
        t_end = 25.0
        _, y_i = _run(KP, KI, KD, fast=False, dt=DT, t_end=t_end)
        _, y_c = _run(KP, KI, KD, fast=True, dt=DT, t_end=t_end)
        assert y_i is not None and y_c is not None, "missing scope trace"

        final_interp = float(y_i[-1, 0])
        final_compiled = float(y_c[-1, 0])

        assert np.isclose(final_interp, 1.0, atol=2e-3), (
            "interpreter did not settle to setpoint: %.6f" % final_interp
        )
        assert np.isclose(final_compiled, 1.0, atol=2e-3), (
            "compiled did not settle to setpoint: %.6f" % final_compiled
        )
        assert np.isclose(final_interp, final_compiled, atol=2e-3), (
            "steady states disagree: interp=%.6f compiled=%.6f" % (final_interp, final_compiled)
        )


def _build_error_input_pid_loop(dsim, kp, ki, kd):
    """Populate ``dsim`` with the *error-input* PID wiring:

        Step(1) --> Sum(+,-) --> PID(port 0 only) --> TranFn 1/(s+1) --> Scope
                      ^------------- feedback -------------|

    This is the topology the September 2026 codegen agent reported as broken on
    the compiled path (see tasks/todo.md). It differs from ``_build_pid_loop``
    in that the subtraction happens in an explicit Sum block rather than inside
    the PID, so the PID has a single connected input and a *feedthrough* block
    (the Sum) sits between the sources and the controller.
    """
    from PyQt6.QtCore import QRect, QPoint
    from blocks.step import StepBlock
    from blocks.sum import SumBlock
    from blocks.pid import PIDBlock
    from blocks.transfer_function import TransferFunctionBlock
    from blocks.scope import ScopeBlock
    from lib.simulation.block import DBlock
    from lib.simulation.connection import DLine

    def _mk(fn, block_obj, cls, n_in, n_out, x, **overrides):
        blk = DBlock(
            fn,
            1,
            coords=QRect(x, 0, 50, 50),
            color="blue",
            in_ports=n_in,
            out_ports=n_out,
            b_type=getattr(block_obj, "b_type", 2),
            params=_defaults(block_obj),
            block_class=cls,
            category=block_obj.category,
        )
        blk.params.update(overrides)
        return blk

    step = _mk("Step", StepBlock(), StepBlock, 0, 1, 0, value=1.0, delay=0.0)
    summ = _mk("Sum", SumBlock(), SumBlock, 2, 1, 100, sign="+-")
    pid = _mk("PID", PIDBlock(), PIDBlock, 1, 1, 200, Kp=kp, Ki=ki, Kd=kd)
    plant = _mk(
        "TranFn",
        TransferFunctionBlock(),
        TransferFunctionBlock,
        1,
        1,
        300,
        numerator=[1.0],
        denominator=[1.0, 1.0],
    )
    scope = _mk("Scope", ScopeBlock(), ScopeBlock, 1, 0, 400, labels="y")

    def _line(sid, src, src_port, dst, dst_port):
        return DLine(
            sid=sid,
            srcblock=src.name,
            srcport=src_port,
            dstblock=dst.name,
            dstport=dst_port,
            points=[QPoint(0, 0), QPoint(1, 1)],
        )

    lines = [
        _line(0, step, 0, summ, 0),
        _line(1, plant, 0, summ, 1),
        _line(2, summ, 0, pid, 0),
        _line(3, pid, 0, plant, 0),
        _line(4, plant, 0, scope, 0),
    ]

    dsim.model.blocks_list[:] = [step, summ, pid, plant, scope]
    dsim.model.line_list[:] = lines
    dsim.blocks_list = dsim.model.blocks_list
    dsim.line_list = dsim.model.line_list
    dsim.connections_list = dsim.line_list


def _run_error_input(kp, ki, kd, fast, dt, t_end):
    from lib.lib import DSim
    from lib.workspace import WorkspaceManager

    WorkspaceManager._instance = None
    dsim = DSim()
    _build_error_input_pid_loop(dsim, kp, ki, kd)
    dsim.use_fast_solver = fast
    ok, err = dsim.run_tuning_simulation(t_end, dt)
    assert ok, "run_tuning_simulation(fast=%s) failed: %r" % (fast, err)
    return _scope_trace(dsim)


@pytest.mark.regression
class TestErrorInputPIDOrdering:
    """The Sum feeding a 1-port PID must execute *before* it on both paths.

    Both engines used to freeze this loop at exactly zero, for two independent
    reasons:

    * **Compiled path (the reported ordering bug).** The engine's hierarchy sort
      is memory-block aware and emits ``[Step, PID, TranFn, Sum]`` so the
      interpreter's Loop 1 can break the cycle with the previous step's inputs.
      The compiled middle group inherited that order verbatim, so ``exec_pid``
      read ``signals['Sum']`` before the Sum had written it -- 0.0 on every RHS
      evaluation, a permanently zero error, and a dead loop. Fixed by
      ``_dataflow_order`` in ``lib/engine/system_compiler.py``, which re-sorts
      the middle group by true dataflow.
    * **Interpreted path.** ``blocks/pid.py`` returned ``_last_output_`` whenever
      port 1 was unconnected, so the error-input wiring never computed at all.
      An unconnected measurement port now reads as 0.0, matching the compiled
      kernel (``build_pid`` leaves ``meas_src`` None).
    """

    def test_middle_group_runs_the_sum_before_the_pid(self, qapp):
        """The compiled middle group is in dataflow order, not hierarchy order."""
        from lib.engine.system_compiler import _dataflow_order

        class _B:
            def __init__(self, name):
                self.name = name

        pid, summ, scope = _B("pid1"), _B("sum1"), _B("scope1")
        # The middle group as the engine's memory-block-aware sort hands it over,
        # with the PID ahead of the Sum that feeds it.
        middle = [pid, summ, scope]
        input_map = {
            "sum1": {0: ("step1", 0), 1: ("tranfn1", 0)},
            "pid1": {0: ("sum1", 0)},
            "tranfn1": {0: ("pid1", 0)},
            "scope1": {0: ("tranfn1", 0)},
        }
        ordered = [b.name for b in _dataflow_order(middle, input_map)]
        assert ordered.index("sum1") < ordered.index("pid1"), (
            "Sum must execute before the PID it feeds, got %r" % ordered
        )
        assert sorted(ordered) == sorted(["pid1", "sum1", "scope1"]), (
            "dataflow ordering must be a permutation of the middle group, got %r" % ordered
        )

    def test_error_input_pid_loop_does_not_freeze_at_zero(self, qapp):
        """Neither engine leaves the error-input PID loop dead at zero."""
        for fast in (False, True):
            y = _run_error_input(KP, KI, KD, fast=fast, dt=DT, t_end=25.0)
            assert y is not None, "missing scope trace (fast=%s)" % fast
            assert float(np.max(np.abs(y[:, 0]))) > 1e-6, (
                "error-input PID loop frozen at zero (fast=%s)" % fast
            )

    def test_error_input_pid_loop_reaches_the_setpoint(self, qapp):
        """Integral action drives both paths to the setpoint of 1.0."""
        y_i = _run_error_input(KP, KI, KD, fast=False, dt=DT, t_end=25.0)
        y_c = _run_error_input(KP, KI, KD, fast=True, dt=DT, t_end=25.0)
        assert y_i is not None and y_c is not None, "missing scope trace"

        final_interp = float(y_i[-1, 0])
        final_compiled = float(y_c[-1, 0])
        assert np.isclose(final_interp, 1.0, atol=2e-3), (
            "interpreted error-input PID loop settled at %.6f, expected 1.0" % final_interp
        )
        assert np.isclose(final_compiled, 1.0, atol=2e-3), (
            "compiled error-input PID loop settled at %.6f, expected 1.0" % final_compiled
        )
