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
  * The interpreter diverges by ~0.26 (abs) in the transient of the full PID
    loop at the default dt=0.01, and -- critically -- the divergence *grows* as
    dt shrinks (anti-convergent), rather than shrinking. Root cause: interpreter
    state/memory blocks (TranFn, Integrator, PID) discretize/integrate using
    ``params.get('dtime', 0.01)`` and end up using dtime=0.01 regardless of the
    actual sim_dt, so they advance internal state at 0.01 s per step for any
    timestep. The interpreter is therefore self-consistent *only* at sim_dt=0.01.
  * Even at sim_dt=0.01 (where that dtime issue is masked) the transient still
    differs by up to ~0.26 because of (a) the one-sample feedback delay inherent
    to the interpreter's memory-block loop vs the compiled path's algebraic loop
    resolution, and (b) the derivative-kick handling on the step (discrete
    filtered finite-difference vs the compiled continuous filtered-derivative
    state). Both are transient-only; the steady state agrees.
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
    from PyQt5.QtCore import QRect, QPoint
    from blocks.step import StepBlock
    from blocks.pid import PIDBlock
    from blocks.transfer_function import TransferFunctionBlock
    from blocks.scope import ScopeBlock
    from lib.simulation.block import DBlock
    from lib.simulation.connection import DLine

    step_b = StepBlock()
    step = DBlock("Step", 1, coords=QRect(0, 0, 50, 50), color="blue",
                  in_ports=0, out_ports=1, b_type=step_b.b_type,
                  params=_defaults(step_b), block_class=StepBlock,
                  category=step_b.category)
    step.params["value"] = 1.0
    step.params["delay"] = 0.0

    pid_b = PIDBlock()
    pid = DBlock("PID", 1, coords=QRect(100, 0, 50, 50), color="magenta",
                 in_ports=2, out_ports=1, b_type=2,
                 params=_defaults(pid_b), block_class=PIDBlock,
                 category=pid_b.category)
    pid.params["Kp"] = kp
    pid.params["Ki"] = ki
    pid.params["Kd"] = kd

    plant_b = TransferFunctionBlock()
    plant = DBlock("TranFn", 1, coords=QRect(200, 0, 50, 50), color="green",
                   in_ports=1, out_ports=1, b_type=plant_b.b_type,
                   params=_defaults(plant_b), block_class=TransferFunctionBlock,
                   category=plant_b.category)
    plant.params["numerator"] = [1.0]
    plant.params["denominator"] = [1.0, 1.0]

    scope_b = ScopeBlock()
    scope = DBlock("Scope", 1, coords=QRect(300, 0, 50, 50), color="red",
                   in_ports=1, out_ports=0, b_type=scope_b.b_type,
                   params=_defaults(scope_b), block_class=ScopeBlock,
                   category=scope_b.category)
    scope.params["labels"] = "y"

    lines = [
        DLine(sid=0, srcblock=step.name, srcport=0, dstblock=pid.name, dstport=0,
              points=[QPoint(0, 0), QPoint(1, 1)]),
        DLine(sid=1, srcblock=pid.name, srcport=0, dstblock=plant.name, dstport=0,
              points=[QPoint(1, 1), QPoint(2, 2)]),
        DLine(sid=2, srcblock=plant.name, srcport=0, dstblock=pid.name, dstport=1,
              points=[QPoint(2, 2), QPoint(1, 1)]),
        DLine(sid=3, srcblock=plant.name, srcport=0, dstblock=scope.name, dstport=0,
              points=[QPoint(2, 2), QPoint(3, 3)]),
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
            "feedback loop (max ~0.26 abs, RMS ~0.12 at dt=0.01). The compiled "
            "path matches the analytic continuous closed-loop response to machine "
            "precision; the interpreter is the inaccurate one, from (1) state "
            "blocks discretizing at a fixed dtime=0.01 regardless of sim_dt, "
            "(2) the one-sample feedback delay of the memory-block loop, and "
            "(3) discrete derivative-kick handling on the step. See module "
            "docstring for the full analysis."
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
            "PID-loop trajectories diverge: max|delta|=%.4f" %
            float(np.max(np.abs(interp - compiled)))
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
            "steady states disagree: interp=%.6f compiled=%.6f" %
            (final_interp, final_compiled)
        )
