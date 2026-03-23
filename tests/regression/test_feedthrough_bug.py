"""
Regression tests for the D*u feedthrough omission bug in the compiled solver.

BUG DESCRIPTION
---------------
In ``SystemCompiler.compile_system`` the pre-population pass (step 4 in the
compiled model closure) initialises feedback signals for state blocks using::

    signals[b_name] = C @ x          # only the state-output term

For strictly-proper transfer functions (D = 0) this is exact.  For
non-strictly-proper transfer functions (D != 0) — such as a PI controller
``(2s+1)/s`` which has D = 2 — the proportional feedthrough term D*u is
dropped.  Any *algebraic* block (Sum, Gain, Saturation, …) that reads
``signals[b_name]`` before the state block's own executor runs will
therefore see an incorrect value and propagate it downstream, producing
wrong derivatives and ultimately wrong simulation trajectories.

The bug only manifests when a D != 0 state block's signal is consumed by at
least one algebraic block that executes before the state block's executor
(algebraic blocks always run before state blocks in the three-way ordering:
sources → algebraic → state blocks).

EXPECTED BEHAVIOUR AFTER FIX
------------------------------
The pre-population should use::

    signals[b_name] = C @ x + D @ u_last

where ``u_last`` is the input computed during the previous ODE step (or
zero at t=0), so that feedback signals seen by algebraic blocks are
accurate.

TEST RESULTS
------------
Tests that expose the bug (algebraic block reads D!=0 state block output):
  - test_pi_with_saturation_closed_loop: FAIL before fix, PASS after fix

Tests that do NOT expose the bug (no algebraic block reads D!=0 output):
  - test_pi_controller_closed_loop:         topology routes PI -> Plant (both
      state blocks); Sum only reads from the D=0 plant. PASS with or without fix.
  - test_lead_compensator_closed_loop:      same topology argument. PASS always.
  - test_pid_block_closed_loop:             no algebraic blocks in loop. PASS.
  - test_strictly_proper_tf_still_works:    D=0 sanity check. PASS always.
  - test_compiled_vs_analytical_three_tf_chain:  all-state chain. PASS always.
"""

import pytest
import numpy as np
from PyQt5.QtCore import QRect, QPoint
from PyQt5.QtGui import QColor


# ---------------------------------------------------------------------------
# Helpers (mirrored from test_feedback_loop.py)
# ---------------------------------------------------------------------------

def _make_block(block_fn, sid, username, in_ports, out_ports, params, b_type=2):
    """Create a DBlock with minimal boilerplate."""
    from lib.simulation.block import DBlock
    return DBlock(
        block_fn=block_fn,
        sid=sid,
        coords=QRect(0, 0, 50, 40),
        color=QColor(150, 150, 150),
        in_ports=in_ports,
        out_ports=out_ports,
        params=params,
        username=username,
        b_type=b_type,
    )


def _make_line(sid, src, srcport, dst, dstport):
    """Create a DLine."""
    from lib.simulation.connection import DLine
    return DLine(
        sid=sid,
        srcblock=src,
        srcport=srcport,
        dstblock=dst,
        dstport=dstport,
        points=[QPoint(0, 0), QPoint(100, 0)],
    )


def _run_compiled(blocks, lines, t_end=20.0):
    """
    Compile ``blocks``/``lines`` and integrate with RK45.

    Returns ``(sol, state_map, block_matrices)``.
    """
    from lib.engine.system_compiler import SystemCompiler
    from scipy.integrate import solve_ivp

    compiler = SystemCompiler()
    model_func, y0, state_map, block_matrices = compiler.compile_system(
        blocks, blocks, lines
    )
    sol = solve_ivp(
        model_func,
        (0, t_end),
        y0,
        method='RK45',
        t_eval=np.linspace(0, t_end, 2000),
        rtol=1e-8,
        atol=1e-10,
    )
    assert sol.success, f"solve_ivp failed: {sol.message}"
    return sol, state_map, block_matrices


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

@pytest.mark.regression
class TestFeedthroughBug:
    """
    Regression tests for compiled solver pre-population omitting D*u.

    See module docstring for a detailed analysis of which topologies expose
    the bug vs. which pass regardless of the bug.
    """

    def test_pi_controller_closed_loop(self, qapp):
        """
        PI controller (2s+1)/s  [D=2]  +  plant 1/(s+1)  [D=0]  in closed loop.

        Topology: step0 -> sum0(+-) -> tranfn0(PI) -> tranfn1(plant) -> sum0:1

        CL transfer function:
            OL = (2s+1) / (s(s+1))
            CL = OL / (1+OL) = (2s+1) / (s^2 + 3s + 1)
            SS  = 1.0

        NOTE: In this topology the feedback path passes through the D=0 plant
        whose pre-populated signal is already correct; no algebraic block reads
        the D!=0 PI output before it is corrected by the PI executor.  The bug
        therefore does NOT manifest here — this test passes both before and
        after the fix.
        """
        blocks = [
            _make_block('Step',   0, '', 0, 1, {'value': 1.0, 'delay': 0.0, 'type': 'up'}, b_type=0),
            _make_block('Sum',    0, '', 2, 1, {'sign': '+-'}),
            _make_block('TranFn', 0, '', 1, 1, {'numerator': [2.0, 1.0], 'denominator': [1.0, 0.0]}),
            _make_block('TranFn', 1, '', 1, 1, {'numerator': [1.0],       'denominator': [1.0, 1.0]}),
        ]
        lines = [
            _make_line(0, 'step0',   0, 'sum0',    0),
            _make_line(1, 'sum0',    0, 'tranfn0',  0),
            _make_line(2, 'tranfn0', 0, 'tranfn1',  0),
            _make_line(3, 'tranfn1', 0, 'sum0',    1),  # feedback
        ]

        sol, state_map, _ = _run_compiled(blocks, lines, t_end=20.0)

        assert 'tranfn1' in state_map, "plant TranFn should have compiled state"
        start, _ = state_map['tranfn1']
        # D=0 plant: output = C*x = state
        final_value = sol.y[start, -1]

        expected_ss = 1.0
        assert abs(final_value - expected_ss) < 0.05, (
            f"PI+plant CL SS should be ~{expected_ss:.3f}, got {final_value:.4f}"
        )

    def test_lead_compensator_closed_loop(self, qapp):
        """
        Lead compensator (2s+1)/(s+10)  [D=2]  +  plant 1/(s+1)  [D=0].

        Topology: step0 -> sum0(+-) -> tranfn0(lead) -> tranfn1(plant) -> sum0:1

        CL steady-state:
            OL = (2s+1) / ((s+10)(s+1))
            SS = OL(0) = 1/10
            CL SS = (1/10) / (1 + 1/10) = 1/11 ≈ 0.0909

        Same topology argument as test_pi_controller_closed_loop: no algebraic
        block sits between the D!=0 Lead and the D=0 plant, so the bug does not
        manifest in this test.
        """
        blocks = [
            _make_block('Step',   0, '', 0, 1, {'value': 1.0, 'delay': 0.0, 'type': 'up'}, b_type=0),
            _make_block('Sum',    0, '', 2, 1, {'sign': '+-'}),
            _make_block('TranFn', 0, '', 1, 1, {'numerator': [2.0, 1.0], 'denominator': [1.0, 10.0]}),
            _make_block('TranFn', 1, '', 1, 1, {'numerator': [1.0],       'denominator': [1.0,  1.0]}),
        ]
        lines = [
            _make_line(0, 'step0',   0, 'sum0',    0),
            _make_line(1, 'sum0',    0, 'tranfn0',  0),
            _make_line(2, 'tranfn0', 0, 'tranfn1',  0),
            _make_line(3, 'tranfn1', 0, 'sum0',    1),  # feedback
        ]

        sol, state_map, _ = _run_compiled(blocks, lines, t_end=30.0)

        assert 'tranfn1' in state_map
        start, _ = state_map['tranfn1']
        final_value = sol.y[start, -1]

        expected_ss = 1.0 / 11.0   # ≈ 0.0909
        assert abs(final_value - expected_ss) < 0.005, (
            f"Lead+plant CL SS should be ~{expected_ss:.4f}, got {final_value:.4f}"
        )

    def test_pid_block_closed_loop(self, qapp):
        """
        PID(Kp=2, Ki=1, Kd=0) + plant 1/(s+1) in closed loop.

        Topology: step0 -> pid0:0,  tranfn0 -> pid0:1,  pid0 -> tranfn0

        The PID output at steady state satisfies e_ss = 0 (integral action
        forces zero steady-state error), so CL SS = 1.0.

        The PID block uses a custom state executor (not block_matrices), so its
        pre-populated signal is the raw state vector [x_i, x_d].  No algebraic
        block reads from pid0, so the wrong pre-populated value is overwritten
        by the PID executor before tranfn0 reads it.  Bug does not manifest here.
        """
        blocks = [
            _make_block('Step',   0, '', 0, 1, {'value': 1.0, 'delay': 0.0, 'type': 'up'}, b_type=0),
            _make_block('PID',    0, '', 2, 1, {'Kp': 2.0, 'Ki': 1.0, 'Kd': 0.0, 'N': 20.0,
                                                 'u_min': -1e9, 'u_max': 1e9}),
            _make_block('TranFn', 0, '', 1, 1, {'numerator': [1.0], 'denominator': [1.0, 1.0]}),
        ]
        lines = [
            _make_line(0, 'step0',   0, 'pid0',   0),   # setpoint
            _make_line(1, 'tranfn0', 0, 'pid0',   1),   # measurement (feedback)
            _make_line(2, 'pid0',    0, 'tranfn0', 0),
        ]

        sol, state_map, _ = _run_compiled(blocks, lines, t_end=20.0)

        assert 'tranfn0' in state_map
        start, _ = state_map['tranfn0']
        # D=0 plant: output = state
        final_value = sol.y[start, -1]

        expected_ss = 1.0
        assert abs(final_value - expected_ss) < 0.05, (
            f"PID+plant CL SS should be ~{expected_ss:.3f}, got {final_value:.4f}"
        )

    def test_pi_with_saturation_closed_loop(self, qapp):
        """
        PI (2s+1)/s  [D=2]  -> Saturation(-100, 100)  -> plant 1/(s+1)  [D=0].

        Topology:
            step0 -> sum0(+-) -> tranfn0(PI) -> saturation0 -> tranfn1(plant) -> sum0:1

        This is the canonical feedthrough-bug reproducer.  Saturation is an
        *algebraic* block; it runs before any state block in the compiled
        execution sequence.  It reads ``signals['tranfn0']`` which has been
        pre-populated as ``C*x`` only (missing ``D*u``), obtaining the wrong
        value.  Plant receives the wrong Saturation output and integrates an
        incorrect derivative.

        CL analysis (ignoring saturation, which is inactive for SS value):
            OL = (2s+1) / (s(s+1))
            CL SS = 1.0

        EXPECTED: FAIL before the feedthrough fix, PASS after the fix.
        """
        blocks = [
            _make_block('Step',       0, '', 0, 1, {'value': 1.0, 'delay': 0.0, 'type': 'up'}, b_type=0),
            _make_block('Sum',        0, '', 2, 1, {'sign': '+-'}),
            _make_block('TranFn',     0, '', 1, 1, {'numerator': [2.0, 1.0], 'denominator': [1.0, 0.0]}),
            _make_block('Saturation', 0, '', 1, 1, {'min': -100.0, 'max': 100.0}),
            _make_block('TranFn',     1, '', 1, 1, {'numerator': [1.0], 'denominator': [1.0, 1.0]}),
        ]
        lines = [
            _make_line(0, 'step0',        0, 'sum0',        0),
            _make_line(1, 'sum0',         0, 'tranfn0',      0),
            _make_line(2, 'tranfn0',      0, 'saturation0',  0),
            _make_line(3, 'saturation0',  0, 'tranfn1',      0),
            _make_line(4, 'tranfn1',      0, 'sum0',        1),  # feedback
        ]

        sol, state_map, _ = _run_compiled(blocks, lines, t_end=20.0)

        assert 'tranfn1' in state_map
        start, _ = state_map['tranfn1']
        final_value = sol.y[start, -1]

        expected_ss = 1.0
        assert abs(final_value - expected_ss) < 0.05, (
            f"PI+Sat+plant CL SS should be ~{expected_ss:.3f}, got {final_value:.4f} "
            f"(feedthrough bug: Saturation consumed C*x-only pre-pop of PI, dropping D*u)"
        )

    def test_lead_through_gain_closed_loop(self, qapp):
        """
        Lead compensator (2s+1)/(s+10) [D=2] -> Gain(1) -> plant 1/(s+1) [D=0].

        Topology:
            step0 -> sum0(+-) -> tranfn0(lead) -> gain0 -> tranfn1(plant) -> sum0:1

        This is the definitive feedthrough-bug reproducer. Gain is algebraic and
        executes before state blocks. It reads signals['tranfn0'] which has been
        pre-populated as C*x = -19*x_lead, missing D*u = 2*u_lead.

        Correct system: eigenvalues at -12.09, -0.91 (stable), SS = 1/11.
        Buggy system:   eigenvalues at -11.76, +0.76 (UNSTABLE), diverges.

        tf2ss([2,1],[1,10]) = A=[[-10]], B=[[1]], C=[[-19]], D=[[2]]
        """
        blocks = [
            _make_block('Step',   0, '', 0, 1,
                        {'value': 1.0, 'delay': 0.0, 'type': 'up'}, b_type=0),
            _make_block('Sum',    0, '', 2, 1, {'sign': '+-'}),
            _make_block('TranFn', 0, '', 1, 1,
                        {'numerator': [2.0, 1.0], 'denominator': [1.0, 10.0]}),
            _make_block('Gain',   0, '', 1, 1, {'gain': 1.0}),
            _make_block('TranFn', 1, '', 1, 1,
                        {'numerator': [1.0], 'denominator': [1.0, 1.0]}),
        ]

        lines = [
            _make_line(0, 'step0',   0, 'sum0',    0),
            _make_line(1, 'sum0',    0, 'tranfn0', 0),
            _make_line(2, 'tranfn0', 0, 'gain0',   0),
            _make_line(3, 'gain0',   0, 'tranfn1', 0),
            _make_line(4, 'tranfn1', 0, 'sum0',    1),  # feedback
        ]

        sol, state_map, _ = _run_compiled(blocks, lines, t_end=10.0)

        start1, _ = state_map['tranfn1']
        final_output = sol.y[start1, -1]

        expected_ss = 1.0 / 11.0  # ~0.0909
        assert abs(final_output - expected_ss) < 0.01, (
            f"Lead+Gain+plant CL SS should be ~{expected_ss:.4f}, got {final_output:.4f} "
            f"(buggy system diverges due to wrong pre-populated Lead output)"
        )

    def test_strictly_proper_tf_still_works(self, qapp):
        """
        Sanity check: strictly-proper D=0 TF in closed loop must still converge.

        Topology: step0(8) -> sum0(+-) -> tranfn0(1/(s+0.2)) -> sum0:1

        CL: 1/(s+1.2),  SS = 8/1.2 ≈ 6.667.

        Pre-population is exact for D=0 blocks; this test must always pass.
        """
        blocks = [
            _make_block('Step',   0, '', 0, 1, {'value': 8.0, 'delay': 0.0, 'type': 'up'}, b_type=0),
            _make_block('Sum',    0, '', 2, 1, {'sign': '+-'}),
            _make_block('TranFn', 0, '', 1, 1, {'numerator': [1.0], 'denominator': [1.0, 0.2]}),
        ]
        lines = [
            _make_line(0, 'step0',   0, 'sum0',   0),
            _make_line(1, 'sum0',    0, 'tranfn0', 0),
            _make_line(2, 'tranfn0', 0, 'sum0',   1),  # feedback
        ]

        sol, state_map, _ = _run_compiled(blocks, lines, t_end=50.0)

        assert 'tranfn0' in state_map
        start, _ = state_map['tranfn0']
        final_value = sol.y[start, -1]

        expected_ss = 8.0 / 1.2  # ≈ 6.667
        assert abs(final_value - expected_ss) < 0.05, (
            f"D=0 TF CL SS should be ~{expected_ss:.3f}, got {final_value:.4f}"
        )

    def test_compiled_vs_analytical_three_tf_chain(self, qapp):
        """
        Three TFs in series in closed loop:
            tranfn0: (2s+1)/s  [D=2],  tranfn1: 1/(s+1)  [D=0],  tranfn2: 1/s  [D=0]

        Topology:
            step0 -> sum0(+-) -> tranfn0 -> tranfn1 -> tranfn2 -> sum0:1

        CL analysis:
            OL = (2s+1) / (s^2(s+1))
            CL SS = 1.0  (Type-2 system, zero SS error to step input)

        In this topology no algebraic block reads any of the TF outputs;
        all state blocks execute in series and correct one another before
        the next reads.  Bug does not manifest.
        """
        blocks = [
            _make_block('Step',   0, '', 0, 1, {'value': 1.0, 'delay': 0.0, 'type': 'up'}, b_type=0),
            _make_block('Sum',    0, '', 2, 1, {'sign': '+-'}),
            _make_block('TranFn', 0, '', 1, 1, {'numerator': [2.0, 1.0], 'denominator': [1.0, 0.0]}),
            _make_block('TranFn', 1, '', 1, 1, {'numerator': [1.0],       'denominator': [1.0, 1.0]}),
            _make_block('TranFn', 2, '', 1, 1, {'numerator': [1.0],       'denominator': [1.0, 0.0]}),
        ]
        lines = [
            _make_line(0, 'step0',   0, 'sum0',    0),
            _make_line(1, 'sum0',    0, 'tranfn0',  0),
            _make_line(2, 'tranfn0', 0, 'tranfn1',  0),
            _make_line(3, 'tranfn1', 0, 'tranfn2',  0),
            _make_line(4, 'tranfn2', 0, 'sum0',    1),  # feedback
        ]

        sol, state_map, _ = _run_compiled(blocks, lines, t_end=30.0)

        assert 'tranfn2' in state_map
        start, _ = state_map['tranfn2']
        # tranfn2 = 1/s (integrator TF): A=[[0]], B=[[1]], C=[[1]], D=[[0]]
        # output = C*x = state
        final_value = sol.y[start, -1]

        expected_ss = 1.0
        assert abs(final_value - expected_ss) < 0.05, (
            f"3-TF chain CL SS should be ~{expected_ss:.3f}, got {final_value:.4f}"
        )
