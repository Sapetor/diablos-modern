"""
Regression test for a bug where ``SystemCompiler`` reads ``block.params``
directly when classifying state blocks (TransferFcn, StateSpace) as
D=0 vs D!=0 for execution ordering.

When the numerator/denominator of a TransferFcn is parameterised by a
workspace variable (i.e. ``block.params['numerator'] = 'K_num'``, a string),
the raw ``block.params`` still holds the unresolved string.  The compiler
must fall back to ``block.exec_params`` (populated upstream by
``WorkspaceManager.resolve_params``) — which is the same pattern that the
PDE branches in the compiler already follow (see ``system_compiler.py``
near lines 1808-1904).

Before the fix, the TransferFcn branch crashes at
``np.array(num, dtype=float)`` / ``signal.tf2ss(num, den)`` when
``num``/``den`` are unresolved strings, so a closed-loop diagram that
uses workspace-variable-parameterised TFs simply refuses to compile.

After the fix, ``block.exec_params`` is consulted first so the compile
succeeds, the block is correctly classified (D!=0 for a proper-but-not-
strictly-proper TF), and the compiled and interpreted paths agree.
"""

import pytest
import numpy as np
from PyQt5.QtCore import QRect, QPoint
from PyQt5.QtGui import QColor


def _make_block(block_fn, sid, username, in_ports, out_ports, params, b_type=2):
    """Create a DBlock with minimal boilerplate (mirrors test_feedthrough_bug)."""
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
    from lib.simulation.connection import DLine
    return DLine(
        sid=sid,
        srcblock=src,
        srcport=srcport,
        dstblock=dst,
        dstport=dstport,
        points=[QPoint(0, 0), QPoint(100, 0)],
    )


@pytest.fixture
def workspace_with_pi_coeffs():
    """
    Install workspace variables used by the test TFs and restore the
    WorkspaceManager singleton afterwards.

    ``pi_num = [2.0, 1.0]``  → PI numerator (D=2 when paired with denom [1, 0])
    ``pi_den = [1.0, 0.0]``  → integrator denominator
    ``plant_num = [1.0]``
    ``plant_den = [1.0, 1.0]``
    """
    from lib.workspace import WorkspaceManager

    # Save existing singleton state so we don't pollute other tests.
    prev_instance = WorkspaceManager._instance
    WorkspaceManager._instance = None

    wm = WorkspaceManager()
    wm.variables = {
        'pi_num':    [2.0, 1.0],
        'pi_den':    [1.0, 0.0],
        'plant_num': [1.0],
        'plant_den': [1.0, 1.0],
    }

    yield wm

    # Restore
    WorkspaceManager._instance = prev_instance


def _compile_and_solve(blocks, lines, t_end=20.0):
    """Resolve exec_params (mirroring SimulationEngine), then compile and solve."""
    from lib.engine.system_compiler import SystemCompiler
    from lib.workspace import WorkspaceManager
    from scipy.integrate import solve_ivp

    wm = WorkspaceManager()
    dt = 0.01
    for block in blocks:
        # Same logic as SimulationEngine.run_compiled_simulation (line 1020-1025)
        block.exec_params = wm.resolve_params(block.params)
        block.exec_params.update(
            {k: v for k, v in block.params.items() if k.startswith('_')}
        )
        block.exec_params['dtime'] = dt

    compiler = SystemCompiler()
    model_func, y0, state_map, block_matrices = compiler.compile_system(
        blocks, blocks, lines
    )
    sol = solve_ivp(
        model_func, (0, t_end), y0, method='RK45',
        t_eval=np.linspace(0, t_end, 2000), rtol=1e-8, atol=1e-10,
    )
    assert sol.success, f"solve_ivp failed: {sol.message}"
    return sol, state_map, block_matrices


@pytest.mark.unit
class TestCompilerWorkspaceVars:
    """
    Covers TransferFcn (and by extension StateSpace) parameterised by
    workspace variables.  The compiler must resolve them via
    ``block.exec_params`` — not read raw ``block.params`` — during the
    state-identification / D-computation pass.
    """

    def test_pi_feedback_with_workspace_vars_matches_literals(
        self, qapp, workspace_with_pi_coeffs
    ):
        """
        Closed loop: step -> sum(+-) -> PI -> plant -> feedback.

        Two equivalent systems — one with literal coefficients, one with
        workspace variables — must produce equal trajectories (the compile
        path must resolve the variables to the same numbers and classify
        the PI block as D!=0 either way).
        """
        # Literal version
        blocks_lit = [
            _make_block('Step',   0, '', 0, 1,
                        {'value': 1.0, 'delay': 0.0, 'type': 'up'}, b_type=0),
            _make_block('Sum',    0, '', 2, 1, {'sign': '+-'}),
            _make_block('TranFn', 0, '', 1, 1,
                        {'numerator': [2.0, 1.0], 'denominator': [1.0, 0.0]}),
            _make_block('TranFn', 1, '', 1, 1,
                        {'numerator': [1.0],       'denominator': [1.0, 1.0]}),
        ]
        lines_lit = [
            _make_line(0, 'step0',   0, 'sum0',    0),
            _make_line(1, 'sum0',    0, 'tranfn0', 0),
            _make_line(2, 'tranfn0', 0, 'tranfn1', 0),
            _make_line(3, 'tranfn1', 0, 'sum0',    1),
        ]

        sol_lit, state_map_lit, _ = _compile_and_solve(blocks_lit, lines_lit)

        # Workspace-variable version
        blocks_var = [
            _make_block('Step',   0, '', 0, 1,
                        {'value': 1.0, 'delay': 0.0, 'type': 'up'}, b_type=0),
            _make_block('Sum',    0, '', 2, 1, {'sign': '+-'}),
            _make_block('TranFn', 0, '', 1, 1,
                        {'numerator': 'pi_num', 'denominator': 'pi_den'}),
            _make_block('TranFn', 1, '', 1, 1,
                        {'numerator': 'plant_num', 'denominator': 'plant_den'}),
        ]
        lines_var = [
            _make_line(0, 'step0',   0, 'sum0',    0),
            _make_line(1, 'sum0',    0, 'tranfn0', 0),
            _make_line(2, 'tranfn0', 0, 'tranfn1', 0),
            _make_line(3, 'tranfn1', 0, 'sum0',    1),
        ]

        sol_var, state_map_var, _ = _compile_and_solve(blocks_var, lines_var)

        # The plant output trajectory (state of tranfn1) must match.
        start_lit, _ = state_map_lit['tranfn1']
        start_var, _ = state_map_var['tranfn1']
        plant_lit = sol_lit.y[start_lit, :]
        plant_var = sol_var.y[start_var, :]

        # Expected steady state: 1.0 (type-1 CL with integral action).
        assert abs(plant_lit[-1] - 1.0) < 0.05
        np.testing.assert_allclose(
            plant_var, plant_lit, rtol=1e-6, atol=1e-8,
            err_msg=(
                "Workspace-var TF must produce identical trajectory to "
                "literal-coefficient TF; differing means the compiler read "
                "raw params and misclassified / misbuilt the PI block."
            ),
        )

    def test_pi_with_saturation_workspace_vars_matches_literals(
        self, qapp, workspace_with_pi_coeffs
    ):
        """
        Canonical feedthrough-bug topology (PI -> Saturation -> plant)
        with workspace-variable coefficients.

        The Saturation block is algebraic and reads the PI output before
        the PI state executor runs.  If the compiler misclassifies the PI
        (as D=0) because it read raw params, the pre-populated ``C*x``
        misses ``D*u`` and the trajectory diverges from the literal version.
        """
        blocks_lit = [
            _make_block('Step',       0, '', 0, 1,
                        {'value': 1.0, 'delay': 0.0, 'type': 'up'}, b_type=0),
            _make_block('Sum',        0, '', 2, 1, {'sign': '+-'}),
            _make_block('TranFn',     0, '', 1, 1,
                        {'numerator': [2.0, 1.0], 'denominator': [1.0, 0.0]}),
            _make_block('Saturation', 0, '', 1, 1, {'min': -100.0, 'max': 100.0}),
            _make_block('TranFn',     1, '', 1, 1,
                        {'numerator': [1.0], 'denominator': [1.0, 1.0]}),
        ]
        lines_lit = [
            _make_line(0, 'step0',       0, 'sum0',        0),
            _make_line(1, 'sum0',        0, 'tranfn0',     0),
            _make_line(2, 'tranfn0',     0, 'saturation0', 0),
            _make_line(3, 'saturation0', 0, 'tranfn1',     0),
            _make_line(4, 'tranfn1',     0, 'sum0',        1),
        ]
        sol_lit, state_map_lit, _ = _compile_and_solve(blocks_lit, lines_lit)

        blocks_var = [
            _make_block('Step',       0, '', 0, 1,
                        {'value': 1.0, 'delay': 0.0, 'type': 'up'}, b_type=0),
            _make_block('Sum',        0, '', 2, 1, {'sign': '+-'}),
            _make_block('TranFn',     0, '', 1, 1,
                        {'numerator': 'pi_num', 'denominator': 'pi_den'}),
            _make_block('Saturation', 0, '', 1, 1, {'min': -100.0, 'max': 100.0}),
            _make_block('TranFn',     1, '', 1, 1,
                        {'numerator': 'plant_num', 'denominator': 'plant_den'}),
        ]
        lines_var = [
            _make_line(0, 'step0',       0, 'sum0',        0),
            _make_line(1, 'sum0',        0, 'tranfn0',     0),
            _make_line(2, 'tranfn0',     0, 'saturation0', 0),
            _make_line(3, 'saturation0', 0, 'tranfn1',     0),
            _make_line(4, 'tranfn1',     0, 'sum0',        1),
        ]
        sol_var, state_map_var, _ = _compile_and_solve(blocks_var, lines_var)

        start_lit, _ = state_map_lit['tranfn1']
        start_var, _ = state_map_var['tranfn1']
        plant_lit = sol_lit.y[start_lit, :]
        plant_var = sol_var.y[start_var, :]

        assert abs(plant_lit[-1] - 1.0) < 0.05
        np.testing.assert_allclose(
            plant_var, plant_lit, rtol=1e-6, atol=1e-8,
            err_msg=(
                "Workspace-var TF in PI+Sat feedback loop diverged from "
                "literal-coefficient version (compiler misread raw params)."
            ),
        )


@pytest.fixture
def workspace_with_hyst_high():
    """Install a workspace variable for the Hysteresis high output and restore."""
    from lib.workspace import WorkspaceManager

    prev_instance = WorkspaceManager._instance
    WorkspaceManager._instance = None

    wm = WorkspaceManager()
    wm.variables = {'hyst_high': 2.0}

    yield wm

    WorkspaceManager._instance = prev_instance


@pytest.mark.unit
class TestCompilerAlgebraicBlockWorkspaceVars:
    """Covers the *executor* branches of ``_create_block_executor`` (algebraic
    blocks: Hysteresis/Selector/Field*), which historically read raw
    ``block.params`` instead of the resolved ``params`` local.  A workspace-
    variable-valued numeric param crashes the compile (``float('hyst_high')``)
    unless the resolved ``exec_params`` are consulted."""

    def test_hysteresis_workspace_var_high_matches_literal(
        self, qapp, workspace_with_hyst_high
    ):
        """Constant -> Hysteresis -> Integrator.

        The Hysteresis ``high`` output is supplied as a workspace variable in
        one diagram and as a literal in the other.  Constant input (1.0) is
        above ``upper`` so the block latches ``high``; the Integrator turns
        that into a ``high * t`` ramp whose slope reveals the resolved value.
        Pre-fix, the workspace-variable diagram raises ``ValueError`` inside
        ``_create_block_executor`` (``float('hyst_high')``) and never compiles.
        """
        def _diagram(high_param):
            blocks = [
                _make_block('Constant', 0, '', 0, 1, {'value': 1.0}, b_type=0),
                _make_block('Hysteresis', 0, '', 1, 1,
                            {'upper': 0.5, 'lower': -0.5,
                             'high': high_param, 'low': 0.0}),
                _make_block('Integrator', 0, '', 1, 1, {'init_conds': 0.0}),
            ]
            lines = [
                _make_line(0, 'constant0',   0, 'hysteresis0', 0),
                _make_line(1, 'hysteresis0', 0, 'integrator0', 0),
            ]
            return blocks, lines

        blocks_lit, lines_lit = _diagram(2.0)
        sol_lit, state_map_lit, _ = _compile_and_solve(blocks_lit, lines_lit, t_end=5.0)

        blocks_var, lines_var = _diagram('hyst_high')
        sol_var, state_map_var, _ = _compile_and_solve(blocks_var, lines_var, t_end=5.0)

        start_lit, _ = state_map_lit['integrator0']
        start_var, _ = state_map_var['integrator0']
        ramp_lit = sol_lit.y[start_lit, :]
        ramp_var = sol_var.y[start_var, :]

        # Slope is the resolved 'high' value (2.0): state(5.0) ~= 10.0.
        assert abs(ramp_lit[-1] - 10.0) < 0.1
        np.testing.assert_allclose(
            ramp_var, ramp_lit, rtol=1e-6, atol=1e-8,
            err_msg=(
                "Hysteresis 'high' as a workspace variable must produce the "
                "same trajectory as the literal value; differing (or a compile "
                "crash) means _create_block_executor read raw block.params."
            ),
        )
