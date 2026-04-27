"""
Regression tests for the interpreter-path closed-loop feedback bug.

Background
----------
A diagram that uses any non-compilable block (e.g. Demux, which is
intentionally absent from `SystemCompiler.COMPILABLE_BLOCKS`) falls through
to the Standard (Interpreter) execution path. Two related bugs in that
path froze the state of any closed-loop system:

(1) `SimulationEngine.initialize_execution()` Loop 1 force-executed every
    block with `b_type == 0`, regardless of whether it was actually a
    source. Algebraic blocks like `Sum` got pinned to `hierarchy=0` with
    `computed_data=True`, so init Loop 2 never re-derived their hierarchy.
    At runtime, Sum/MatrixGain ran at hier=0 in cascade order — fragile,
    and any "wrong" block-list order would silently break the cascade.

(2) `lib.lib.DSim.execution_loop` and `execution_loop_headless` did a
    single pass per hier level. Within a level, two blocks with an
    intra-level data dependency could be ordered such that the consumer
    was checked before its producer fired. Across levels, memory blocks
    pinned to hier=0 needed to fire their state-update execute *after*
    inputs produced at hier>0 arrived — but the for-hier loop never
    returned to lower levels, so the state-update never ran.

Together these conspired to make `examples/c06_lqr_state_feedback.diablos`
appear nearly static. The example was reordered around the bug
(commit d1833d8) so the LQR demo would show its dynamics in interpreter
mode; this test verifies that ANY valid ordering produces correct
dynamics.

Bug signature: scope vector for the state goes from initial conditions
[2, 1] to [2.01, 0.985] over 12 s — exactly one init-time forward-Euler
step under u=0, then frozen. Correct dynamics under LQR feedback drive
x toward 0 (analytical x(12) ~ [-0.032, 0.014]).
"""

import sys
from pathlib import Path

import numpy as np
import pytest


# Resolve repo root for the example file
EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"
C06_EXAMPLE = EXAMPLES_DIR / "c06_lqr_state_feedback.diablos"


def _load_and_swap(qapp, swap_sum_before_matrixgain: bool):
    """
    Load c06_lqr_state_feedback in-memory and optionally reorder its
    blocks_list so Sum precedes MatrixGain (the order that USED to
    produce a frozen state).

    Returns the configured DSim instance, ready for run_tuning_simulation.
    """
    from lib.lib import DSim

    dsim = DSim()
    data = dsim.file_service.load(filepath=str(C06_EXAMPLE))
    assert data is not None, f"Failed to load {C06_EXAMPLE}"
    sim_params = dsim.file_service.apply_loaded_data(data)

    blocks = dsim.blocks_list
    sum_idx = next(i for i, b in enumerate(blocks) if b.block_fn == 'Sum')
    mg_idx = next(i for i, b in enumerate(blocks) if b.block_fn == 'MatrixGain')

    if swap_sum_before_matrixgain and sum_idx > mg_idx:
        blocks[sum_idx], blocks[mg_idx] = blocks[mg_idx], blocks[sum_idx]

    # The committed example uses Demux (not in COMPILABLE_BLOCKS), so the
    # interpreter path is the natural choice. Force-disable the fast solver
    # so the test stays exercising the interpreter path even if someone
    # later adds Demux to COMPILABLE_BLOCKS.
    dsim.use_fast_solver = False
    return dsim, sim_params


def _state_scope_vector(dsim):
    """
    Return the (n_samples, vec_dim) ndarray captured by the scope wired
    to the state demux outputs. c06 has two scopes; the one with vec_dim
    matching the StateSpace state dimension (2) is the "state-scope".
    """
    candidates = []
    for b in dsim.engine.active_blocks_list:
        if b.block_fn != 'Scope':
            continue
        params = getattr(b, 'exec_params', b.params)
        vec = params.get('vector')
        vec_dim = params.get('vec_dim')
        if vec is None or not vec_dim:
            continue
        arr = np.asarray(vec).reshape(-1, vec_dim)
        candidates.append((b.name, vec_dim, arr))

    state_scopes = [c for c in candidates if c[1] == 2]
    assert state_scopes, (
        f"Expected at least one Scope with vec_dim=2 (state-scope); "
        f"got {[(n, d) for n, d, _ in candidates]}"
    )
    return state_scopes[0][2]


@pytest.mark.regression
class TestInterpreterFeedback:
    """LQR closed-loop must produce dynamic state evolution in interpreter mode."""

    def _assert_state_evolves(self, dsim):
        """Shared assertion: x1 must move at least 1 unit from its initial value."""
        scope = _state_scope_vector(dsim)
        assert scope.shape[0] > 100, f"Too few samples: {scope.shape}"

        x1_initial = scope[0, 0]
        x1_max_excursion = float(np.max(np.abs(scope[:, 0] - x1_initial)))

        # Bug signature: x1 ~ 2 throughout (frozen at init). Correct
        # behavior: LQR drives x1 from 2 toward 0 with a slight overshoot,
        # so |x1 - 2| reaches order-of-1.
        assert x1_max_excursion >= 1.0, (
            f"State did not evolve. |x1 - x1[0]| max = {x1_max_excursion:.4f} "
            f"(expected >= 1.0). Trajectory frozen at {scope[0]} -> {scope[-1]} — "
            f"this is the interpreter feedback bug signature."
        )

    def test_good_block_order_still_works(self, qapp):
        """
        The c06 example as committed (MatrixGain before Sum) must continue to
        produce a dynamic LQR response. Guards against regressions that would
        re-break this example.
        """
        dsim, sim_params = _load_and_swap(qapp, swap_sum_before_matrixgain=False)
        success, err = dsim.run_tuning_simulation(
            sim_params['sim_time'], sim_params['sim_dt']
        )
        assert success, f"Simulation failed: {err}"
        self._assert_state_evolves(dsim)

    def test_swapped_block_order_now_works(self, qapp):
        """
        With Sum declared BEFORE MatrixGain in blocks_list (the order that
        USED to fail because the single-pass hier loop walked consumer
        before producer), the simulation must still drive the state to ~0.

        This is the canonical regression test for the interpreter feedback
        bug — without the fix, x1 freezes at ~2.01 and this assertion fails.
        """
        dsim, sim_params = _load_and_swap(qapp, swap_sum_before_matrixgain=True)

        # Sanity: confirm the swap took effect
        names = [b.block_fn for b in dsim.blocks_list]
        sum_idx = names.index('Sum')
        mg_idx = names.index('MatrixGain')
        assert sum_idx < mg_idx, (
            f"Test setup error — expected Sum before MatrixGain: {names}"
        )

        success, err = dsim.run_tuning_simulation(
            sim_params['sim_time'], sim_params['sim_dt']
        )
        assert success, f"Simulation failed: {err}"
        self._assert_state_evolves(dsim)

    def test_state_converges_toward_zero(self, qapp):
        """
        Tighter check: the LQR-tuned closed loop should drive x1 close to
        zero by t=12s. Catches regressions that let the state drift but
        not converge (e.g. a wrong-sign feedback that happens to move x1
        away from initial conditions).
        """
        dsim, sim_params = _load_and_swap(qapp, swap_sum_before_matrixgain=True)
        success, err = dsim.run_tuning_simulation(
            sim_params['sim_time'], sim_params['sim_dt']
        )
        assert success, f"Simulation failed: {err}"

        scope = _state_scope_vector(dsim)
        x1_final = float(scope[-1, 0])
        # Analytical solve gives x1(12) ~= -0.032; allow generous slack
        # for forward-Euler discretization error.
        assert abs(x1_final) < 0.5, (
            f"x1(t=12) = {x1_final:.4f} — expected close to 0 under LQR feedback"
        )
