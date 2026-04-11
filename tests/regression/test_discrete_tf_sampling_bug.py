"""
Regression test for the discrete-time transfer function + sampling bug.

BUG DESCRIPTION
---------------
Strictly-proper ``DiscreteTranFn`` blocks (``len(den) > len(num)``, e.g.
``H(z) = 1/(z-0.5)``) with an explicit ``sampling_time > 0`` produced an
all-zero scope output.

In ``execution_loop`` / ``execution_loop_headless`` the memory-block
first-pass loop ran the block with ``output_only=True`` (which deliberately
skips the state update) and *then* called ``schedule_next_execution``,
advancing ``_next_execution_time`` past the current step.  When the
hierarchy loop later asked ``should_execute()``, it returned ``False``, so
the state-updating execute call was skipped forever.  The internal state
``x`` stayed at its initial value, ``y = C @ x`` stayed at zero, and every
downstream sample was zero.

A second, related issue: the memory-block first-pass skip branch excluded
``b_type==1`` from held-output propagation, so strictly-proper discrete
blocks dropped their held output between samples.

THE FIX
-------
- Stop calling ``schedule_next_execution`` (and the redundant
  ``set_held_output``) in the memory-block first-pass loop.  The hierarchy
  loop's state-updating execute is the canonical place for both.
- Allow ``b_type==1`` blocks to propagate held outputs through the
  first-pass skip branch (only sinks, ``b_type==3``, stay excluded).
"""

import pytest
import numpy as np
from PyQt5.QtCore import QPoint


def _build_step_dtf_scope(num, den, sampling_time, sim_time=1.0, sim_dt=0.1):
    """Wire Step -> DiscreteTranFn -> Scope and run the headless loop."""
    from lib.lib import DSim

    dsim = DSim()
    # The headless harness pokes ``buttons_list[6].active`` after init —
    # provide a stub so we can run without the GUI buttons attached.
    dsim.buttons_list = [type('B', (), {'active': False})() for _ in range(20)]

    menu_by_fn = {b.fn_name: b for b in dsim.menu_blocks}
    step = dsim.add_block(menu_by_fn['step'], QPoint(100, 100))
    dtf = dsim.add_block(menu_by_fn['discrete_transfer_function'], QPoint(300, 100))
    scope = dsim.add_block(menu_by_fn['scope'], QPoint(500, 100))

    dtf.params['numerator'] = num
    dtf.params['denominator'] = den
    dtf.params['sampling_time'] = sampling_time

    dsim.add_line((step.name, 0, step.out_coords[0]),
                  (dtf.name, 0, dtf.in_coords[0]))
    dsim.add_line((dtf.name, 0, dtf.out_coords[0]),
                  (scope.name, 0, scope.in_coords[0]))

    dsim.sim_time = sim_time
    dsim.sim_dt = sim_dt
    dsim.plot_trange = sim_time
    dsim.execution_init_time = lambda: dsim.sim_time

    assert dsim.execution_init() is True
    while dsim.time_step <= dsim.sim_time:
        dsim.execution_loop_headless()

    return dtf, np.asarray(scope.exec_params['vector']).flatten()


@pytest.mark.regression
class TestDiscreteTfSamplingBug:
    def test_strictly_proper_with_sampling_advances_state(self, qapp):
        """``H(z) = 1/(z-0.5)`` with Ts=0.2 must produce a non-zero step response."""
        dtf, vec = _build_step_dtf_scope(
            num=[1.0], den=[1.0, -0.5], sampling_time=0.2,
        )

        # The state must actually advance — bug symptom was x stuck at 0.
        x_final = np.asarray(dtf.exec_params['_x_']).flatten()
        assert np.any(np.abs(x_final) > 1e-9), (
            f"Discrete TF state never advanced: x={x_final}"
        )

        # The scope vector must contain non-zero samples — bug symptom was
        # ``[0, 0, 0, ...]`` for the entire run.
        assert np.any(np.abs(vec) > 1e-9), f"Scope output is all zeros: {vec}"

        # Sanity-check the converged value.  H(1) = 1/(1-0.5) = 2 → step
        # response asymptotes to 2.
        assert vec[-1] == pytest.approx(2.0, abs=0.2), (
            f"Step response final value should approach 2.0, got {vec[-1]}"
        )

    def test_strictly_proper_continuous_rate_still_works(self, qapp):
        """Regression guard: the no-sampling path must remain correct.

        Also pins the off-by-one fix added alongside the sampling fix:
        ``vec[1]`` used to be ``0`` (a duplicate of the initial sample) because
        memory blocks never had their state advanced during init.  After the
        fix, ``vec[0]=0`` (initial output) and ``vec[1]=1`` (state after one
        update with u=1).
        """
        _, vec = _build_step_dtf_scope(
            num=[1.0], den=[1.0, -0.5], sampling_time=-1.0,
        )
        # Initial output is zero (zero state, D=0).
        assert vec[0] == pytest.approx(0.0, abs=1e-9)
        # First sample after init must be y[1] = C @ x[1] = B = 1 — not the
        # buggy duplicate zero.
        assert vec[1] == pytest.approx(1.0, abs=1e-9)
        # Step response asymptotes to 2.
        assert vec[-1] == pytest.approx(2.0, abs=0.05)

    def test_proper_with_sampling_unchanged(self, qapp):
        """Regression guard: the proper (D!=0) sampling path was already
        correct and must stay that way."""
        _, vec = _build_step_dtf_scope(
            num=[1.0, 0.0], den=[1.0, -0.5], sampling_time=0.2,
        )
        # First held value is the D-feedthrough at t=0.
        assert vec[0] == pytest.approx(1.0, abs=1e-9)
        assert vec[-1] == pytest.approx(2.0, abs=0.2)
