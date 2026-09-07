"""A PID loop assembled from blocks against its analytic closed-loop response.

This is the first case where the diagram's *structure* carries meaning: the
controller, the plant and the feedback path are three separate blocks, and the
reference is the single rational function ``CP/(1+CP)`` those three imply. The
compiled path has to realise the documented controller
``C(s) = Kp + Ki/s + Kd N s/(s+N)`` and close the loop exactly for the two to
agree.

The interpreter realises the same controller as a fixed-step difference
equation, so it is first order in ``dt`` rather than exact; what is checked
there is that refining ``dt`` actually drives it onto the same analytic loop.
"""

import numpy as np
import pytest

from tests.validation import _cases as cases
from tests.validation import _harness as H

pytestmark = pytest.mark.validation


def test_pid_closed_loop_matches_the_analytic_transfer_function():
    rows = cases.case_pid_closed_loop()
    failures = cases.format_failures(rows)
    assert not failures, failures


def test_closed_loop_reference_is_stable_and_tracks_the_step():
    """Sanity on the reference: this loop settles at the setpoint.

    A closed loop that did not converge to 1 would make the comparison above
    meaningless -- both sides could be wrong in the same way.
    """
    num, den = cases.pid_closed_loop_tf(
        cases.PID_GAINS[0],
        cases.PID_GAINS[1],
        cases.PID_GAINS[2],
        cases.PID_GAINS[3],
        cases.PID_PLANT_NUM,
        cases.PID_PLANT_DEN,
    )
    assert np.max(np.real(np.roots(den))) < 0.0, "closed loop is not stable"
    # DC gain of CP/(1+CP) is 1 whenever the controller has integral action.
    assert num[-1] / den[-1] == pytest.approx(1.0, abs=1e-12)


def test_interpreted_pid_loop_converges_to_the_analytic_closed_loop():
    """Refining ``dt`` must drive the interpreted loop onto ``CP/(1+CP)``.

    The controller's derivative branch is the part of the loop a fixed-step
    engine can get structurally wrong: filtering the finite difference of the
    error and seeding it from the first sample makes ``de = 0`` at ``t0``, which
    silently deletes the derivative's whole response to the reference step. That
    is an O(1) error, invisible at any single step size and visible immediately
    here -- it does not shrink under refinement. Both refinements below must
    roughly halve the error, which is what first order means.
    """
    steps = (0.008, 0.004, 0.002, 0.001)
    errors = cases.pid_loop_errors(steps)
    orders = H.observed_order(errors)

    assert all(fine < coarse for coarse, fine in zip(errors[:-1], errors[1:])), errors
    assert min(orders) > 0.8, "interpreted PID loop errors {0} give orders {1}".format(
        ["{0:.2e}".format(e) for e in errors], ["{0:.2f}".format(p) for p in orders]
    )


def test_the_derivative_branch_responds_to_a_step():
    """``Kd`` alone must produce a response, and the same one on both paths.

    With ``Kp = Ki = 0`` the loop is driven entirely by ``Kd N s/(s+N)`` acting
    on the reference step, so a derivative branch that never sees the step gives
    an identically zero output. The compiled kernel integrates the filter state
    from zero and is the reference here.
    """
    gains = (0.0, 0.0, cases.PID_GAINS[2], cases.PID_GAINS[3])
    peaks = []
    for compiled in (True, False):
        builder = cases.build_pid_loop(
            gains,
            cases.PID_PLANT_NUM,
            cases.PID_PLANT_DEN,
            cases.PID_LOOP["sim_time"],
            cases.PID_LOOP["sim_dt"],
        )
        result = cases.H.run(builder, compiled=compiled)
        _t, y = result.signal("y")
        peaks.append(float(np.max(np.abs(y))))
        cases.H.release(result)

    assert peaks[0] > 0.3, "compiled derivative-only loop is flat: {0}".format(peaks[0])
    assert peaks[1] == pytest.approx(peaks[0], rel=0.05), (
        "interpreted derivative-only peak {1:.4f} does not match the compiled {0:.4f}".format(
            *peaks
        )
    )
