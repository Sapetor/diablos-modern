"""A PID loop assembled from blocks against its analytic closed-loop response.

This is the first case where the diagram's *structure* carries meaning: the
controller, the plant and the feedback path are three separate blocks, and the
reference is the single rational function ``CP/(1+CP)`` those three imply. The
compiled path has to realise the documented controller
``C(s) = Kp + Ki/s + Kd N s/(s+N)`` and close the loop exactly for the two to
agree.
"""

import numpy as np
import pytest

from tests.validation import _cases as cases

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
