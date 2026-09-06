"""Transport delay against the analytically shifted signal.

``TransportDelay`` claims ``y(t) = u(t - tau)``, with the initial value held
until the buffer fills. Feeding it a sine makes both halves of that claim
checkable in closed form, and running it with ``tau`` on and off the output grid
separates buffer replay from interpolation.
"""

import numpy as np
import pytest

from tests.validation import _cases as cases
from tests.validation import _harness as H

pytestmark = pytest.mark.validation


def test_transport_delay_shifts_a_sine():
    rows = cases.case_transport_delay()
    failures = cases.format_failures(rows)
    assert not failures, failures


def test_off_grid_delay_error_is_second_order_in_dt():
    """Halving the step must shrink the interpolation error by about four.

    A nearest-sample lookup would be first order and would not survive this;
    the check distinguishes a delay that interpolates from one that rounds.
    """
    coarse, _ = cases._run_transport_delay(0.333, 0.005)
    fine, _ = cases._run_transport_delay(0.333, 0.0025)
    order = H.observed_order([coarse, fine])[0]
    assert order > 1.7, "observed order {0:.2f}; interpolation looks first order".format(order)
    assert np.isfinite(order)
