"""Convergence order of the interpreter's fixed-step integration methods.

An error tolerance at one step size says a run was accurate; a convergence
order says *why*, and is the check that notices when a method silently stops
being the method it is named after. Integrating ``sin(wt)`` at dt, dt/2 and dt/4
must shrink the error by 2 for explicit Euler and by 16 for classical RK4.

This is an interpreter-only property: the compiled path folds the diagram into
one system and hands it to an adaptive solver, so a per-block method has no
meaning there (the compiler logs a warning and integrates with the method from
Simulation settings).
"""

import numpy as np
import pytest

from tests.validation import _cases as cases

pytestmark = pytest.mark.validation


def test_observed_orders_match_the_named_methods():
    rows = cases.case_integration_order()
    failures = cases.format_failures(rows)
    assert not failures, failures


@pytest.mark.parametrize("method,expected", [("FWD_EULER", 1.0), ("RK4", 4.0)])
def test_errors_shrink_monotonically_with_the_step(method, expected):
    """Refinement must actually reduce the error, not just look like an order.

    ``observed_order`` is a ratio of two errors and would report a clean number
    for a sequence that happened to scale while staying large, so the errors
    themselves are pinned too: each refinement strictly improves, and the finest
    step lands where the method's order predicts.
    """
    errors, orders = cases.integrator_order(method)
    assert all(fine < coarse for coarse, fine in zip(errors[:-1], errors[1:])), errors
    predicted = errors[0] / (2.0**expected) ** (len(errors) - 1)
    assert errors[-1] == pytest.approx(predicted, rel=0.25), (
        "{0}: errors {1} do not follow order {2}".format(method, errors, expected)
    )
    assert np.all(np.isfinite(orders))
