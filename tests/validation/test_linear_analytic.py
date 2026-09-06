"""Closed-form linear responses, on both execution paths.

These are the cases with an exact answer written down in any controls text, so
they need no numerical reference at all: a first-order lag and the three damping
regimes of a second-order plant under a unit step, and an integrator tracking a
ramp and a sine. Both paths are exercised, because they are two different
numerical engines (adaptive ``solve_ivp`` over the assembled system versus a
fixed-step loop over the blocks) and either can regress alone.
"""

import numpy as np
import pytest

from tests.validation import _cases as cases

pytestmark = pytest.mark.validation


def test_first_order_lag_step():
    """``K/(tau s + 1)`` under a unit step: ``y = K(1 - e^{-t/tau})``."""
    rows = cases.case_first_order_lag()
    assert len(rows) == 2
    failures = cases.format_failures(rows)
    assert not failures, failures


def test_second_order_step_all_damping_regimes():
    """Under-, critically and over-damped step responses.

    The three regimes are three different closed forms (oscillatory, repeated
    root, two real roots), so a single formula cannot accidentally satisfy all
    of them.
    """
    rows = cases.case_second_order_step()
    assert len(rows) == 2 * len(cases.SECOND_ORDER_ZETAS)
    failures = cases.format_failures(rows)
    assert not failures, failures


def test_integrator_tracks_a_ramp_and_a_sine():
    """``int a s ds = a t^2/2`` and ``int A sin(ws) ds = A(1 - cos wt)/w``."""
    rows = cases.case_integrator_tracking()
    failures = cases.format_failures(rows)
    assert not failures, failures


def test_second_order_reference_matches_its_own_transfer_function():
    """The analytic formulas are checked against the ODE they claim to solve.

    A validation suite whose reference is wrong validates nothing, so the
    closed forms are themselves pinned: substituting the analytic ``y`` into
    ``y'' + 2 zeta wn y' + wn^2 y = wn^2`` must leave a residual at the level of
    the finite-difference truncation, for every damping regime.
    """
    wn = cases.SECOND_ORDER["wn"]
    t = np.linspace(0.0, 5.0, 200001)
    dt = t[1] - t[0]
    for zeta in cases.SECOND_ORDER_ZETAS:
        y = cases.second_order_step(t, zeta, wn)
        d1 = np.gradient(y, dt)
        d2 = np.gradient(d1, dt)
        residual = d2 + 2 * zeta * wn * d1 + wn**2 * y - wn**2
        assert np.max(np.abs(residual[10:-10])) < 1e-4, (
            "closed form for zeta={0} does not satisfy its own ODE".format(zeta)
        )
        # np.gradient falls back to a one-sided difference at the endpoints,
        # so the initial slope is only resolved to O(dt) of this grid.
        assert abs(y[0]) < 1e-12 and abs(d1[0]) < 1e-4
