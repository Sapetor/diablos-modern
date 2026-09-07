"""Discontinuities whose switching instant is known in closed form.

Each diagram drives a discontinuous block with a unit-rate ramp, so the
switching surface is crossed at a time that can be written down: a Saturation
limit of 0.7 is reached at t = 0.7, a Switch threshold of 0.5 at t = 0.5.
Downstream of the discontinuity sits an integrator, which turns a mislocated
switch into a visible, accumulating trajectory error rather than a one-sample
glitch.

The contrast between the two paths is the point: with zero-crossing detection
the compiled path locates the instant to machine precision and follows the
analytic trajectory to ~1e-8, while the fixed-step interpreter smears the switch
across whichever step straddles it and lands three orders worse at the same dt.

Locating an instant is only half of it: the solver has to be able to carry on
afterwards. A crossing the trajectory passes through once must produce exactly
one event, and a run with two corners must locate both -- otherwise the
chattering guard gives up on a system that never chattered and everything after
the first switch is back to step accuracy.
"""

import numpy as np
import pytest

from tests.validation import _cases as cases
from tests.validation import _harness as H

pytestmark = pytest.mark.validation


def test_saturation_corner_is_located_exactly():
    rows = cases.case_saturation_event()
    failures = cases.format_failures(rows)
    assert not failures, failures

    compiled = [r for r in rows if r.path == "compiled" and "trajectory" in r.name][0]
    interpreted = [r for r in rows if r.path == "interpreter"][0]
    assert compiled.error < interpreted.error / 100.0, (
        "event detection should beat the smeared fixed-step switch by orders, "
        "got {0:.2e} vs {1:.2e}".format(compiled.error, interpreted.error)
    )


def test_switch_threshold_is_located_exactly():
    rows = cases.case_switch_event()
    failures = cases.format_failures(rows)
    assert not failures, failures


def test_located_instant_does_not_depend_on_the_output_step():
    """The switching time is a property of the system, not of the grid.

    Two output steps, neither a divisor of the other, must locate the same
    corner. Without event detection the switch lands wherever the step happened
    to fall and the two answers separate.
    """
    limit = cases.SATURATION["limit"]
    located = []
    for dt in (0.02, 0.003):
        builder = cases.build_saturating_ramp(limit, cases.SATURATION["sim_time"], dt)
        result = H.run(builder, compiled=True, zero_crossing=True)
        located.append(cases.first_event_time(result))
        H.release(result)
    assert located[0] == pytest.approx(limit, abs=1e-9)
    assert located[1] == pytest.approx(limit, abs=1e-9)


def test_saturation_trajectory_is_flat_after_the_corner():
    """Past the limit the accumulated signal is a straight line of slope 0.7.

    A structural check the scalar error cannot make: the second derivative of
    the trajectory must vanish after the corner, which is what "the clipped
    signal really is constant" means.
    """
    cfg = cases.SATURATION
    builder = cases.build_saturating_ramp(cfg["limit"], cfg["sim_time"], cfg["sim_dt"])
    result = H.run(builder, compiled=True, zero_crossing=True)
    t, z = result.signal("z")
    after = t > cfg["limit"] + 5 * cfg["sim_dt"]
    slope = np.diff(z[after]) / np.diff(t[after])
    assert np.allclose(slope, cfg["limit"], atol=1e-7)
    H.release(result)


def test_a_single_state_crossing_does_not_trip_the_chattering_guard():
    """One monotone crossing is one event, not twenty.

    ``Constant(1) -> Integrator -> Saturation(max=0.7) -> Integrator -> Scope``
    compiled with zero crossing on: ``y = t`` crosses the limit once, at
    t = 0.7. The guard function is written on the *state*, so a restart that
    nudges only ``t`` leaves ``g`` exactly zero at the new segment start and the
    same root is found again -- twenty times at 2e-11 spacing, after which the
    run finishes on the fixed-step fallback.
    """
    cfg = cases.SATURATION
    builder = cases.build_saturating_ramp(cfg["limit"], cfg["sim_time"], cfg["sim_dt"])
    result = H.run(builder, compiled=True, zero_crossing=True)
    info = result.diagnostics.get("zero_crossing") or {}
    H.release(result)

    assert info.get("enabled"), "the diagram should register the saturation guard"
    assert not info.get("guard_tripped"), (
        "chattering guard tripped on a single monotone crossing; located {0} events".format(
            info.get("n_events")
        )
    )
    assert info.get("n_events") == 1


def test_both_corners_of_a_two_sided_saturation_are_located():
    """A second state crossing is located as exactly as the first.

    This is what a re-fired event actually costs: the first instant survives it
    (it is found before the streak builds), the second does not, because the
    guard has switched detection off by then and the tail runs on a fixed step.
    """
    rows = cases.case_two_state_corners()
    failures = cases.format_failures(rows)
    assert not failures, failures


def test_a_two_corner_run_reports_exactly_two_events():
    cfg = cases.TWO_CORNERS
    builder = cases.build_saturating_ramp(
        cfg["upper"], cfg["sim_time"], cfg["sim_dt"], lower=cfg["lower"]
    )
    result = H.run(builder, compiled=True, zero_crossing=True)
    info = result.diagnostics.get("zero_crossing") or {}
    located = cases.located_event_times(result)
    H.release(result)

    assert not info.get("guard_tripped"), info.get("guard_reason")
    assert info.get("n_events") == 2, located
    assert located == pytest.approx([cfg["lower"], cfg["upper"]], abs=1e-9)
