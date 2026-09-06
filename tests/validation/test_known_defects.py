"""Validation cases that currently fail, each pinned as a strict xfail.

Nothing here is a tolerance quibble: every one of these is a wrong answer, a
crash, or a documented invariant the engine does not hold. They are marked
``xfail(strict=True)`` so the suite stays green while the defect stands and
turns red the moment it is fixed -- at which point the marker comes off and the
case joins the table in ``docs/VALIDATION.md``.

Each test is written as the check that *should* pass, with the observed failure
described in its docstring, so the test doubles as the reproducer.
"""

import pytest

from tests.validation import _cases as cases
from tests.validation import _harness as H

pytestmark = pytest.mark.validation


@pytest.mark.xfail(
    strict=True,
    reason=(
        "A state-dependent zero crossing that the trajectory passes through ONCE "
        "re-fires until the chattering guard gives up. lib/engine/zero_crossing.py "
        "restarts each segment at the located root plus a nudge of 1e-11 of the span, "
        "which advances t but carries the state across unchanged; a guard written on "
        "the state (Saturation's 'u - max', where u is an Integrator output) is "
        "therefore still exactly 0 at the new segment start, and scipy's "
        "find_active_events counts a zero as active. The run logs 'chattering: 20 "
        "consecutive events closer than 2e-09s (around t=0.7s, saturation2:upper_limit)' "
        "and finishes on the fixed-step fallback, so switching instants after the first "
        "are located only to step accuracy. The doc's nudge protects only guards that "
        "are functions of t (Step, Ramp), not of y. Same symptom for Switch."
    ),
)
def test_a_single_state_crossing_does_not_trip_the_chattering_guard():
    """One monotone crossing is one event, not twenty.

    Reproducer: ``Constant(1) -> Integrator -> Saturation(max=0.7) -> Integrator
    -> Scope`` compiled with zero crossing on. ``y = t`` crosses the limit once,
    at t = 0.7; the diagnostics report 20 events at 2e-11 spacing and
    ``guard_tripped``.
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
