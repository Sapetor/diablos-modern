"""Validation cases that currently fail, each pinned as a strict xfail.

Nothing here is a tolerance quibble: every one of these is a wrong answer, a
crash, or a documented invariant the engine does not hold. They are marked
``xfail(strict=True)`` so the suite stays green while the defect stands and
turns red the moment it is fixed -- at which point the marker comes off and the
case joins the table in ``docs/VALIDATION.md``.

Each test is written as the check that *should* pass, with the observed failure
described in its docstring, so the test doubles as the reproducer.
"""

import numpy as np
import pytest

from tests.validation import _cases as cases
from tests.validation import _harness as H

pytestmark = pytest.mark.validation


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Interpreted PID never sees the reference step in its derivative branch: "
        "blocks/pid.py seeds _prev_e with the first error sample on the initializing "
        "call, so de = 0 at t0 and the filtered derivative's response to the step "
        "discontinuity is lost. The compiled kernel starts its filter state at 0 and "
        "does produce it (u(0+) = Kd*N), matching C(s) = Kp + Ki/s + Kd*N*s/(s+N). "
        "The resulting closed-loop error does not vanish with dt: max |y - y_analytic| "
        "is 0.2866 / 0.2843 / 0.2834 / 0.2831 at dt = 8e-3 / 4e-3 / 2e-3 / 1e-3 "
        "(observed order 0.00). With Kp = Ki = 0 and Kd = 0.5 the interpreted loop "
        "output is identically zero while the true response peaks at 0.36."
    ),
)
def test_interpreted_pid_loop_converges_to_the_analytic_closed_loop():
    """Refining dt must drive the interpreted loop onto ``CP/(1+CP)``.

    Reproducer: build ``Step -> PID(Kp=4, Ki=2, Kd=0.5, N=20) -> plant
    4/(s^2+1.2s+4) -> PID.measurement``, run it interpreted at two step sizes,
    and compare against ``lsim`` of the analytic closed loop. A first-order
    method would quarter its error over these two refinements; this one does not
    move.
    """
    reference_dt = 0.002
    fine_dt = 0.0005
    errors = []
    for dt in (reference_dt, fine_dt):
        builder = cases.build_pid_loop(
            cases.PID_GAINS,
            cases.PID_PLANT_NUM,
            cases.PID_PLANT_DEN,
            cases.PID_LOOP["sim_time"],
            dt,
        )
        result = H.run(builder, compiled=False)
        _t, y = result.signal("y")
        reference = cases.pid_loop_reference(cases.PID_LOOP["sim_time"], dt)
        errors.append(H.max_abs_error(y, reference[: len(y)]))
        H.release(result)

    assert errors[1] < 0.5 * errors[0], (
        "interpreted PID loop error {0:.4f} -> {1:.4f} over a 4x refinement".format(*errors)
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Interpreted Integrator crashes for method TUSTIN and BWD_EULER when the "
        "upstream block emits a 0-d array. blocks/sine.py returns "
        "np.array(scalar) (shape ()), which blocks/integrator.py does not promote "
        "(its guard tests isinstance(inputs[0], (float, int))); the shape check then "
        "rewrites params['mem'] as np.full((), ...), a 0-d array, while "
        "params['mem_list'][0] was allocated with shape (1,) at init. The in-place "
        "'mem += ...' of those two branches raises ValueError: non-broadcastable "
        "output operand with shape () doesn't match the broadcast shape (1,). "
        "FWD_EULER, RK4 and SOLVE_IVP survive because they never mix the two shapes, "
        "and Step/Ramp/Constant sources survive because they emit shape (1,)."
    ),
)
@pytest.mark.parametrize("method", ["TUSTIN", "BWD_EULER"])
def test_trapezoidal_and_backward_euler_integrate_a_sine(method):
    """``int A sin(wt) dt`` must run (and converge) for every declared method.

    Reproducer: ``Sine -> Integrator(method='TUSTIN') -> Scope`` on the
    interpreter. TUSTIN is second order, so this asserts only that it runs and
    beats explicit Euler's first-order error.
    """
    builder = cases.build_integrator(
        "Sine",
        {"amplitude": 1.0, "omega": 2.0, "init_angle": 0.0},
        2.0,
        0.01,
        method=method,
    )
    result = H.run(builder, compiled=False)
    t, y = result.signal("y")
    analytic = (1.0 - np.cos(2.0 * t)) / 2.0
    assert H.max_abs_error(y, analytic) < 1e-3
    H.release(result)


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
