import numpy as np
import pytest

from blocks.variable_transport_delay import VariableTransportDelayBlock
from blocks.transport_delay import TransportDelayBlock


def _run_variable(signal_fn, tau_fn, t_end=2.0, dt=0.01,
                  max_delay=1.0, initial_value=0.0):
    """Drive VariableTransportDelay via an execute() loop.

    Returns (times, outputs) as numpy arrays of scalars.
    """
    block = VariableTransportDelayBlock()
    params = {p: spec["default"] for p, spec in block.params.items()}
    params["max_delay"] = max_delay
    params["initial_value"] = initial_value

    times = []
    outputs = []
    t = 0.0
    n = int(round(t_end / dt)) + 1
    for k in range(n):
        t = k * dt
        u = np.atleast_1d(float(signal_fn(t)))
        tau = float(tau_fn(t))
        result = block.execute(time=t, inputs={0: u, 1: tau}, params=params, dtime=dt)
        times.append(t)
        outputs.append(float(np.atleast_1d(result[0])[0]))
    return np.array(times), np.array(outputs)


def _run_fixed(signal_fn, delay_time, t_end=2.0, dt=0.01, initial_value=0.0):
    """Reference: fixed TransportDelay driven the same way."""
    block = TransportDelayBlock()
    params = {p: spec["default"] for p, spec in block.params.items()}
    params["delay_time"] = delay_time
    params["initial_value"] = initial_value

    outputs = []
    n = int(round(t_end / dt)) + 1
    for k in range(n):
        t = k * dt
        u = np.atleast_1d(float(signal_fn(t)))
        result = block.execute(time=t, inputs={0: u}, params=params, dtime=dt)
        outputs.append(float(np.atleast_1d(result[0])[0]))
    return np.array(outputs)


@pytest.mark.unit
class TestVariableTransportDelay:
    def test_constant_tau_matches_fixed_delay_on_ramp(self):
        """Constant τ should reproduce a fixed TransportDelay on a ramp."""
        ramp = lambda t: t
        tau = 0.3
        dt = 0.01
        times, var_out = _run_variable(ramp, lambda t: tau, t_end=2.0, dt=dt,
                                       max_delay=1.0)
        fixed_out = _run_fixed(ramp, tau, t_end=2.0, dt=dt)

        # Same buffering/interpolation scheme => outputs should match closely.
        assert np.allclose(var_out, fixed_out, atol=1e-9)

        # And the delayed ramp equals u(t - tau) once t >= tau.
        mask = times >= tau + dt
        expected = times[mask] - tau
        assert np.allclose(var_out[mask], expected, atol=2 * dt)

    def test_constant_tau_matches_fixed_delay_on_step(self):
        step = lambda t: 1.0 if t >= 0.5 else 0.0
        tau = 0.2
        dt = 0.01
        _, var_out = _run_variable(step, lambda t: tau, t_end=1.5, dt=dt)
        fixed_out = _run_fixed(step, tau, t_end=1.5, dt=dt)
        assert np.allclose(var_out, fixed_out, atol=1e-9)

    def test_tau_zero_passes_current_value_through(self):
        """τ = 0 should output the current input (passthrough)."""
        sig = lambda t: np.sin(3.0 * t)
        times, out = _run_variable(sig, lambda t: 0.0, t_end=2.0, dt=0.01)
        expected = np.array([sig(t) for t in times])
        assert np.allclose(out, expected, atol=1e-12)

    def test_time_varying_tau_changes_effective_delay(self):
        """A τ that grows over time delays a ramp by an increasing amount."""
        ramp = lambda t: t
        dt = 0.01
        max_delay = 1.0

        # Small constant delay run.
        times, out_small = _run_variable(ramp, lambda t: 0.1, t_end=2.0, dt=dt,
                                         max_delay=max_delay)
        # Larger constant delay run.
        _, out_large = _run_variable(ramp, lambda t: 0.5, t_end=2.0, dt=dt,
                                     max_delay=max_delay)

        # In the steady region, a larger delay yields a smaller (more lagged)
        # output for a monotonically increasing ramp.
        mask = times >= 0.7
        assert np.all(out_large[mask] < out_small[mask])

        # Effective delay for the time-varying run should track tau(t).
        tau_fn = lambda t: 0.1 + 0.2 * t  # grows from 0.1 toward larger lag
        times, out_var = _run_variable(ramp, tau_fn, t_end=2.0, dt=dt,
                                       max_delay=max_delay)
        mask = times >= 0.5
        expected = times[mask] - np.array([min(tau_fn(t), max_delay) for t in times[mask]])
        assert np.allclose(out_var[mask], expected, atol=3 * dt)

    def test_tau_clamped_to_max_delay(self):
        """A τ exceeding max_delay is clamped to max_delay."""
        ramp = lambda t: t
        dt = 0.01
        max_delay = 0.3

        # Request a huge tau; it must clamp to max_delay (0.3).
        times, out_huge = _run_variable(ramp, lambda t: 5.0, t_end=2.0, dt=dt,
                                        max_delay=max_delay)
        # Reference: a fixed delay exactly at max_delay.
        fixed_at_max = _run_fixed(ramp, max_delay, t_end=2.0, dt=dt)

        assert np.allclose(out_huge, fixed_at_max, atol=1e-9)

        # In the steady region the lag equals max_delay, not 5.0.
        mask = times >= max_delay + dt
        expected = times[mask] - max_delay
        assert np.allclose(out_huge[mask], expected, atol=2 * dt)

    def test_tau_clamped_to_zero_for_negative(self):
        """A negative τ is clamped to 0 (passthrough)."""
        sig = lambda t: 2.0 * t + 1.0
        times, out = _run_variable(sig, lambda t: -1.0, t_end=1.0, dt=0.01)
        expected = np.array([sig(t) for t in times])
        assert np.allclose(out, expected, atol=1e-12)

    def test_request_before_buffer_returns_initial_value(self):
        """Before enough history exists, output is the initial value."""
        initial = 7.0
        ramp = lambda t: t + 100.0  # signal far from initial value
        # Large tau so target_time precedes the buffer start for early steps.
        times, out = _run_variable(ramp, lambda t: 1.0, t_end=2.0, dt=0.01,
                                   max_delay=1.0, initial_value=initial)
        # At t=0 (and while target_time <= buffer start) output is initial_value.
        assert np.isclose(out[0], initial)
        # Early steps where target_time < 0 stay at initial_value.
        early = times < 1.0
        assert np.allclose(out[early], initial, atol=1e-9)
