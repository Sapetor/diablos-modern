"""Derivative and PID on the interpreter: fresh inputs at their own instants.

Found while compiling rated blocks for the hybrid path (2026-10-10):

* ``Derivative`` ignored the ``output_only`` flag. The memory pass calls it
  with ``output_only=True`` while the input queue still holds the previous
  step's input; the block differenced that stale input and stamped the
  current time, so the real call at the same time returned the stale result.
  Unrated, the derivative lagged one solver step; rated (Ts = 0.1, dt = 0.02),
  the sample at t = 0.2 was (u(0.18) - u(0.08)) / 0.1 instead of
  (u(0.2) - u(0.1)) / 0.1.
* ``DiagramBuilder`` gave PID ``b_type`` 1 (strictly proper), while the GUI
  saves 2. PID has direct feedthrough through Kp, and only ``b_type`` 2 memory
  blocks have their fresh output delivered to consumers in the same step, so a
  rated PID built by script showed each new sample one solver step late.
"""

import numpy as np
import pytest

U = lambda t: np.sin(2.0 * t + 0.3)  # noqa: E731


def _run(fn, params, sim_time=0.5, sim_dt=0.02):
    import tempfile
    from pathlib import Path

    from lib.analysis.resim import harvest_scope_signals
    from lib.cli import load_diagram
    from lib.diagram_builder import DiagramBuilder

    b = DiagramBuilder(sim_time=sim_time, sim_dt=sim_dt)
    s = b.add_block(
        "Sine", 0, 0, name="s", params={"amplitude": 1.0, "omega": 2.0, "init_angle": 0.3}
    )
    p = b.add_block(fn, 0, 0, name="p", params=params)
    sc = b.add_block("Scope", 0, 0, name="sc", params={"labels": "y"}, in_ports=1, out_ports=0)
    b.connect(s, 0, p, 0)
    b.connect(p, 0, sc, 0)
    path = Path(tempfile.mkdtemp()) / "rated.diablos"
    b.save(str(path))
    dsim, _ = load_diagram(str(path))
    dsim.use_fast_solver = False
    ok, err = dsim.run_tuning_simulation(sim_time, sim_dt)
    assert ok, err
    data = harvest_scope_signals(dsim)
    return np.ravel(data["timeline"]), np.ravel(data["signals"]["y"])


@pytest.mark.regression
class TestDerivativeUsesTheCurrentInput:
    def test_unrated_is_the_backward_difference_at_each_step(self, qapp):
        t, y = _run("Deriv", {})
        expected = (U(t[1:]) - U(t[:-1])) / 0.02
        assert np.allclose(y[1:], expected, atol=1e-9)

    def test_rated_is_the_sampled_difference_at_each_instant(self, qapp):
        t, y = _run("Deriv", {"sampling_time": 0.1})
        for k in (1, 2, 3, 4, 5):
            i = int(np.argmin(np.abs(t - 0.1 * k)))
            expected = (U(0.1 * k) - U(0.1 * (k - 1))) / 0.1
            assert y[i] == pytest.approx(expected, abs=1e-9), k
            # ... and held until the next instant.
            assert np.allclose(y[i : i + 5], y[i], atol=1e-12)


@pytest.mark.regression
class TestRatedPidActsAtItsInstant:
    def test_builder_marks_pid_as_feedthrough(self, qapp):
        from lib.diagram_builder import BLOCK_DEFAULTS

        assert BLOCK_DEFAULTS["PID"]["b_type"] == 2

    def test_new_sample_shows_at_the_instant_itself(self, qapp):
        t, y = _run("PID", {"Kp": 1.0, "Ki": 0.0, "Kd": 0.0, "sampling_time": 0.1})
        # t = 0 is excluded: the interpreter reports 0 there for any PID
        # (tasks/todo.md).
        for k in (1, 2, 3, 4):
            i = int(np.argmin(np.abs(t - 0.1 * k)))
            assert y[i] == pytest.approx(U(0.1 * k), abs=1e-9), k
