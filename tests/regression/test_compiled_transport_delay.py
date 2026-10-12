"""TransportDelay on the compiled path (method of steps).

The block used to send the whole diagram to the fixed-step interpreter, whose
linear-interpolated buffer is first order: x' = -x(t - 1) came out 6e-3 off,
and the PathSim platoon benchmark (S3) lost at N >= 20. Every k*tau is now a
breakpoint and the delayed input is recorded from the accepted steps
(``lib/engine/compiled_delays.py``).
"""

import numpy as np
import pytest
from numpy.polynomial import polynomial as P


def _run(builder, tmp_path, sim_time, sim_dt, fast=True, solver=None, twice=False):
    from lib.analysis.resim import harvest_scope_signals
    from lib.cli import load_diagram

    path = tmp_path / "delay.diablos"
    builder.save(str(path))
    dsim, _ = load_diagram(str(path))
    dsim.use_fast_solver = fast
    if solver is not None:
        dsim.solver_method = solver
    ok, err = dsim.run_tuning_simulation(sim_time, sim_dt)
    assert ok, err
    first = harvest_scope_signals(dsim)
    if twice:
        ok, err = dsim.run_tuning_simulation(sim_time, sim_dt)
        assert ok, err
        return dsim, first, harvest_scope_signals(dsim)
    return dsim, first


def _diag(dsim):
    return dsim.engine.last_solver_diagnostics or {}


def _trace(data, label):
    return np.ravel(data["timeline"]), np.ravel(data["signals"][label])


def _delayed_source(src, src_params, tau=0.3, initial=0.5, sim_time=2.0, sim_dt=0.01):
    from lib.diagram_builder import DiagramBuilder

    b = DiagramBuilder(sim_time=sim_time, sim_dt=sim_dt)
    s = b.add_block(src, 0, 0, name="s", params=src_params)
    d = b.add_block(
        "TransportDelay", 0, 0, name="d", params={"delay_time": tau, "initial_value": initial}
    )
    sc = b.add_block("Scope", 0, 0, name="sc", params={"labels": "y"}, in_ports=1, out_ports=0)
    b.connect(s, 0, d, 0)
    b.connect(d, 0, sc, 0)
    return b


def _dde(sim_time=5.0, sim_dt=0.01):
    """x' = -x(t - 1), x = 1 on [-1, 0]."""
    from lib.diagram_builder import DiagramBuilder

    b = DiagramBuilder(sim_time=sim_time, sim_dt=sim_dt)
    g = b.add_block("Gain", 0, 0, name="g", params={"gain": -1.0})
    x = b.add_block("Integrator", 0, 0, name="x", params={"init_conds": 1.0})
    d = b.add_block(
        "TransportDelay", 0, 0, name="d", params={"delay_time": 1.0, "initial_value": 1.0}
    )
    sc = b.add_block("Scope", 0, 0, name="sc", params={"labels": "x"}, in_ports=1, out_ports=0)
    b.connect(x, 0, d, 0)
    b.connect(d, 0, g, 0)
    b.connect(g, 0, x, 0)
    b.connect(x, 0, sc, 0)
    return b


def _dde_exact(t):
    """Method of steps by hand: x is a polynomial of degree k on [k, k + 1]."""
    polys, prev, x0 = [], np.array([1.0]), 1.0
    for _ in range(int(np.max(t)) + 1):
        cur = P.polyadd([x0], P.polyint(-prev))
        polys.append(cur)
        x0, prev = P.polyval(1.0, cur), cur
    k = np.minimum(np.floor(t).astype(int), len(polys) - 1)
    return np.array([P.polyval(tt - kk, polys[kk]) for tt, kk in zip(t, k)])


@pytest.mark.regression
class TestCompiledTransportDelay:
    def test_delayed_sine_is_exact_with_a_stateless_diagram(self, qapp, tmp_path):
        b = _delayed_source("Sine", {"amplitude": 1.0, "omega": 3.0, "init_angle": 0.0})
        dsim, data = _run(b, tmp_path, 2.0, 0.01)
        assert _diag(dsim)["backend"] == "scipy+hybrid"
        t, y = _trace(data, "y")
        ref = np.where(t > 0.3 + 1e-9, np.sin(3.0 * (t - 0.3)), 0.5)
        assert np.max(np.abs(y - ref)) < 1e-9

    def test_a_delayed_step_jumps_at_its_shifted_edge(self, qapp, tmp_path):
        b = _delayed_source(
            "Step", {"value": 2.0, "delay": 0.5, "type": "up"}, tau=0.2, initial=0.0
        )
        dsim, data = _run(b, tmp_path, 1.5, 0.01)
        t, y = _trace(data, "y")
        assert np.all(y[t < 0.7 - 1e-9] == 0.0)
        assert np.all(y[t > 0.7 + 1e-9] == pytest.approx(2.0, abs=1e-12))

    def test_delay_differential_equation_matches_the_method_of_steps(self, qapp, tmp_path):
        dsim, data = _run(_dde(), tmp_path, 5.0, 0.01)
        diag = _diag(dsim)
        assert diag["backend"] == "scipy+hybrid"
        assert diag["zero_crossing"]["n_segments"] == 5  # one per tau
        assert diag["zero_crossing"]["n_sample_hits"] == 0
        t, x = _trace(data, "x")
        assert np.max(np.abs(x - _dde_exact(t))) < 1e-7
        # The interpreter's buffer is first order: the reason this compiles.
        _, slow = _run(_dde(), tmp_path, 5.0, 0.01, fast=False)
        assert np.max(np.abs(_trace(slow, "x")[1] - _dde_exact(t))) > 1e-3

    def test_a_cached_compile_starts_each_run_afresh(self, qapp, tmp_path):
        dsim, first, second = _run(_dde(), tmp_path, 3.0, 0.01, twice=True)
        assert _diag(dsim).get("compile_cache_hit") is True
        assert np.array_equal(_trace(first, "x")[1], _trace(second, "x")[1])

    def test_a_rerun_with_a_new_source_value_reads_the_new_history(self, qapp, tmp_path):
        """The rerun recompiles (source values are part of the compile key) and
        must not read the first run's history."""
        from lib.analysis.resim import harvest_scope_signals
        from lib.cli import load_diagram

        path = tmp_path / "sine.diablos"
        _delayed_source("Sine", {"amplitude": 1.0, "omega": 3.0, "init_angle": 0.0}).save(str(path))
        dsim, _ = load_diagram(str(path))
        dsim.use_fast_solver = True
        assert dsim.run_tuning_simulation(2.0, 0.01)[0]
        sine = next(b for b in dsim.engine.model.blocks_list if b.block_fn == "Sine")
        sine.params["amplitude"] = 2.0
        assert dsim.run_tuning_simulation(2.0, 0.01)[0]
        t, y = _trace(harvest_scope_signals(dsim), "y")
        ref = np.where(t > 0.3 + 1e-9, 2.0 * np.sin(3.0 * (t - 0.3)), 0.5)
        assert np.max(np.abs(y - ref)) < 1e-9

    def test_held_input_through_a_delay_integrates_exactly(self, qapp, tmp_path):
        """ZOH(0.2) -> delay 0.2 -> Integrator: segments exactly tau long.

        Each segment's last stage reads the delay at the segment's own start;
        it must see the held value of the segment before, not the fresh
        sample, or the RHS jumps inside the step.
        """
        from lib.diagram_builder import DiagramBuilder

        b = DiagramBuilder(sim_time=2.0, sim_dt=0.01)
        r = b.add_block("Ramp", 0, 0, name="r", params={"slope": 1.0, "delay": 0.0})
        z = b.add_block("ZeroOrderHold", 0, 0, name="z", params={"sampling_time": 0.2})
        d = b.add_block(
            "TransportDelay", 0, 0, name="d", params={"delay_time": 0.2, "initial_value": 0.0}
        )
        x = b.add_block("Integrator", 0, 0, name="x", params={"init_conds": 0.0})
        sc = b.add_block("Scope", 0, 0, name="sc", params={"labels": "x"}, in_ports=1, out_ports=0)
        for src, dst in [(r, z), (z, d), (d, x), (x, sc)]:
            b.connect(src, 0, dst, 0)
        dsim, data = _run(b, tmp_path, 2.0, 0.01)
        t, xs = _trace(data, "x")
        # u = 0.2 (k - 1) on [0.2 k, 0.2 (k + 1)), 0 before 0.4.
        k = np.floor(t / 0.2 + 1e-9)
        u = np.maximum(0.2 * (k - 1), 0.0)
        exact = 0.04 * np.maximum(k - 2, 0) * np.maximum(k - 1, 0) / 2 + (t - 0.2 * k) * u
        assert np.max(np.abs(xs - exact)) < 1e-8

    def test_sample_hits_count_only_sample_instants(self, qapp, tmp_path):
        """Delay breakpoints share the schedule but are not sample hits."""
        from lib.diagram_builder import DiagramBuilder

        b = DiagramBuilder(sim_time=1.0, sim_dt=0.01)
        s = b.add_block(
            "Sine", 0, 0, name="s", params={"amplitude": 1.0, "omega": 2.0, "init_angle": 0.0}
        )
        z = b.add_block("ZeroOrderHold", 0, 0, name="z", params={"sampling_time": 0.1})
        d = b.add_block(
            "TransportDelay", 0, 0, name="d", params={"delay_time": 0.15, "initial_value": 0.0}
        )
        g = b.add_block(
            "TranFn", 0, 0, name="g", params={"numerator": [1.0], "denominator": [1.0, 1.0]}
        )
        sc = b.add_block(
            "Scope", 0, 0, name="sc", params={"labels": "y, h"}, in_ports=2, out_ports=0
        )
        b.connect(s, 0, z, 0)
        b.connect(z, 0, d, 0)
        b.connect(d, 0, g, 0)
        b.connect(g, 0, sc, 0)
        b.connect(d, 0, sc, 1)
        dsim, fast = _run(b, tmp_path, 1.0, 0.01)
        assert _diag(dsim)["zero_crossing"]["n_sample_hits"] == 11  # t = 0, 0.1, ..., 1.0
        # The held staircase, shifted by 0.15: each step of the ZOH lands
        # 0.15 s later (right-continuous at the shifted instant).
        t, h = _trace(fast, "h")
        for k in range(1, 6):
            i = int(np.argmin(np.abs(t - (0.1 * k + 0.15 + 0.02))))
            assert h[i] == pytest.approx(np.sin(0.2 * k), abs=1e-9), k
            # ... and the previous sample until then: the history of a
            # segment is recorded before the update at its end.
            i = int(np.argmin(np.abs(t - (0.1 * k + 0.15 - 0.02))))
            assert h[i] == pytest.approx(np.sin(0.2 * (k - 1)), abs=1e-9), k

    def test_events_and_delay_breakpoints_share_one_solve(self, qapp, tmp_path):
        """Delayed feedback through a Saturation: located events plus k*tau."""
        from lib.diagram_builder import DiagramBuilder

        def build():
            b = DiagramBuilder(sim_time=6.0, sim_dt=0.01)
            r = b.add_block("Step", 0, 0, name="r", params={"value": 1.0, "delay": 0.0})
            e = b.add_block("Sum", 0, 0, name="e", params={"sign": "+-"}, in_ports=2)
            k = b.add_block("Gain", 0, 0, name="k", params={"gain": 4.0})
            sat = b.add_block("Saturation", 0, 0, name="sat", params={"min": -1.0, "max": 1.0})
            g = b.add_block(
                "TranFn", 0, 0, name="g", params={"numerator": [1.0], "denominator": [1.0, 1.0]}
            )
            d = b.add_block(
                "TransportDelay",
                0,
                0,
                name="d",
                params={"delay_time": 0.25, "initial_value": 0.0},
            )
            sc = b.add_block(
                "Scope", 0, 0, name="sc", params={"labels": "y"}, in_ports=1, out_ports=0
            )
            for src, dst, port in [
                (r, e, 0),
                (d, e, 1),
                (e, k, 0),
                (k, sat, 0),
                (sat, g, 0),
                (g, d, 0),
                (g, sc, 0),
            ]:
                b.connect(src, 0, dst, port)
            return b

        dsim, fast = _run(build(), tmp_path, 6.0, 0.01)
        zc = _diag(dsim)["zero_crossing"]
        assert zc["n_events"] > 0
        _, slow = _run(build(), tmp_path, 6.0, 0.001, fast=False)
        t, y = _trace(fast, "y")
        ts, ys = _trace(slow, "y")
        # The interpreter is first order: compare at dt = 0.001 on the shared grid.
        assert np.max(np.abs(y - np.interp(t, ts, ys))) < 5e-3


@pytest.mark.regression
class TestTransportDelayFallbacks:
    def test_fixed_step_keeps_the_interpreter(self, qapp, tmp_path):
        dsim, _ = _run(_dde(), tmp_path, 2.0, 0.01, solver="RK4")
        assert _diag(dsim).get("backend") is None
        assert "fixed-step RK4" in dsim.engine.get_compile_fallback_reason()

    def test_zero_delay_keeps_the_interpreter(self, qapp, tmp_path):
        b = _delayed_source("Sine", {"amplitude": 1.0, "omega": 3.0, "init_angle": 0.0}, tau=0.0)
        dsim, _ = _run(b, tmp_path, 1.0, 0.01)
        assert _diag(dsim).get("backend") is None
        assert "not positive" in dsim.engine.get_compile_fallback_reason()

    def test_linearizer_refuses_a_delay(self, qapp, tmp_path):
        from lib.analysis.linearizer import Linearizer
        from lib.cli import load_diagram

        path = tmp_path / "dde.diablos"
        _dde().save(str(path))
        dsim, _ = load_diagram(str(path))
        with pytest.raises(ValueError, match="finite-dimensional"):
            Linearizer(dsim)._compile_diagram()


@pytest.mark.unit
class TestDelayHistory:
    @staticmethod
    def _history():
        from types import SimpleNamespace

        from lib.engine.compiled_delays import DelayHistory

        h = DelayHistory()
        h.add("d", 0.1, 0.5, "u")
        h.model_func = SimpleNamespace(evaluate=lambda t, y: (None, {"u": t * t}))
        h.reset(0.0, 0.01, 1.0)

        class Dense:
            ts = np.array([0.0, 0.04, 0.1])

            def __call__(self, t):
                return np.zeros((1, np.size(t)))

        return h, SimpleNamespace(sol=Dense())

    def test_reads_the_recorded_segment_and_refuses_beyond_it(self):
        from lib.engine.compiled_delays import CompiledDelayError

        h, sol = self._history()
        assert h.value(0, 0.0) == pytest.approx(0.5)  # initial at t0
        with pytest.raises(CompiledDelayError):
            h.value(0, 0.05)  # nothing integrated yet
        h.finish(sol)
        assert h.value(0, 0.07) == pytest.approx(0.0049, abs=1e-12)  # u = t^2, cubic-exact
        with pytest.raises(CompiledDelayError):
            h.value(0, 0.15)

    def test_reset_forgets_the_previous_run(self):
        from lib.engine.compiled_delays import CompiledDelayError

        h, sol = self._history()
        h.finish(sol)
        h.reset(0.0, 0.01, 1.0)
        with pytest.raises(CompiledDelayError):
            h.value(0, 0.07)
