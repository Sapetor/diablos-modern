"""The t = 0 sample of feedthrough memory blocks, on both paths.

Interpreter: a memory block with direct feedthrough (``b_type`` 2: PID,
ZeroOrderHold, Hysteresis, FirstOrderHold, RateTransition, RateLimiter) used
to report its Loop 1 ``output_only`` value at t = 0, taken before its input
was known: PID showed 0 instead of Kp*e(0), and a ZOH behind a Gain held 0
instead of u(0). Loop 2 now runs it once its inputs are ready
(``SimulationEngine._defers_t0_output``).

Compiled hybrid path: the event latches were seeded before the t0 sample, so a
Hysteresis behind a ZOH seeded against the ZOH's empty holder and, frozen,
kept the wrong mode for the whole run (``zero_crossing.solve_with_events``).
"""

import numpy as np
import pytest

from tests.regression.test_hybrid_compiled_path import _backend, _run

U0 = np.sin(0.6)  # the Sine's value at t = 0


def _chain(fn, params, via_gain, sim_time=0.3, sim_dt=0.01, ports=1):
    """Sine (u(0) = sin 0.6) [-> Gain] -> block -> Scope."""
    from lib.diagram_builder import DiagramBuilder

    b = DiagramBuilder(sim_time=sim_time, sim_dt=sim_dt)
    s = b.add_block(
        "Sine", 0, 0, name="s", params={"amplitude": 1.0, "omega": 2.0, "init_angle": 0.6}
    )
    blk = b.add_block(fn, 0, 0, name="g", params=params, in_ports=ports)
    sc = b.add_block("Scope", 0, 0, name="sc", params={"labels": "y"}, in_ports=1, out_ports=0)
    src = s
    if via_gain:  # resolves only in Loop 2, after the memory pass
        src = b.add_block("Gain", 0, 0, name="k", params={"gain": 1.0})
        b.connect(s, 0, src, 0)
    b.connect(src, 0, blk, 0)
    b.connect(blk, 0, sc, 0)
    return b


def _y(data):
    return np.ravel(data["signals"]["y"])


PID = ("PID", {"Kp": 2.0, "Ki": 1.0, "Kd": 0.0})
ZOH = ("ZeroOrderHold", {"sampling_time": 0.1})
HYST = ("Hysteresis", {"upper": 0.2, "lower": -0.2, "high": 1.0, "low": -1.0})


@pytest.mark.regression
class TestInterpreterT0:
    @pytest.mark.parametrize("via_gain", [False, True], ids=["direct", "via_gain"])
    @pytest.mark.parametrize("block", [PID, ZOH, HYST], ids=["pid", "zoh", "hysteresis"])
    def test_matches_the_compiled_path(self, qapp, tmp_path, block, via_gain):
        b = _chain(*block, via_gain)
        dsim, fast = _run(b, tmp_path, 0.3, 0.01)
        assert _backend(dsim) is not None, dsim.engine.get_compile_fallback_reason()
        _, slow = _run(b, tmp_path, 0.3, 0.01, fast=False)
        a, c = _y(fast), _y(slow)
        assert a[0] == pytest.approx(c[0], abs=1e-12), (a[0], c[0])
        if block is not PID:  # PID's integral: rectangle steps vs continuous
            np.testing.assert_allclose(a, c, rtol=0, atol=1e-12)

    @pytest.mark.parametrize("via_gain", [False, True], ids=["direct", "via_gain"])
    def test_pid_trace_is_one_execute_per_step(self, qapp, tmp_path, via_gain):
        """Kp*e(0) at t = 0, and exactly one state advance per step after it:
        executing at t = 0 in both Loop 2 and Loop 3 would add Ki*e(0)*dt to
        every later sample."""
        from blocks.pid import PIDBlock

        _, slow = _run(_chain(*PID, via_gain), tmp_path, 0.3, 0.01, fast=False)
        y = _y(slow)
        assert y[0] == pytest.approx(2.0 * U0, abs=1e-12)
        block, p = PIDBlock(), {**PID[1], "dtime": 0.01, "_init_start_": True}
        expected = [
            float(
                block.execute(time=t, inputs={0: np.atleast_1d(np.sin(2 * t + 0.6))}, params=p)[0][
                    0
                ]
            )
            for t in np.arange(y.size) * 0.01
        ]
        np.testing.assert_allclose(y, expected, rtol=0, atol=1e-12)

    @pytest.mark.parametrize(
        "block",
        [
            ("FirstOrderHold", {"input_sample_time": 0.1}),
            ("RateTransition", {"output_sample_time": 0.1}),
            ("RateLimiter", {}),  # latches its first input
        ],
        ids=["foh", "rate_transition", "rate_limiter"],
    )
    def test_interpreter_only_blocks_start_at_their_input(self, qapp, tmp_path, block):
        _, slow = _run(_chain(*block, via_gain=True), tmp_path, 0.3, 0.01, fast=False)
        assert _y(slow)[0] == pytest.approx(U0, abs=1e-12)

    def test_a_loop_of_feedthrough_blocks_still_resolves(self, qapp, tmp_path):
        """Step -> Sum -> PID -> Gain -> back to the Sum has no dynamic block
        to break it at t = 0: the PID falls back to its output_only value
        there, as before, and the run proceeds."""
        from lib.diagram_builder import DiagramBuilder

        b = DiagramBuilder(sim_time=0.1, sim_dt=0.01)
        r = b.add_block("Step", 0, 0, name="r", params={"value": 1.0, "delay": 0.0})
        e = b.add_block("Sum", 0, 0, name="e", params={"sign": "+-"}, in_ports=2)
        pid = b.add_block("PID", 0, 0, name="c", params={"Kp": 0.5, "Ki": 0.0, "Kd": 0.0})
        k = b.add_block("Gain", 0, 0, name="k", params={"gain": 1.0})
        sc = b.add_block("Scope", 0, 0, name="sc", params={"labels": "y"}, in_ports=1, out_ports=0)
        b.connect(r, 0, e, 0)
        b.connect(e, 0, pid, 0)
        b.connect(pid, 0, k, 0)
        b.connect(k, 0, e, 1)
        b.connect(pid, 0, sc, 0)
        _, slow = _run(b, tmp_path, 0.1, 0.01, fast=False)
        y = _y(slow)
        assert y.size == 11 and np.all(np.isfinite(y))
        assert y[0] == 0.0


def _sampled_latch(order, upper=0.2, lower=-0.2, init_angle=0.6, omega=2.0):
    """Sine -> ZOH(0.1) -> Hysteresis -> Scope, or Sine -> Hysteresis -> ZOH."""
    from lib.diagram_builder import DiagramBuilder

    b = DiagramBuilder(sim_time=1.0, sim_dt=0.01)
    s = b.add_block(
        "Sine", 0, 0, name="s", params={"amplitude": 1.0, "omega": omega, "init_angle": init_angle}
    )
    z = b.add_block("ZeroOrderHold", 0, 0, name="z", params={"sampling_time": 0.1})
    h = b.add_block(
        "Hysteresis",
        0,
        0,
        name="h",
        params={"upper": upper, "lower": lower, "high": 1.0, "low": -1.0},
    )
    sc = b.add_block("Scope", 0, 0, name="sc", params={"labels": "y"}, in_ports=1, out_ports=0)
    first, second = (z, h) if order == "zoh_first" else (h, z)
    b.connect(s, 0, first, 0)
    b.connect(first, 0, second, 0)
    b.connect(second, 0, sc, 0)
    return b


@pytest.mark.regression
class TestHybridT0Seeding:
    @pytest.mark.parametrize("order", ["zoh_first", "hysteresis_first"])
    def test_latch_and_sample_agree_with_the_interpreter(self, qapp, tmp_path, order):
        """u stays above the upper threshold all run, so the right answer is
        'high' throughout; the old order gave 'low' throughout behind a ZOH."""
        b = _sampled_latch(order)
        dsim, fast = _run(b, tmp_path, 1.0, 0.01)
        assert _backend(dsim) == "scipy+hybrid"
        _, slow = _run(b, tmp_path, 1.0, 0.01, fast=False)
        np.testing.assert_array_equal(_y(fast), _y(slow))
        assert np.all(_y(fast) == 1.0)

    def test_the_first_seed_does_not_leak_into_the_final_mode(self, qapp, tmp_path):
        """The empty holder (0) is above an upper threshold of -0.2, but the
        first sample, -0.3, is inside the band: the latch must keep its
        initial 'low', not the 'high' a seed against the empty holder gives."""
        b = _sampled_latch(
            "zoh_first", upper=-0.2, lower=-0.5, init_angle=float(np.arcsin(-0.3)), omega=0.05
        )
        dsim, fast = _run(b, tmp_path, 1.0, 0.01)
        assert _backend(dsim) == "scipy+hybrid"
        _, slow = _run(b, tmp_path, 1.0, 0.01, fast=False)
        np.testing.assert_array_equal(_y(fast), _y(slow))
        assert np.all(_y(fast) == -1.0)


RATE_LIMITER = ("RateLimiter", {"rising_slew": 0.5, "falling_slew": 0.5})


@pytest.mark.regression
class TestCompiledRateLimiterT0:
    """The RateLimiter latches its first input, y(0) = u(0). Its compiled state
    is allocated before u(0) is known and seeded at t0 by
    ``compiled_runner.seed_initial_states``."""

    @pytest.mark.parametrize("via_gain", [False, True], ids=["direct", "via_gain"])
    def test_starts_at_its_input_like_the_interpreter(self, qapp, tmp_path, via_gain):
        b = _chain(*RATE_LIMITER, via_gain, sim_time=1.0)
        dsim, fast = _run(b, tmp_path, 1.0, 0.01)
        assert _backend(dsim) == "scipy"
        _, slow = _run(b, tmp_path, 1.0, 0.01, fast=False)
        a, c = _y(fast), _y(slow)
        assert a[0] == pytest.approx(U0, abs=1e-12)
        # Slew-limited from u(0) on both paths (exact clamp vs the compiled
        # stiff chase); from 0 the compiled trace would trail by ~0.5.
        np.testing.assert_allclose(a, c, atol=2e-3)

    def test_behind_a_zoh_it_seeds_from_the_first_sample(self, qapp, tmp_path):
        from lib.diagram_builder import DiagramBuilder

        b = DiagramBuilder(sim_time=1.0, sim_dt=0.01)
        s = b.add_block(
            "Sine", 0, 0, name="s", params={"amplitude": 1.0, "omega": 2.0, "init_angle": 0.6}
        )
        z = b.add_block("ZeroOrderHold", 0, 0, name="z", params={"sampling_time": 0.1})
        r = b.add_block("RateLimiter", 0, 0, name="r", params=RATE_LIMITER[1])
        sc = b.add_block("Scope", 0, 0, name="sc", params={"labels": "y"}, in_ports=1, out_ports=0)
        b.connect(s, 0, z, 0)
        b.connect(z, 0, r, 0)
        b.connect(r, 0, sc, 0)
        dsim, fast = _run(b, tmp_path, 1.0, 0.01)
        assert _backend(dsim) == "scipy+hybrid"
        _, slow = _run(b, tmp_path, 1.0, 0.01, fast=False)
        a, c = _y(fast), _y(slow)
        # Until the ZOH's first jump at t = 0.1 both hold the first sample.
        np.testing.assert_allclose(a[:10], U0, rtol=0, atol=1e-12)
        np.testing.assert_allclose(c[:10], U0, rtol=0, atol=1e-12)
        # After a jump the interpreter's per-step clamp moves one slew step
        # (0.5 * dt) at the jump itself, the compiled limiter right after it.
        np.testing.assert_allclose(a, c, atol=0.5 * 0.01 + 1e-3)
        # The solve's own t0 sample ran once on clean holders.
        assert dsim.engine.last_solver_diagnostics["zero_crossing"]["n_sample_hits"] == 11

    def test_a_cached_compile_is_seeded_afresh_each_run(self, qapp, tmp_path):
        b = _chain(*RATE_LIMITER, via_gain=True, sim_time=1.0)
        dsim, first, second = _run(b, tmp_path, 1.0, 0.01, twice=True)
        assert dsim.engine.last_solver_diagnostics.get("compile_cache_hit") is True
        np.testing.assert_array_equal(_y(first), _y(second))
        cached_y0 = dsim.engine._compiled_system_cache_value[1]
        assert np.all(cached_y0 == 0.0)  # the seeded value lives in a copy only


@pytest.mark.unit
def test_seed_initial_states_costs_nothing_without_seeders():
    from lib.engine.compiled_runner import seed_initial_states

    class NoEval:
        def evaluate(self, t, y):  # pragma: no cover - must not be called
            raise AssertionError("evaluated without a seeder")

    y0 = np.zeros(3)
    assert seed_initial_states(NoEval(), {"a": lambda *a: None}, 0.0, y0) is y0


@pytest.mark.unit
def test_seed_initial_states_seeds_a_copy():
    from lib.engine.compiled_runner import seed_initial_states

    class Model:
        def evaluate(self, t, y):
            return np.zeros_like(y), {"src": 0.7}

    def executor():
        pass

    executor.seed_state = lambda y0, signals: y0.__setitem__(1, signals["src"])
    y0 = np.zeros(3)
    seeded = seed_initial_states(Model(), {"r": executor}, 0.0, y0)
    assert seeded is not y0 and np.all(y0 == 0.0)
    np.testing.assert_array_equal(seeded, [0.0, 0.7, 0.0])


@pytest.mark.regression
def test_seeding_leaves_the_sampled_holders_clean(qapp, tmp_path):
    """The seeding pass takes a t0 sample; the holders must be reset after it,
    or a DiscreteTranFn advances twice at t0 and its staircase runs a period
    ahead of the interpreter's."""
    from lib.diagram_builder import DiagramBuilder

    b = DiagramBuilder(sim_time=1.0, sim_dt=0.05)
    r = b.add_block("Ramp", 0, 0, name="r", params={"slope": 1.0, "delay": 0.0})
    z = b.add_block("ZeroOrderHold", 0, 0, name="z", params={"sampling_time": 0.2})
    d = b.add_block(
        "DiscreteTranFn",
        0,
        0,
        name="d",
        params={"numerator": [1.0], "denominator": [1.0, -0.5], "sampling_time": 0.0},
    )
    lim = b.add_block("RateLimiter", 0, 0, name="lim", params=RATE_LIMITER[1])
    sc = b.add_block("Scope", 0, 0, name="sc", params={"labels": "y, d"}, in_ports=2, out_ports=0)
    b.connect(r, 0, z, 0)
    b.connect(z, 0, d, 0)
    b.connect(d, 0, lim, 0)
    b.connect(lim, 0, sc, 0)
    b.connect(d, 0, sc, 1)
    dsim, fast = _run(b, tmp_path, 1.0, 0.05)
    assert _backend(dsim) == "scipy+hybrid"
    _, slow = _run(b, tmp_path, 1.0, 0.05, fast=False)
    np.testing.assert_allclose(
        np.ravel(fast["signals"]["d"]), np.ravel(slow["signals"]["d"]), rtol=0, atol=1e-12
    )
