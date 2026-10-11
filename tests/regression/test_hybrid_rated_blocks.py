"""Rated continuous blocks on the compiled (hybrid) path.

TranFn, StateSpace, PID, Integrator and Derivative given ``sampling_time > 0``
run only at their sample instants in the interpreter, discretized over
``dtime = Ts``, and hold their output in between. They used to send the whole
diagram to the interpreter; they now compile as sampled holders
(``lib/engine/compiler_kernels/discrete.py``) whose update is the block's own
``execute()``, so with only sources upstream the compiled trace equals the
interpreter's exactly.
"""

import numpy as np
import pytest

from tests.regression.test_hybrid_compiled_path import _backend, _run

SINE = ("Sine", {"amplitude": 1.0, "omega": 2.0, "init_angle": 0.3})
STEP = ("Step", {"value": 1.0, "delay": 0.25, "type": "up"})

RATED = {
    "tf_strict": ("TranFn", {"numerator": [2.0], "denominator": [1.0, 3.0, 2.0]}, [SINE]),
    "tf_proper": ("TranFn", {"numerator": [1.0, 0.5], "denominator": [1.0, 2.0]}, [SINE]),
    "ss": (
        "StateSpace",
        {
            "A": [[0.0, 1.0], [-2.0, -0.7]],
            "B": [[0.0], [1.0]],
            "C": [[1.0, 0.0]],
            "D": [[0.0]],
            "init_conds": [0.2, 0.0],
        },
        [SINE],
    ),
    "integrator": ("Integrator", {"init_conds": 0.5}, [SINE]),
    "integrator_euler": ("Integrator", {"init_conds": 0.5, "method": "FWD_EULER"}, [SINE]),
    "deriv": ("Deriv", {}, [SINE]),
}

#: Also checked against an exact sampled oracle: the block's own execute() on
#: the exact inputs at k*Ts, independent of either engine's scheduling.
ORACLE = {
    "pid": ("PID", {"Kp": 2.0, "Ki": 1.5, "Kd": 0.1, "N": 20.0}, [STEP, SINE]),
    "deriv": ("Deriv", {}, [SINE]),
}


def _source_value(src, t):
    fn, p = src
    if fn == "Sine":
        return p["amplitude"] * np.sin(p["omega"] * t + p["init_angle"])
    return p["value"] if t >= p["delay"] else 0.0


def _oracle(fn, params, sources, ts, t_out):
    """The block's own execute() at k*Ts on the exact inputs, held to the next
    instant: what a sampled block outputs by definition."""
    from blocks.derivative import DerivativeBlock
    from blocks.pid import PIDBlock

    block = {"PID": PIDBlock, "Deriv": DerivativeBlock}[fn]()
    p = {**params, "sampling_time": ts, "dtime": ts, "_init_start_": True}
    n = int(round(t_out[-1] / ts)) + 1
    samples = []
    for k in range(n):
        t = k * ts
        inputs = {i: np.atleast_1d(_source_value(src, t)) for i, src in enumerate(sources)}
        samples.append(float(np.ravel(block.execute(time=t, inputs=inputs, params=p)[0])[0]))
    idx = np.floor(np.asarray(t_out) / ts + 1e-9).astype(int)
    return np.asarray(samples)[idx], idx


def _open_loop(fn, params, sources, ts=0.1, sim_time=2.0, sim_dt=0.02):
    """sources -> rated block (sampling_time=ts) -> Scope."""
    from lib.diagram_builder import DiagramBuilder

    b = DiagramBuilder(sim_time=sim_time, sim_dt=sim_dt)
    blk = b.add_block(
        fn, 0, 0, name="g", params={**params, "sampling_time": ts}, in_ports=len(sources)
    )
    sc = b.add_block("Scope", 0, 0, name="sc", params={"labels": "y"}, in_ports=1, out_ports=0)
    for i, (src_fn, src_params) in enumerate(sources):
        src = b.add_block(src_fn, 0, 0, name=f"src{i}", params=src_params)
        b.connect(src, 0, blk, i)
    b.connect(blk, 0, sc, 0)
    return b


@pytest.mark.regression
class TestRatedBlocksMatchTheInterpreter:
    @pytest.mark.parametrize("case", sorted(RATED))
    def test_source_fed_trace_is_identical(self, qapp, tmp_path, case):
        fn, params, sources = RATED[case]
        b = _open_loop(fn, params, sources)
        dsim, fast = _run(b, tmp_path, 2.0, 0.02)
        assert _backend(dsim) == "scipy+hybrid", dsim.engine.get_compile_fallback_reason()
        assert dsim.engine.last_solver_diagnostics["zero_crossing"]["n_sample_hits"] == 21
        _, slow = _run(b, tmp_path, 2.0, 0.02, fast=False)
        a, c = np.ravel(fast["signals"]["y"]), np.ravel(slow["signals"]["y"])
        assert a.shape == c.shape
        np.testing.assert_allclose(a, c, rtol=0, atol=1e-12)

    @pytest.mark.parametrize("case", sorted(ORACLE))
    def test_matches_the_exact_sampled_oracle(self, qapp, tmp_path, case):
        fn, params, sources = ORACLE[case]
        b = _open_loop(fn, params, sources)
        dsim, fast = _run(b, tmp_path, 2.0, 0.02)
        assert _backend(dsim) == "scipy+hybrid", dsim.engine.get_compile_fallback_reason()
        t = np.ravel(fast["timeline"])
        expected, _ = _oracle(fn, params, sources, 0.1, t)
        np.testing.assert_allclose(np.ravel(fast["signals"]["y"]), expected, rtol=0, atol=1e-12)

    def test_pid_equals_the_interpreter_after_t0(self, qapp, tmp_path):
        """Same discrete math; the interpreter currently reports 0 for any PID
        at t = 0 (tasks/todo.md), so the first sample is left out."""
        fn, params, sources = ORACLE["pid"]
        b = _open_loop(fn, params, sources)
        _, fast = _run(b, tmp_path, 2.0, 0.02)
        _, slow = _run(b, tmp_path, 2.0, 0.02, fast=False)
        a, c = np.ravel(fast["signals"]["y"]), np.ravel(slow["signals"]["y"])
        assert a.shape == c.shape
        np.testing.assert_allclose(a[1:], c[1:], rtol=0, atol=1e-12)


def _rated_loop(ctrl_fn, ctrl_params, ts=0.1, sim_time=6.0, sim_dt=0.01):
    """Step -> e -> rated controller -> 2/((s+1)(s+2)) -> feedback; Scope y, u."""
    from lib.diagram_builder import DiagramBuilder

    b = DiagramBuilder(sim_time=sim_time, sim_dt=sim_dt)
    r = b.add_block("Step", 0, 0, name="r", params={"value": 1.0, "delay": 0.93, "type": "up"})
    e = b.add_block("Sum", 0, 0, name="e", params={"sign": "+-"}, in_ports=2)
    c = b.add_block(ctrl_fn, 0, 0, name="c", params={**ctrl_params, "sampling_time": ts})
    g = b.add_block(
        "TranFn", 0, 0, name="g", params={"numerator": [2.0], "denominator": [1.0, 3.0, 2.0]}
    )
    sc = b.add_block("Scope", 0, 0, name="sc", params={"labels": "y, u"}, in_ports=2, out_ports=0)
    b.connect(r, 0, e, 0)
    b.connect(e, 0, c, 0)
    b.connect(c, 0, g, 0)
    b.connect(c, 0, sc, 1)
    b.connect(g, 0, e, 1)
    b.connect(g, 0, sc, 0)
    return b


@pytest.mark.regression
class TestRatedBlocksInLoops:
    @pytest.mark.parametrize(
        "ctrl",
        [
            ("TranFn", {"numerator": [1.2, 1.0], "denominator": [1.0, 0.05]}),
            ("Integrator", {"init_conds": 0.0}),
        ],
        ids=["tf", "integrator"],
    )
    def test_loop_with_a_continuous_plant_matches_the_interpreter(self, qapp, tmp_path, ctrl):
        b = _rated_loop(*ctrl)
        dsim, fast = _run(b, tmp_path, 6.0, 0.01)
        assert _backend(dsim) == "scipy+hybrid", dsim.engine.get_compile_fallback_reason()
        assert dsim.engine.last_solver_diagnostics["zero_crossing"]["n_sample_hits"] == 61
        _, slow = _run(b, tmp_path, 6.0, 0.01, fast=False)
        for label in ("y", "u"):
            a, c = np.ravel(fast["signals"][label]), np.ravel(slow["signals"][label])
            assert a.shape == c.shape
            # Exact discrete controller; the plant is RK45 vs the interpreter's
            # exact held-input step, so they agree to the solver tolerance.
            assert np.max(np.abs(a - c)) < 1e-7, label

    def test_an_inherited_rate_integrates_over_the_inherited_period(self, qapp, tmp_path):
        """ramp -> ZOH(0.2) -> TranFn(sampling_time=0) inherits 0.2 s, and its
        holder discretizes over dtime = 0.2, as the interpreter does."""
        from lib.diagram_builder import DiagramBuilder

        b = DiagramBuilder(sim_time=2.0, sim_dt=0.05)
        r = b.add_block("Ramp", 0, 0, name="r", params={"slope": 1.0, "delay": 0.0})
        z = b.add_block("ZeroOrderHold", 0, 0, name="z", params={"sampling_time": 0.2})
        g = b.add_block(
            "TranFn",
            0,
            0,
            name="g",
            params={"numerator": [1.0], "denominator": [1.0, 1.0], "sampling_time": 0.0},
        )
        sc = b.add_block("Scope", 0, 0, name="sc", params={"labels": "y"}, in_ports=1, out_ports=0)
        b.connect(r, 0, z, 0)
        b.connect(z, 0, g, 0)
        b.connect(g, 0, sc, 0)
        dsim, fast = _run(b, tmp_path, 2.0, 0.05)
        assert _backend(dsim) == "scipy+hybrid", dsim.engine.get_compile_fallback_reason()
        holders = [
            h
            for h in dsim.engine.compiler._mode_registry.values()
            if h.get("kind") == "sampled" and h["block"].block_fn == "TranFn"
        ]
        assert len(holders) == 1
        assert holders[0]["ts"] == pytest.approx(0.2)
        assert holders[0]["template"]["dtime"] == pytest.approx(0.2)
        _, slow = _run(b, tmp_path, 2.0, 0.05, fast=False)
        np.testing.assert_allclose(
            np.ravel(fast["signals"]["y"]), np.ravel(slow["signals"]["y"]), rtol=0, atol=1e-12
        )

    def test_a_rated_block_gets_no_ode_state(self, qapp, tmp_path):
        fn, params, sources = RATED["tf_strict"]
        dsim, _ = _run(_open_loop(fn, params, sources), tmp_path, 2.0, 0.02)
        assert dsim.engine.compiler.block_executors  # compiled
        model_func = dsim.engine._compiled_system_cache_value[0]
        assert model_func.state_map == {}


@pytest.mark.regression
class TestFixedStepHybrid:
    """Euler / RK4 solver settings with sampled blocks: fixed steps on the
    output grid plus every sample instant (``solve_fixed_step_with_breakpoints``)."""

    @pytest.mark.parametrize("solver", ["Euler", "RK4"])
    @pytest.mark.parametrize("case", ["tf_strict", "integrator", "staircase"])
    def test_stateless_diagrams_equal_the_interpreter(self, qapp, tmp_path, solver, case):
        """No continuous state: the scheme is irrelevant and the trace exact."""
        from tests.regression.test_hybrid_compiled_path import _staircase

        if case == "staircase":
            b, sim_time, sim_dt = _staircase(sim_dt=0.05), 1.0, 0.05
        else:
            b, sim_time, sim_dt = _open_loop(*RATED[case]), 2.0, 0.02
        dsim, fast = _run(b, tmp_path, sim_time, sim_dt, solver=solver)
        assert _backend(dsim) == "fixed_step+hybrid", dsim.engine.get_compile_fallback_reason()
        _, slow = _run(b, tmp_path, sim_time, sim_dt, fast=False, solver=solver)
        a, c = np.ravel(fast["signals"]["y"]), np.ravel(slow["signals"]["y"])
        assert a.shape == c.shape
        np.testing.assert_allclose(a, c, rtol=0, atol=1e-12)

    @pytest.mark.parametrize("solver, floor, tol", [("RK4", 0.0, 1e-6), ("Euler", 1e-4, 3e-2)])
    def test_loop_with_a_plant_converges_to_the_adaptive_hybrid(
        self, qapp, tmp_path, solver, floor, tol
    ):
        b = _rated_loop("TranFn", {"numerator": [1.2, 1.0], "denominator": [1.0, 0.05]})
        dsim, fixed = _run(b, tmp_path, 6.0, 0.01, solver=solver)
        assert _backend(dsim) == "fixed_step+hybrid"
        assert dsim.engine.last_solver_diagnostics["zero_crossing"]["n_sample_hits"] == 61
        _, adaptive = _run(b, tmp_path, 6.0, 0.01)
        for label in ("y", "u"):
            a, c = np.ravel(fixed["signals"][label]), np.ravel(adaptive["signals"][label])
            assert a.shape == c.shape
            err = np.max(np.abs(a - c))
            # The floor proves the scheme is really first-order Euler, not RK4.
            assert floor <= err < tol, (label, err)

    def test_instants_off_the_grid_end_a_shortened_step(self, qapp, tmp_path):
        """Ts = 0.15 on a dt = 0.1 grid: every other instant falls between
        grid points, and the fixed step must stop there."""
        from lib.diagram_builder import DiagramBuilder

        b = DiagramBuilder(sim_time=1.2, sim_dt=0.1)
        s = b.add_block(
            "Sine", 0, 0, name="s", params={"amplitude": 1.0, "omega": 3.0, "init_angle": 0.0}
        )
        z = b.add_block("ZeroOrderHold", 0, 0, name="z", params={"sampling_time": 0.15})
        g = b.add_block(
            "TranFn", 0, 0, name="g", params={"numerator": [1.0], "denominator": [1.0, 1.0]}
        )
        sc = b.add_block("Scope", 0, 0, name="sc", params={"labels": "y"}, in_ports=1, out_ports=0)
        b.connect(s, 0, z, 0)
        b.connect(z, 0, g, 0)
        b.connect(g, 0, sc, 0)
        dsim, fixed = _run(b, tmp_path, 1.2, 0.1, solver="RK4")
        assert _backend(dsim) == "fixed_step+hybrid"
        assert dsim.engine.last_solver_diagnostics["zero_crossing"]["n_sample_hits"] == 9
        # Exact: y' = -y + u with u held over each 0.15 s interval.
        t = np.ravel(fixed["timeline"])
        y_exact = []
        knots = np.arange(0.0, 1.2 + 1e-9, 0.15)
        for ti in t:
            y = 0.0
            for k0 in knots[knots < ti - 1e-12]:
                u = np.sin(3.0 * k0)
                y = u + (y - u) * np.exp(-(min(k0 + 0.15, ti) - k0))
            y_exact.append(y)
        # RK4 at h <= 0.1 on a held input: ~1e-6. Ignoring the off-grid
        # instants would latch the ZOH up to 0.05 s late: errors ~1e-2.
        np.testing.assert_allclose(np.ravel(fixed["signals"]["y"]), y_exact, atol=2e-6)
