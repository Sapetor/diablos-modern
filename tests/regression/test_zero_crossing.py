"""Zero-crossing (event) detection on the compiled fast-solver path.

The compiled path folds a diagram into one ODE right-hand side; a discontinuous
block makes that RHS piecewise, and without event detection an adaptive step
straddling the switching instant only ever locates it to step accuracy.  These
tests pin the four properties that buys us:

1. Switching instants land on their analytic time (relay feedback, saturation
   limits, a step edge) rather than somewhere inside a step.
2. The located instants -- and the trajectory through them -- do not depend on
   the output step size, which is the visible symptom of a smeared switch.
3. A chattering relay terminates: the guard gives up, says so, and finishes with
   a fixed step instead of grinding the adaptive step toward machine epsilon.
4. With the setting off, the compiled path is byte-identical to the plain
   single-shot ``solve_ivp`` call it has always made -- including for every
   example diagram that registers no events at all.

See ``lib/engine/zero_crossing.py`` and ``docs/FAST_SOLVER.md``.
"""

import gc
import time
from pathlib import Path

import numpy as np
import pytest

from lib.block_loader import load_blocks
from lib.diagram_builder import DiagramBuilder


pytestmark = pytest.mark.regression

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"


@pytest.fixture(autouse=True)
def _collect_dsims():
    """Reclaim each test's DSim while Qt is idle.

    Every test here builds one or more DSim objects, each of which owns
    pyqtgraph-backed plotting state.  Left to CPython's own timing, that state
    is freed at whatever arbitrary point a later allocation triggers a
    collection -- and a pyqtgraph item deleted from under a widget that is
    mid-construction segfaults the interpreter, so a module that leaks a few
    dozen of them shows up as a crash somewhere else entirely.  Collecting on
    the way out of each test keeps the deletions here, where nothing is being
    built.
    """
    yield
    gc.collect()


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #
_PARAM_DEFAULTS = {}


def _defaults(block_name):
    """Declared defaults for a block, so builder diagrams are fully populated.

    DiagramBuilder writes only the params it is handed, but the engine's
    interpreted initialization step reads a block's full parameter set (an
    Integrator wants ``method``, say).  Pull the defaults from the block class
    itself rather than restating them here.
    """
    if not _PARAM_DEFAULTS:
        for cls in load_blocks():
            try:
                block = cls()
                _PARAM_DEFAULTS[block.block_name] = {
                    name: meta["default"]
                    for name, meta in (block.params or {}).items()
                    if isinstance(meta, dict) and "default" in meta
                }
            except Exception:  # noqa: BLE001 - not every block is bare-constructible
                continue
    return dict(_PARAM_DEFAULTS.get(block_name, {}))


def _add(builder, block_type, name, params=None, **kwargs):
    merged = _defaults(block_type)
    merged.update(params or {})
    x = 60 + 120 * len(builder.blocks)
    return builder.add_block(block_type, x, 100, name=name, params=merged, **kwargs)


def _run(builder, tmp_path, sim_time, sim_dt, zero_crossing=True, max_events=None):
    """Save the built diagram, load it and run it through the compiled path."""
    from lib.lib import DSim
    from lib.workspace import WorkspaceManager

    path = tmp_path / "diagram.diablos"
    builder.save(str(path))

    WorkspaceManager._instance = None
    dsim = DSim()
    data = dsim.file_service.load(filepath=str(path))
    assert data is not None
    dsim.file_service.apply_loaded_data(data)
    dsim.use_fast_solver = True
    dsim.zero_crossing = zero_crossing
    dsim.engine.update_sim_params(sim_time, sim_dt, zero_crossing=zero_crossing)
    if max_events is not None:
        dsim.engine.zero_crossing_max_events = max_events

    ok, err = dsim.run_tuning_simulation(sim_time, sim_dt)
    assert ok, "compiled run failed: {}".format(err)
    diagnostics = dsim.engine.get_solver_diagnostics()
    assert diagnostics.get("backend") in ("scipy", "scipy+events"), (
        "expected the compiled adaptive path, got backend={!r}".format(diagnostics.get("backend"))
    )
    return dsim, diagnostics


def _event_times(diagnostics):
    info = diagnostics.get("zero_crossing") or {}
    return [t for t, _labels in info.get("first_events", [])]


def _scope_trace(dsim):
    """The (n_samples,) trace captured by the diagram's single Scope."""
    for block in dsim.engine.active_blocks_list:
        if block.block_fn != "Scope":
            continue
        params = getattr(block, "exec_params", block.params)
        vector = params.get("vector")
        if vector is None:
            continue
        return np.asarray(vector, dtype=float).reshape(-1, params.get("vec_dim", 1)).ravel()
    raise AssertionError("no Scope captured a signal")


def _relay_loop(sim_dt, half_width=0.25):
    """Relay feedback: Integrator -> Hysteresis -> Gain(-1) -> back to Integrator.

    ``dy/dt = -relay(y)`` with a relay of +-1 and thresholds +-``half_width``.
    Starting at y = 0 latched low (the relay outputs -1 between thresholds), y
    rises at unit rate to +h, the relay flips, y falls at unit rate to -h, and
    so on: switching at t = h, 3h, 5h, ... exactly, forever.

    A *bare* sign relay has no such analytic schedule -- it reaches the
    switching surface and then slides, switching infinitely fast -- which is
    why the relay-feedback fixture has hysteresis (and why the sliding case is
    the chattering-guard test instead).
    """
    builder = DiagramBuilder(sim_time=2.0, sim_dt=sim_dt)
    _add(builder, "Integrator", "plant", params={"init_conds": 0.0})
    _add(
        builder,
        "Hysteresis",
        "relay",
        params={"upper": half_width, "lower": -half_width, "high": 1.0, "low": -1.0},
    )
    _add(builder, "Gain", "invert", params={"gain": -1.0})
    _add(builder, "Scope", "scope", params={"labels": "y"})
    builder.connect("plant", 0, "relay", 0)
    builder.connect("relay", 0, "invert", 0)
    builder.connect("invert", 0, "plant", 0)
    builder.connect("plant", 0, "scope", 0)
    return builder


# --------------------------------------------------------------------------- #
# (a) Relay feedback: switching times are exact and step-size independent
# --------------------------------------------------------------------------- #
class TestRelayFeedback:
    HALF_WIDTH = 0.25
    # Switching at h, 3h, 5h, ... while t <= 2.0.
    EXPECTED = [0.25, 0.75, 1.25, 1.75]

    def test_switching_times_are_analytic(self, qapp, tmp_path):
        _dsim, diagnostics = _run(_relay_loop(0.01), tmp_path, 2.0, 0.01)

        info = diagnostics["zero_crossing"]
        assert info["enabled"] and not info["guard_tripped"]
        assert info["n_events"] == len(self.EXPECTED)
        located = _event_times(diagnostics)
        assert located == pytest.approx(self.EXPECTED, abs=1e-6)

    def test_switching_times_do_not_depend_on_the_step(self, qapp, tmp_path):
        """The whole point: the located instant is a property of the system.

        Two unrelated output steps (and one of them not a divisor of any
        switching time) must give the same switching schedule and the same
        trajectory.  Without events the switch lands wherever the adaptive step
        happened to fall and the answers drift apart.
        """
        coarse_times = _event_times(_run(_relay_loop(0.05), tmp_path, 2.0, 0.05)[1])
        fine_times = _event_times(_run(_relay_loop(0.004), tmp_path, 2.0, 0.004)[1])

        assert coarse_times == pytest.approx(self.EXPECTED, abs=1e-6)
        assert fine_times == pytest.approx(self.EXPECTED, abs=1e-6)
        assert coarse_times == pytest.approx(fine_times, abs=1e-9)

    def test_trajectory_matches_the_analytic_triangle_wave(self, qapp, tmp_path):
        dsim, _diagnostics = _run(_relay_loop(0.01), tmp_path, 2.0, 0.01)
        t = np.asarray(dsim.engine.timeline, dtype=float)
        y = _scope_trace(dsim)

        # Unit-rate triangle wave: up from 0 to +h, then down to -h, period 4h.
        h = self.HALF_WIDTH
        phase = np.mod(t + h, 4 * h)  # 0 at the bottom of the rising edge
        expected = np.where(phase <= 2 * h, phase - h, 3 * h - phase)
        assert np.max(np.abs(y - expected)) < 1e-6


# --------------------------------------------------------------------------- #
# (b) Saturation: entering and leaving a limit
# --------------------------------------------------------------------------- #
class TestSaturation:
    def test_entering_and_leaving_the_upper_limit(self, qapp, tmp_path):
        """sin(t) clipped at 0.5 enters the limit at pi/6 and leaves at 5pi/6."""
        builder = DiagramBuilder(sim_time=3.0, sim_dt=0.01)
        _add(builder, "Sine", "src", params={"amplitude": 1.0, "frequency": 1.0})
        _add(builder, "Saturation", "sat", params={"min": -0.5, "max": 0.5})
        _add(builder, "Integrator", "acc", params={"init_conds": 0.0})
        _add(builder, "Scope", "scope", params={"labels": "y"})
        builder.connect("src", 0, "sat", 0)
        builder.connect("sat", 0, "acc", 0)
        builder.connect("acc", 0, "scope", 0)

        dsim, diagnostics = _run(builder, tmp_path, 3.0, 0.01)

        assert _event_times(diagnostics) == pytest.approx([np.pi / 6, 5 * np.pi / 6], abs=1e-6)

        # Integral of the clipped sine: sin until pi/6, flat 0.5 to 5pi/6, sin after.
        t = np.asarray(dsim.engine.timeline, dtype=float)
        t1, t2 = np.pi / 6, 5 * np.pi / 6
        y1 = 1 - np.cos(t1)
        y2 = y1 + 0.5 * (t2 - t1)
        expected = np.where(
            t <= t1,
            1 - np.cos(t),
            np.where(t <= t2, y1 + 0.5 * (t - t1), y2 + (np.cos(t2) - np.cos(t))),
        )
        assert np.max(np.abs(_scope_trace(dsim) - expected)) < 1e-6

    def test_an_infinite_limit_contributes_no_event(self, qapp, tmp_path):
        """A limit that can never be reached is not a discontinuity."""
        builder = DiagramBuilder(sim_time=1.0, sim_dt=0.01)
        _add(builder, "Sine", "src", params={"amplitude": 1.0, "frequency": 1.0})
        _add(builder, "Saturation", "sat", params={"min": -np.inf, "max": np.inf})
        _add(builder, "Integrator", "acc", params={"init_conds": 0.0})
        _add(builder, "Scope", "scope", params={"labels": "y"})
        builder.connect("src", 0, "sat", 0)
        builder.connect("sat", 0, "acc", 0)
        builder.connect("acc", 0, "scope", 0)

        _dsim, diagnostics = _run(builder, tmp_path, 1.0, 0.01)
        assert diagnostics["backend"] == "scipy"  # no events registered at all
        assert not diagnostics["zero_crossing"]["enabled"]


# --------------------------------------------------------------------------- #
# (c) Hysteresis: the relay loop closes, and it compiles only with events
# --------------------------------------------------------------------------- #
class TestHysteresis:
    def test_latch_direction_alternates(self, qapp, tmp_path):
        """A sine through a relay: rise through `upper`, fall through `lower`.

        sin(2t) reaches +0.5 at t = pi/12 and comes back down through -0.5 at
        7pi/12; the relay's guard switches which threshold it watches at each
        flip, so the located instants alternate between the two.
        """
        builder = DiagramBuilder(sim_time=4.0, sim_dt=0.01)
        _add(builder, "Sine", "src", params={"amplitude": 1.0, "frequency": 2.0})
        _add(
            builder,
            "Hysteresis",
            "relay",
            params={"upper": 0.5, "lower": -0.5, "high": 1.0, "low": 0.0},
        )
        _add(builder, "Integrator", "acc", params={"init_conds": 0.0})
        _add(builder, "Scope", "scope", params={"labels": "y"})
        builder.connect("src", 0, "relay", 0)
        builder.connect("relay", 0, "acc", 0)
        builder.connect("acc", 0, "scope", 0)

        dsim, diagnostics = _run(builder, tmp_path, 4.0, 0.01)

        expected = [np.pi / 12, 7 * np.pi / 12, 13 * np.pi / 12]
        assert _event_times(diagnostics) == pytest.approx(expected, abs=1e-6)

        # The relay output is 1 between the first and second switch and between
        # the third and the end, 0 otherwise; the integrator accumulates that.
        on_time = (expected[1] - expected[0]) + (4.0 - expected[2])
        assert _scope_trace(dsim)[-1] == pytest.approx(on_time, abs=1e-6)

    def test_hysteresis_is_interpreter_only_without_events(self, qapp):
        """Its latch cannot be a pure function of (t, y) without located events.

        With detection off the compiler must refuse the block so the diagram
        falls back to the interpreter, rather than compiling a relay whose
        latch the solver's out-of-order probing would corrupt.
        """
        from lib.lib import DSim

        engine = DSim().engine

        class _Block:
            block_fn = "Hysteresis"
            name = "relay"
            params = {}

        engine.update_sim_params(1.0, 0.01, zero_crossing=True)
        assert engine.check_compilability([_Block()]) is True

        engine.update_sim_params(1.0, 0.01, zero_crossing=False)
        assert engine.check_compilability([_Block()]) is False


# --------------------------------------------------------------------------- #
# (d) A step edge lands exactly on its own time
# --------------------------------------------------------------------------- #
class TestStepEdge:
    def test_step_at_one_second(self, qapp, tmp_path):
        builder = DiagramBuilder(sim_time=3.0, sim_dt=0.1)
        _add(builder, "Step", "src", params={"value": 1.0, "delay": 1.0, "type": "up"})
        _add(builder, "Integrator", "acc", params={"init_conds": 0.0})
        _add(builder, "Scope", "scope", params={"labels": "y"})
        builder.connect("src", 0, "acc", 0)
        builder.connect("acc", 0, "scope", 0)

        dsim, diagnostics = _run(builder, tmp_path, 3.0, 0.1)

        # The edge is located exactly, not to solver-step accuracy.
        located = _event_times(diagnostics)
        assert len(located) == 1
        assert located[0] == 1.0

        t = np.asarray(dsim.engine.timeline, dtype=float)
        y = _scope_trace(dsim)
        assert np.max(np.abs(y - np.maximum(0.0, t - 1.0))) < 1e-8
        # The ramp starts at the edge, so the sample at t = 1.0 is still zero.
        assert y[np.argmin(np.abs(t - 1.0))] == pytest.approx(0.0, abs=1e-9)

    def test_a_step_edge_needs_no_step_cap(self, qapp, tmp_path):
        """``t - delay`` is monotonic, so it cannot hide a round trip in a step.

        Marking it so keeps the max-step cap (which state-dependent guards do
        need) away from the many diagrams whose only discontinuity is a step.
        """
        from lib.engine.compiler_kernels import EVENT_BUILDERS, BuildContext

        ctx = BuildContext(
            block=None,
            b_name="src",
            fn="Step",
            params={"type": "up", "delay": 1.0},
            input_sources=[],
            deps={},
            state_map={},
            block_matrices={},
        )
        specs = EVENT_BUILDERS["Step"](ctx)
        assert [spec.monotonic for spec in specs] == [True]


# --------------------------------------------------------------------------- #
# (e) The chattering guard
# --------------------------------------------------------------------------- #
class TestChatteringGuard:
    def _sliding_relay(self, sim_dt=0.01):
        """``dy/dt = -100*sign(y)`` from y = 0.1: reaches 0 and then slides.

        Past t = 0.001 the exact solution switches infinitely often, so every
        event restart lands on top of the previous one.  Nothing bounds this on
        its own -- and an adaptive solver with events switched off does not
        finish either, since its error control just keeps shrinking the step.
        """
        builder = DiagramBuilder(sim_time=1.0, sim_dt=sim_dt)
        _add(builder, "MathFunction", "sgn", params={"function": "sign"})
        _add(builder, "Gain", "gain", params={"gain": -100.0})
        _add(builder, "Integrator", "acc", params={"init_conds": 0.1})
        _add(builder, "Scope", "scope", params={"labels": "y"})
        builder.connect("acc", 0, "sgn", 0)
        builder.connect("sgn", 0, "gain", 0)
        builder.connect("gain", 0, "acc", 0)
        builder.connect("acc", 0, "scope", 0)
        return builder

    def test_guard_trips_and_the_run_still_finishes(self, qapp, tmp_path):
        started = time.perf_counter()
        dsim, diagnostics = _run(self._sliding_relay(), tmp_path, 1.0, 0.01, max_events=500)
        elapsed = time.perf_counter() - started

        info = diagnostics["zero_crossing"]
        assert info["guard_tripped"]
        assert "chattering" in info["guard_reason"]
        # Bounded work, not a hang: the guard stops well short of its own cap.
        assert info["n_events"] < 500
        assert elapsed < 30.0

        # The output grid is intact -- the fixed-step fallback fills the tail.
        t = np.asarray(dsim.engine.timeline, dtype=float)
        assert t[0] == pytest.approx(0.0)
        assert t[-1] == pytest.approx(1.0)
        assert len(_scope_trace(dsim)) == len(t)
        assert np.all(np.isfinite(_scope_trace(dsim)))

    def test_event_cap_also_trips_the_guard(self, qapp, tmp_path):
        """A low cap is the second bound, for chattering too gradual to streak."""
        _dsim, diagnostics = _run(self._sliding_relay(), tmp_path, 1.0, 0.01, max_events=5)
        info = diagnostics["zero_crossing"]
        assert info["guard_tripped"]
        assert info["n_events"] == 5
        assert "event cap" in info["guard_reason"]


# --------------------------------------------------------------------------- #
# (f) The setting off reproduces the pre-existing single-shot solve
# --------------------------------------------------------------------------- #
class TestDisabled:
    def test_off_matches_a_plain_solve_ivp(self, qapp, tmp_path):
        """With the setting off the compiled path must be the old code path.

        Not "close to" it: the same compiled RHS handed to one ``solve_ivp``
        call over the same grid, so the states must agree exactly.
        """
        from scipy.integrate import solve_ivp

        builder = DiagramBuilder(sim_time=3.0, sim_dt=0.01)
        _add(builder, "Sine", "src", params={"amplitude": 1.0, "frequency": 1.0})
        _add(builder, "Saturation", "sat", params={"min": -0.5, "max": 0.5})
        _add(builder, "Integrator", "acc", params={"init_conds": 0.0})
        _add(builder, "Scope", "scope", params={"labels": "y"})
        builder.connect("src", 0, "sat", 0)
        builder.connect("sat", 0, "acc", 0)
        builder.connect("acc", 0, "scope", 0)

        dsim, diagnostics = _run(builder, tmp_path, 3.0, 0.01, zero_crossing=False)
        assert diagnostics["backend"] == "scipy"
        assert not diagnostics["zero_crossing"]["enabled"]

        blocks = dsim.engine.active_blocks_list
        lines = dsim.engine.active_line_list or dsim.line_list
        model_func, y0, _state_map, _matrices = dsim.engine.compiler.compile_system(
            blocks, sorted(blocks, key=lambda b: b.hierarchy), lines
        )
        reference = solve_ivp(
            model_func,
            (0.0, 3.0),
            y0,
            t_eval=np.asarray(dsim.engine.timeline, dtype=float),
            method="RK45",
            rtol=dsim.engine.rtol,
            atol=dsim.engine.atol,
        )
        assert np.array_equal(dsim.engine.outs, reference.y)

    def test_a_block_can_opt_out_on_its_own(self, qapp, tmp_path):
        """The per-block param drops that block's events, not the diagram's."""
        builder = DiagramBuilder(sim_time=3.0, sim_dt=0.01)
        _add(builder, "Sine", "src", params={"amplitude": 1.0, "frequency": 1.0})
        _add(
            builder,
            "Saturation",
            "sat",
            params={"min": -0.5, "max": 0.5, "zero_crossing": "off"},
        )
        _add(builder, "Step", "kick", params={"value": 1.0, "delay": 1.0, "type": "up"})
        _add(builder, "Sum", "mix", params={"sign": "++"})
        _add(builder, "Integrator", "acc", params={"init_conds": 0.0})
        _add(builder, "Scope", "scope", params={"labels": "y"})
        builder.connect("src", 0, "sat", 0)
        builder.connect("sat", 0, "mix", 0)
        builder.connect("kick", 0, "mix", 1)
        builder.connect("mix", 0, "acc", 0)
        builder.connect("acc", 0, "scope", 0)

        _dsim, diagnostics = _run(builder, tmp_path, 3.0, 0.01)
        info = diagnostics["zero_crossing"]
        assert info["enabled"]
        labels = {label for _t, group in info["first_events"] for label in group}
        assert labels == {"step2:edge"}, "the opted-out Saturation still fired events"


# --------------------------------------------------------------------------- #
# (g) Every compilable example: events on vs off
# --------------------------------------------------------------------------- #
def _example_files():
    return sorted(p.name for p in EXAMPLES_DIR.glob("*.diablos"))


@pytest.mark.slow
@pytest.mark.parametrize("filename", _example_files())
def test_examples_agree_with_events_on_and_off(filename, qapp):
    """No example may change shape -- or, without discontinuities, value.

    A diagram that registers no event functions has nothing for the machinery
    to do, so its output must be bit-for-bit what it was before.  One that does
    register events legitimately moves (that is the fix), but only to solver
    tolerance, and never in shape.
    """
    from lib.lib import DSim
    from lib.workspace import WorkspaceManager

    def run(zero_crossing):
        WorkspaceManager._instance = None
        dsim = DSim()
        data = dsim.file_service.load(filepath=str(EXAMPLES_DIR / filename))
        assert data is not None, "failed to load {}".format(filename)
        dsim.file_service.apply_loaded_data(data)
        dsim.use_fast_solver = True
        dsim.zero_crossing = zero_crossing
        if not dsim.engine.check_compilability(dsim.blocks_list):
            return None
        ok, err = dsim.run_tuning_simulation(dsim.sim_time, dsim.sim_dt)
        assert ok, "{} failed with zero_crossing={}: {}".format(filename, zero_crossing, err)
        if dsim.engine.get_solver_diagnostics().get("backend") not in (
            "scipy",
            "scipy+events",
        ):
            return None  # algebraic or fixed-step: no adaptive solve to compare
        return dsim

    with_events = run(True)
    if with_events is None:
        pytest.skip("{} does not reach the compiled adaptive solver".format(filename))
    without_events = run(False)
    if without_events is None:
        # Hysteresis is in SystemCompiler.ZERO_CROSSING_ONLY_BLOCKS: a diagram
        # containing one compiles only while events are available, and drops to
        # the interpreter when they are switched off. The two configurations
        # then run different engines, so there is no adaptive solve to compare
        # (examples/relay_thermostat_events.diablos, nonlinear_blocks.diablos).
        pytest.skip("{} leaves the compiled path when zero-crossing is off".format(filename))

    info = with_events.engine.get_solver_diagnostics()["zero_crossing"] or {}
    assert not info.get("guard_tripped"), "{} tripped the chattering guard".format(filename)

    on = np.asarray(with_events.engine.outs, dtype=float)
    off = np.asarray(without_events.engine.outs, dtype=float)
    assert on.shape == off.shape, "{}: event stitching changed the output shape".format(filename)
    assert np.array_equal(
        np.asarray(with_events.engine.timeline), np.asarray(without_events.engine.timeline)
    )

    if not info.get("enabled"):
        # No discontinuous blocks: the two runs are literally the same call.
        assert np.array_equal(on, off), "{} changed without any events".format(filename)
    else:
        scale = max(1.0, float(np.max(np.abs(off))) if off.size else 1.0)
        assert np.max(np.abs(on - off)) < 1e-4 * scale, (
            "{}: events moved the trajectory far more than solver tolerance".format(filename)
        )
