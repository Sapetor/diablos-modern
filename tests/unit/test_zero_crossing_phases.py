"""Pin the phase helpers ``solve_with_events`` is built from.

The segmented event driver in ``lib/engine/zero_crossing.py`` is an
orchestrator over small private helpers (step cap, mode seeding, gap fill, one
segment, discrete updates, event record, chattering guard, restart point,
guard trip, tail truncation).  Each is pinned here in isolation so a change
to one phase fails next to the phase, not somewhere in a 30-segment relay run.
The end-to-end behaviour is covered by ``tests/regression/test_zero_crossing.py``
and ``tests/validation/test_events.py``.
"""

import logging

import numpy as np
import pytest

from lib.engine import zero_crossing as zc
from lib.engine.zero_crossing import (
    CHATTER_STREAK_LIMIT,
    EventSolveResult,
    EventSpec,
    _apply_discrete_updates,
    _disable_events,
    _fill_gap_samples,
    _record_event,
    _restart_point,
    _run_segment,
    _seed_modes,
    _SegmentLoop,
    _SignalCache,
    _step_cap,
    _truncate_to_produced,
    _update_chatter_guard,
    solve_with_events,
)


class _Model:
    """Compiled-RHS stand-in: ``(t, y) -> dy`` with ``.evaluate -> (dy, signals)``."""

    def __init__(self, rhs):
        self._rhs = rhs
        self.calls = 0

    def evaluate(self, t, y):
        self.calls += 1
        y = np.asarray(y, dtype=float)
        dy = np.asarray(self._rhs(t, y), dtype=float)
        return dy, {"x": float(y[0]), "t": float(t)}

    def __call__(self, t, y):
        return self.evaluate(t, y)[0]


def _spec(label="g", **kw):
    kw.setdefault("func", lambda t, y, s: s["x"] - 0.5)
    return EventSpec("Blk", label, **kw)


def _holder(init, event_driven=True):
    return {"mode": init, "init": init, "frozen": False, "event_driven": event_driven}


def _result(n_states, n_points):
    result = EventSolveResult()
    result.t = np.arange(n_points, dtype=float)
    result.y = np.zeros((n_states, n_points))
    return result


@pytest.mark.unit
class TestStepCap:
    def test_unbounded_without_max_step(self):
        assert _step_cap(None, [_spec()]) == np.inf
        assert _step_cap(0.0, [_spec()]) == np.inf
        assert _step_cap(np.inf, [_spec()]) == np.inf
        assert _step_cap(-0.1, [_spec()]) == np.inf

    def test_capped_when_any_event_is_non_monotonic(self):
        specs = [_spec("a", monotonic=True), _spec("b", monotonic=False)]
        assert _step_cap(0.01, specs) == 0.01

    def test_uncapped_when_every_event_is_monotonic(self):
        specs = [_spec("a", monotonic=True), _spec("b", monotonic=True)]
        assert _step_cap(0.01, specs) == np.inf


@pytest.mark.unit
class TestSeedModes:
    def test_no_hooks_costs_no_evaluation(self):
        model = _Model(lambda t, y: -y)
        cache = _SignalCache(model.evaluate)
        _seed_modes([_spec()], cache, 0.0, np.array([1.0]))
        assert model.calls == 0

    def test_hooks_share_one_evaluation_and_cache_is_dropped(self):
        model = _Model(lambda t, y: -y)
        cache = _SignalCache(model.evaluate)
        seen = []
        specs = [
            _spec("a", on_start=lambda t, y, s: seen.append(("a", t, s["x"]))),
            _spec("b"),
            _spec("c", on_start=lambda t, y, s: seen.append(("c", t, s["x"]))),
        ]
        _seed_modes(specs, cache, 0.0, np.array([0.9]))
        assert seen == [("a", 0.0, 0.9), ("c", 0.0, 0.9)]
        assert model.calls == 1
        # The cache was invalidated: the next probe evaluates again.
        cache.signals(0.0, np.array([0.9]))
        assert model.calls == 2

    def test_failing_hook_is_swallowed(self, caplog):
        model = _Model(lambda t, y: -y)
        cache = _SignalCache(model.evaluate)

        def boom(t, y, s):
            raise RuntimeError("boom")

        with caplog.at_level(logging.DEBUG, logger=zc.__name__):
            _seed_modes([_spec(on_start=boom)], cache, 0.0, np.array([1.0]))
        assert "seeding failed" in caplog.text


@pytest.mark.unit
class TestFillGapSamples:
    def test_writes_carried_state_to_points_before_t_start(self):
        t_eval = np.array([0.0, 0.1, 0.2, 0.3])
        result = _result(2, 4)
        idx = _fill_gap_samples(result, t_eval, 1, 0.25, np.array([7.0, -1.0]))
        assert idx == 3
        assert np.array_equal(result.y[:, 1], [7.0, -1.0])
        assert np.array_equal(result.y[:, 2], [7.0, -1.0])
        assert np.array_equal(result.y[:, 0], [0.0, 0.0])
        assert np.array_equal(result.y[:, 3], [0.0, 0.0])

    def test_point_equal_to_t_start_is_left_for_the_solver(self):
        t_eval = np.array([0.0, 0.1, 0.2])
        result = _result(1, 3)
        assert _fill_gap_samples(result, t_eval, 1, 0.1, np.array([5.0])) == 1
        assert result.y[0, 1] == 0.0

    def test_runs_off_the_end(self):
        t_eval = np.array([0.0, 0.1])
        result = _result(1, 2)
        assert _fill_gap_samples(result, t_eval, 0, 1.0, np.array([5.0])) == 2
        assert np.array_equal(result.y[0], [5.0, 5.0])


@pytest.mark.unit
class TestRunSegment:
    class _Sol:
        def __init__(self, t, y, **attrs):
            self.t = t
            self.y = y
            self.success = True
            self.status = 0
            self.message = "done"
            self.__dict__.update(attrs)

    def test_writes_samples_and_folds_counters(self):
        seen = {}

        def fake_solve_ivp(f, span, y0, **kw):
            seen.update(kw, span=span, y0=y0)
            return self._Sol(
                np.array([0.2, 0.3]), np.array([[2.0, 3.0]]), nfev=7, njev=1, nlu=2, status=0
            )

        t_eval = np.array([0.0, 0.1, 0.2, 0.3])
        result = _result(1, 4)
        loop = _SegmentLoop(idx=2, t_start=0.15, y_start=np.array([1.5]))
        events = [object()]
        sol = _run_segment(
            fake_solve_ivp, None, 1.0, t_eval, loop, result, "RK45", 1e-6, 1e-9, 0.01, events
        )
        assert sol.status == 0
        assert loop.idx == 4
        assert np.array_equal(result.y[0], [0.0, 0.0, 2.0, 3.0])
        assert (result.n_segments, result.nfev, result.njev, result.nlu) == (1, 7, 1, 2)
        assert result.message == "done" and result.status == 0
        assert seen["span"] == (0.15, 1.0)
        assert np.array_equal(seen["t_eval"], [0.2, 0.3])
        assert seen["max_step"] == 0.01 and seen["events"] is events
        assert (seen["method"], seen["rtol"], seen["atol"]) == ("RK45", 1e-6, 1e-9)

    def test_events_off_means_no_cap_and_no_events(self):
        seen = {}

        def fake_solve_ivp(f, span, y0, **kw):
            seen.update(kw)
            return self._Sol([], [], nfev=None)

        result = _result(1, 2)
        loop = _SegmentLoop(idx=0, t_start=0.0, y_start=np.array([1.0]), events_active=False)
        _run_segment(
            fake_solve_ivp,
            None,
            1.0,
            np.array([0.0, 1.0]),
            loop,
            result,
            "RK45",
            1e-6,
            1e-9,
            0.01,
            [object()],
        )
        assert seen["max_step"] == np.inf and seen["events"] is None
        # Empty scipy lists: nothing produced, counters tolerate None.
        assert loop.idx == 0 and result.nfev == 0 and result.n_segments == 1


@pytest.mark.unit
class TestApplyDiscreteUpdates:
    def test_hooks_see_signals_at_the_root_once(self):
        model = _Model(lambda t, y: -y)
        cache = _SignalCache(model.evaluate)
        cache.signals(0.0, np.array([0.0]))  # stale entry that must be dropped
        seen = []
        fired = [
            _spec("a", on_event=lambda t, y, s: seen.append(("a", t, s["x"]))),
            _spec("b"),
            _spec("c", on_event=lambda t, y, s: seen.append(("c", t, s["x"]))),
        ]
        _apply_discrete_updates(fired, 0.5, np.array([0.25]), cache)
        assert seen == [("a", 0.5, 0.25), ("c", 0.5, 0.25)]
        assert model.calls == 2  # the stale probe plus exactly one at the root
        cache.signals(0.5, np.array([0.25]))
        assert model.calls == 3  # invalidated afterwards

    def test_hook_may_reset_the_state_in_place(self):
        model = _Model(lambda t, y: np.array([y[1], -9.81]))
        cache = _SignalCache(model.evaluate)

        def bounce(t, y, s):
            y[1] = -0.8 * y[1]

        y_event = np.array([0.0, -4.0])
        _apply_discrete_updates([_spec(on_event=bounce)], 1.0, y_event, cache)
        assert np.isclose(y_event[1], 3.2)

    def test_failing_hook_is_logged_not_raised(self, caplog):
        model = _Model(lambda t, y: -y)
        cache = _SignalCache(model.evaluate)

        def boom(t, y, s):
            raise RuntimeError("boom")

        with caplog.at_level(logging.WARNING, logger=zc.__name__):
            _apply_discrete_updates([_spec(on_event=boom)], 0.5, np.array([1.0]), cache)
        assert "Discrete update for event Blk:g failed" in caplog.text


@pytest.mark.unit
class TestRecordEvent:
    def test_appends_log_history_and_count(self):
        result = EventSolveResult()
        holders = [_holder(True), _holder(0)]
        holders[0]["mode"] = False
        _record_event(result, holders, np.float64(0.5), [_spec("a"), _spec("b")])
        assert result.event_log == [(0.5, ["Blk:a", "Blk:b"])]
        assert result.mode_history == [(0.5, [False, 0])]
        assert result.n_events == 1
        assert type(result.event_log[0][0]) is float

    def test_no_holders_gives_empty_snapshot(self):
        result = EventSolveResult()
        _record_event(result, None, 0.5, [])
        assert result.mode_history == [(0.5, [])]


@pytest.mark.unit
class TestUpdateChatterGuard:
    def test_streak_counts_only_consecutive_close_events(self):
        result = EventSolveResult()
        loop = _SegmentLoop(idx=0, t_start=0.0, y_start=np.zeros(1))
        _update_chatter_guard(result, loop, 1.0, [], 1e-9, 100)
        assert (loop.chatter_streak, loop.last_event_t) == (0, 1.0)
        _update_chatter_guard(result, loop, 1.0 + 1e-12, [], 1e-9, 100)
        assert loop.chatter_streak == 1
        _update_chatter_guard(result, loop, 1.0 + 2e-12, [], 1e-9, 100)
        assert loop.chatter_streak == 2
        _update_chatter_guard(result, loop, 2.0, [], 1e-9, 100)  # far apart: reset
        assert loop.chatter_streak == 0
        assert not result.guard_tripped

    def test_trips_on_streak_limit_naming_the_events(self):
        result = EventSolveResult()
        loop = _SegmentLoop(idx=0, t_start=0.0, y_start=np.zeros(1))
        # The first event starts the streak at 0; each following event closer
        # than min_separation adds one, so the limit needs LIMIT + 1 events.
        t = 1.0
        for _ in range(CHATTER_STREAK_LIMIT + 1):
            assert not result.guard_tripped
            _update_chatter_guard(result, loop, t, [_spec("zero")], 1e-9, 10**6)
            t += 1e-12
        assert result.guard_tripped
        assert loop.chatter_streak == CHATTER_STREAK_LIMIT
        assert result.guard_reason.startswith(
            "chattering: {} consecutive".format(CHATTER_STREAK_LIMIT)
        )
        assert "Blk:zero" in result.guard_reason

    def test_trips_on_event_cap(self):
        result = EventSolveResult()
        result.n_events = 3
        loop = _SegmentLoop(idx=0, t_start=0.0, y_start=np.zeros(1))
        _update_chatter_guard(result, loop, 0.75, [], 1e-9, 3)
        assert result.guard_tripped
        assert result.guard_reason == "event cap of 3 reached at t=0.75s"

    def test_streak_takes_precedence_over_cap(self):
        result = EventSolveResult()
        result.n_events = 10**6
        loop = _SegmentLoop(
            idx=0,
            t_start=0.0,
            y_start=np.zeros(1),
            last_event_t=1.0,
            chatter_streak=CHATTER_STREAK_LIMIT - 1,
        )
        _update_chatter_guard(result, loop, 1.0 + 1e-12, [_spec()], 1e-9, 1)
        assert result.guard_reason.startswith("chattering")


@pytest.mark.unit
class TestRestartPoint:
    def test_nudges_time_and_takes_one_euler_step(self):
        model = _Model(lambda t, y: np.array([2.0]))
        t_start, y_start = _restart_point(model, 1.0, np.array([0.5]), 1e-11)
        assert t_start == 1.0 + 1e-11
        assert np.array_equal(y_start, np.array([0.5]) + (t_start - 1.0) * 2.0)

    def test_lost_nudge_falls_back_to_nextafter(self):
        model = _Model(lambda t, y: np.array([1.0]))
        t_event = 1e17  # ulp is 16, so a 1e-11 nudge is lost to rounding
        t_start, y_start = _restart_point(model, t_event, np.array([0.0]), 1e-11)
        assert t_start == np.nextafter(t_event, np.inf)
        assert t_start > t_event
        assert np.array_equal(y_start, [t_start - t_event])

    def test_unusable_rhs_keeps_the_state(self):
        model = _Model(lambda t, y: np.array([np.nan]))
        _, y_start = _restart_point(model, 1.0, np.array([0.5]), 1e-11)
        assert np.array_equal(y_start, [0.5])


@pytest.mark.unit
class TestDisableEvents:
    def test_records_instant_unfreezes_holders_and_warns(self, caplog):
        result = EventSolveResult()
        result.guard_reason = "event cap of 3 reached at t=0.75s"
        holders = [_holder(True), _holder(False, event_driven=False)]
        for h in holders:
            h["frozen"] = True
        loop = _SegmentLoop(idx=5, t_start=np.float64(0.75), y_start=np.zeros(1))
        with caplog.at_level(logging.WARNING, logger=zc.__name__):
            _disable_events(result, loop, holders)
        assert loop.events_active is False
        assert result.events_off_at == 0.75 and type(result.events_off_at) is float
        assert all(h["frozen"] is False for h in holders)
        assert holders[0]["mode"] is True  # values untouched
        assert "event cap of 3 reached" in caplog.text
        assert "Zero-crossing detection disabled" in caplog.text


@pytest.mark.unit
class TestTruncateToProduced:
    def test_short_run_is_cut_to_the_produced_columns(self):
        t_eval = np.linspace(0.0, 1.0, 5)
        result = _result(2, 5)
        result.t = t_eval
        _truncate_to_produced(result, t_eval, 3)
        assert result.t.shape == (3,) and result.y.shape == (2, 3)

    def test_complete_run_is_untouched(self):
        t_eval = np.linspace(0.0, 1.0, 5)
        result = _result(2, 5)
        result.t = t_eval
        _truncate_to_produced(result, t_eval, 5)
        assert result.t is t_eval and result.y.shape == (2, 5)


@pytest.mark.unit
class TestOrchestrator:
    """The phases stitched together on a problem with a known answer."""

    def test_bouncing_ball_bounces_where_it_should(self):
        model = _Model(lambda t, y: np.array([y[1], -9.81]))

        def bounce(t, y, s):
            y[1] = -0.5 * y[1]

        spec = EventSpec("Ball", "ground", lambda t, y, s: s["x"], direction=-1.0, on_event=bounce)
        t_eval = np.linspace(0.0, 1.0, 101)
        res = solve_with_events(
            model, (0.0, 1.0), [1.0, 0.0], t_eval, [spec], "RK45", 1e-8, 1e-10, max_step=0.01
        )
        assert res.success and res.status == 0
        assert res.t.shape == (101,) and res.y.shape == (2, 101)
        t_first = np.sqrt(2.0 / 9.81)
        assert res.n_events >= 2 and res.n_segments == res.n_events + 1
        assert np.isclose(res.event_log[0][0], t_first, atol=1e-7)
        assert res.event_log[0][1] == ["Ball:ground"]
        assert len(res.mode_history) == res.n_events + 1
        assert res.y[0].min() > -1e-6  # never far below the floor

    def test_guard_trip_hands_the_tail_to_the_fallback(self):
        model = _Model(lambda t, y: np.array([-np.sign(y[0]) - 0.5 * y[0]]))
        spec = EventSpec("Sign", "zero", lambda t, y, s: s["x"], direction=0.0)
        t_eval = np.linspace(0.0, 3.0, 301)
        calls = []

        def fallback(f, grid, y0, scheme):
            calls.append((grid[0], scheme))
            return np.repeat(np.asarray(y0, dtype=float)[:, None], len(grid), axis=1)

        res = solve_with_events(
            model,
            (0.0, 3.0),
            [1.0],
            t_eval,
            [spec],
            "RK45",
            1e-6,
            1e-9,
            max_step=0.01,
            fallback_integrator=fallback,
        )
        assert res.guard_tripped and res.guard_reason.startswith("chattering")
        assert res.events_off_at is not None and calls and calls[0][1] == "rk4"
        assert np.isclose(calls[0][0], res.events_off_at)
        assert res.t.shape == (301,) and res.y.shape == (1, 301)
        assert res.message.startswith("Finished with a fixed step")
        assert res.n_events == CHATTER_STREAK_LIMIT + 1
