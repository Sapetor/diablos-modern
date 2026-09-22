"""Zero-crossing (event) detection for the compiled (fast-solver) path.

The compiled path assembles the whole diagram into a single ODE right-hand side
``f(t, y)`` and hands it to ``scipy.integrate.solve_ivp``.  Discontinuous blocks
(Saturation, Switch, Deadband, Hysteresis, Step, ...) make that RHS piecewise:
an adaptive Runge-Kutta step that straddles a switching instant sees a kink or a
jump it cannot represent, so it either smears the discontinuity over the step or
burns a pile of rejected steps shrinking around it.

The fix is the standard hybrid-system one: give the solver a set of scalar
*event functions* ``g(t, y)`` whose sign change marks a switching instant, stop
integration exactly at the root, apply any discrete update (a relay latch, say),
and restart a fresh integration from there.  Each continuous segment is then
genuinely smooth and the solver's error control is meaningful again.

Layout of this module:

* :class:`EventSpec` -- one scalar event contributed by one block.  Kernels in
  ``lib.engine.compiler_kernels`` build these through the ``@events(...)``
  registry, so a new discontinuous block adds events without touching the
  runner.
* :class:`_SignalCache` -- one-entry memo around the compiled evaluator so all
  event functions at the same ``(t, y)`` share a single RHS evaluation.
* :func:`solve_with_events` -- the segmented driver: solve, stop on a terminal
  event, apply discrete updates, restart, and stitch the segments back onto the
  caller's ``t_eval`` grid so the returned ``(t, y)`` has exactly the shape a
  plain ``solve_ivp`` call would have produced.

Chattering guard
----------------
A relay in sliding mode switches infinitely often in finite time; naive event
restarts then never reach ``t_end``.  Two limits bound the work: a minimum
separation between consecutive events (a streak of closer-together events is
chattering) and a hard cap on the event count.  Tripping either logs a warning
and finishes the run with a fixed step instead -- degraded accuracy, never a
hang (an adaptive solver turned loose on a chattering system does not finish at
all: with no event to stop at, its error control drives the step toward machine
precision).
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Guard / restart tuning.  All the time-like constants are fractions of the
# simulation span so they behave the same for a 1 ms and a 1000 s run.
# --------------------------------------------------------------------------- #

#: Hard cap on located events per run before the guard trips.
DEFAULT_MAX_EVENTS = 10000

#: Consecutive too-close events that count as chattering.
CHATTER_STREAK_LIMIT = 20

#: Events closer than this fraction of the span are "too close".
MIN_SEPARATION_FRACTION = 1e-9

#: How far past a located event integration restarts.  Without a nudge the
#: restart sees ``g(t_restart) == 0`` and scipy immediately re-detects the same
#: crossing (``find_active_events`` treats a zero as active), which would spin
#: forever at one instant.  The state is carried across the gap by one explicit
#: Euler step (see ``_state_at_restart``), so a guard written on the *state* is
#: also strictly past its root at the restart; the error is the O(nudge^2) of
#: that single step, ~1e-22 of the span.
RESTART_NUDGE_FRACTION = 1e-11


@dataclass
class EventSpec:
    """One scalar event function contributed by one compiled block.

    Attributes:
        block: Owning block name (diagnostics only).
        label: Short description of *which* crossing this is ("upper_limit").
        func: ``(t, y, signals) -> float``.  ``signals`` is the compiled
            evaluator's signal dict for this exact ``(t, y)``, so an event sees
            precisely the values the RHS saw -- including the three-group
            (sources -> algebraic -> D=0 state) execution order.
        direction: ``+1`` rising, ``-1`` falling, ``0`` either (scipy semantics).
        on_event: Optional ``(t, y, signals) -> None`` discrete update applied
            at the located root before integration restarts (relay latch flip,
            sample-and-hold capture, ...).
        on_start: Optional ``(t, y, signals) -> None`` run once at ``t0`` to
            reconcile a discrete mode with the actual initial condition (a relay
            whose input already sits past a threshold starts latched, not at its
            nominal default).
        monotonic: True when ``g`` is monotonic in ``t`` and so cannot change
            sign twice inside one step -- a pure time event like a Step edge
            (``t - delay``).  ``solve_ivp`` only compares event signs at
            accepted step ends, so a *non*-monotonic ``g`` can dip through zero
            and back inside a single large step and be missed entirely; the
            driver caps the step size when any such event is present.  Marking a
            genuinely monotonic event keeps that cap (and its cost) away from
            diagrams that only have step/ramp edges.
    """

    block: str
    label: str
    func: Callable[[float, np.ndarray, Dict[str, Any]], float]
    direction: float = 0.0
    on_event: Optional[Callable[[float, np.ndarray, Dict[str, Any]], None]] = None
    on_start: Optional[Callable[[float, np.ndarray, Dict[str, Any]], None]] = None
    monotonic: bool = False

    @property
    def name(self) -> str:
        return "{}:{}".format(self.block, self.label)


def signal_scalar(signals: Dict[str, Any], key: Optional[str], default: float = 0.0) -> float:
    """Read ``signals[key]`` as a scalar float.

    Event functions must be scalar, but compiled signals are freely scalar or
    vector (a Mux output, a length-1 Constant, ...).  Mirrors the reduction the
    scalar kernels already do (``np.ravel(v)[0]``) so an event never disagrees
    with the block whose switching it is watching.
    """
    if not key:
        return float(default)
    val = signals.get(key, default)
    arr = np.ravel(np.asarray(val, dtype=float))
    if arr.size == 0:
        return float(default)
    return float(arr[0])


class _SignalCache:
    """One-entry ``(t, y) -> signals`` memo around the compiled evaluator.

    ``solve_ivp`` calls every event function in turn at the same ``(t, y)``.
    Without this each would trigger its own full-diagram evaluation; with it a
    diagram contributing N events costs one evaluation per probe instead of N.
    """

    __slots__ = ("_evaluate", "_t", "_y", "_signals")

    def __init__(self, evaluate: Callable):
        self._evaluate = evaluate
        self._t = None
        self._y = None
        self._signals = None

    def signals(self, t, y) -> Dict[str, Any]:
        y = np.asarray(y, dtype=float)
        if (
            self._t is not None
            and t == self._t
            and self._y is not None
            and self._y.shape == y.shape
            and np.array_equal(self._y, y)
        ):
            return self._signals
        _, signals = self._evaluate(t, y)
        self._t = t
        self._y = y.copy()
        self._signals = signals
        return signals

    def invalidate(self) -> None:
        self._t = None
        self._y = None
        self._signals = None


class EventSolveResult:
    """``solve_ivp``-shaped result for a segmented (event-driven) solve.

    Carries the attributes ``SimulationEngine`` and
    ``lib.engine.solver_diagnostics`` read off a scipy result, plus the
    event bookkeeping this module adds.
    """

    __slots__ = (
        "t",
        "y",
        "success",
        "message",
        "status",
        "nfev",
        "njev",
        "nlu",
        "event_log",
        "n_events",
        "n_segments",
        "guard_tripped",
        "guard_reason",
        "mode_history",
        "events_off_at",
    )

    def __init__(self):
        self.t = None
        self.y = None
        self.success = True
        self.message = ""
        self.status = 0
        self.nfev = 0
        self.njev = 0
        self.nlu = 0
        self.event_log: List[Tuple[float, List[str]]] = []
        self.n_events = 0
        self.n_segments = 0
        self.guard_tripped = False
        self.guard_reason = None
        # (t, [mode per holder in mode_states order]) after seeding at t0 and
        # after every located event; the post-solve replay walks this so a
        # latched block's recorded output switches exactly where the solve did.
        self.mode_history: List[Tuple[float, List[Any]]] = []
        # Instant from which latches free-ran again (chattering guard), or None.
        self.events_off_at: Optional[float] = None

    def summary(self) -> Dict[str, Any]:
        """Compact dict for the run diagnostics / log line."""
        return {
            "enabled": True,
            "n_events": int(self.n_events),
            "n_segments": int(self.n_segments),
            "guard_tripped": bool(self.guard_tripped),
            "guard_reason": self.guard_reason,
            # Keep the log bounded: the first few instants are what a user
            # needs to see; the count above says how many there were in total.
            "first_events": [(float(t), list(labels)) for t, labels in self.event_log[:10]],
        }


def _make_event_callable(spec: EventSpec, cache: _SignalCache):
    """Wrap an EventSpec as the ``event(t, y) -> float`` scipy expects."""

    def _event(t, y, _spec=spec, _cache=cache):
        try:
            return float(_spec.func(t, y, _cache.signals(t, y)))
        except Exception:  # noqa: BLE001 - an event must never break a solve
            logger.debug("Event %s failed to evaluate at t=%s", _spec.name, t, exc_info=True)
            # A constant keeps the sign flat so no spurious crossing is seen.
            return 1.0

    _event.terminal = True
    _event.direction = float(spec.direction)
    return _event


def _reset_mode_states(mode_states, frozen: bool) -> None:
    """Restore every latch/mode holder to its initial value.

    Compiled closures are cached across runs (``_compile_system_cached``), so a
    latch holding last run's mode would leak into the next one.  ``frozen``
    says whether the mode is under event control (updated only at located
    roots) or free-running (legacy behaviour, used after a guard trip).

    Only a holder whose block actually contributed an event is frozen: a block
    that opted out of zero-crossing keeps its latch free-running even while
    other blocks in the same diagram are event-driven.
    """
    for holder in mode_states or ():
        holder["mode"] = holder.get("init")
        holder["frozen"] = bool(frozen) and bool(holder.get("event_driven", False))


@dataclass
class _SegmentLoop:
    """Mutable state the segmented driver carries from one segment to the next.

    Attributes:
        idx: Next unfilled column of the output grid.
        t_start, y_start: Where the next segment starts (a nudge past the
            last root, with the state carried across the gap).
        events_active: False once the chattering guard has tripped.
        last_event_t: Instant of the previous located root, or None.
        chatter_streak: Consecutive roots closer than the minimum separation.
    """

    idx: int
    t_start: float
    y_start: np.ndarray
    events_active: bool = True
    last_event_t: Optional[float] = None
    chatter_streak: int = 0


def solve_with_events(
    model_func,
    t_span,
    y0,
    t_eval,
    specs: List[EventSpec],
    method: str,
    rtol: float,
    atol: float,
    max_events: int = DEFAULT_MAX_EVENTS,
    mode_states=None,
    max_step=None,
    fallback_integrator=None,
    solve_ivp=None,
) -> EventSolveResult:
    """Integrate ``model_func`` with terminal zero-crossing events.

    Integration proceeds in segments.  Each segment runs to ``t_span[1]`` with
    every event marked terminal; if one fires, the root is the segment end, the
    discrete updates of all events that fired at that instant are applied, and
    the next segment restarts a hair past the root with the same (continuous)
    state.  Every segment writes its share of the caller's ``t_eval`` grid, so
    the returned ``t`` is exactly ``t_eval`` and ``y`` has the same
    ``(n_states, len(t_eval))`` shape a plain ``solve_ivp`` would return --
    replay, scope capture, CSV/NPZ export and plotting see no difference.

    The phases, in order, are the private helpers below: ``_step_cap`` (cap the
    solver stride when a non-monotonic event is present), ``_seed_modes`` (run
    the ``on_start`` hooks at ``t0``), then per segment ``_fill_gap_samples``
    (grid points inside a restart nudge), ``_run_segment`` (one ``solve_ivp``
    call, output written in place), ``_resolve_event`` (earliest root and every
    event at that instant), ``_apply_discrete_updates`` (``on_event`` hooks),
    ``_record_event`` (log, mode snapshot, count), ``_update_chatter_guard``
    (streak / cap bookkeeping), ``_restart_point`` (nudged ``t`` and the state
    one Euler step past the root) and, once the guard trips, ``_disable_events``
    followed by ``_finish_fixed_step``.

    Args:
        model_func: Compiled RHS with an ``.evaluate(t, y)`` attribute (see
            ``SystemCompiler.compile_system``).
        t_span: ``(t0, t_end)``.
        y0: Initial state.
        t_eval: Output grid (must be sorted and inside ``t_span``).
        specs: Event functions to watch.  Must be non-empty.
        method: A scipy adaptive method name (RK45, LSODA, ...).
        rtol, atol: Solver tolerances.
        max_events: Hard cap before the chattering guard trips.
        mode_states: Latch/mode holder dicts to reset (and unfreeze on a guard
            trip).
        max_step: Cap on the solver's internal step, applied only when some
            event is non-monotonic (see ``EventSpec.monotonic``).  ``solve_ivp``
            compares event signs at accepted step ends only, so a signal that
            crosses a threshold and comes back inside one large step would be
            missed; capping at the output step makes anything visible in the
            output visible to the detector.  Pass the simulation ``dt``.
        fallback_integrator: ``(f, grid, y0, scheme) -> y_history`` used to
            finish the run after the chattering guard trips.  A chattering
            system is one an adaptive solver cannot integrate at all -- its
            step collapses toward machine precision and the run never ends --
            so the fallback must be a fixed-step scheme, which terminates by
            construction (and matches what the interpreted path would produce).
            Omit it and the remainder is attempted with events simply switched
            off.
        solve_ivp: Injectable for tests; defaults to scipy's.

    Returns:
        EventSolveResult
    """
    if solve_ivp is None:
        from scipy.integrate import solve_ivp as solve_ivp

    t0 = float(t_span[0])
    tf = float(t_span[1])
    span = abs(tf - t0) or 1.0
    min_separation = MIN_SEPARATION_FRACTION * span
    nudge = RESTART_NUDGE_FRACTION * span

    t_eval = np.asarray(t_eval, dtype=float)
    n_points = int(t_eval.size)
    y0 = np.asarray(y0, dtype=float)

    result = EventSolveResult()
    result.t = t_eval
    result.y = np.zeros((y0.size, n_points))

    _reset_mode_states(mode_states, frozen=True)
    step_cap = _step_cap(max_step, specs)
    cache = _SignalCache(model_func.evaluate)
    _seed_modes(specs, cache, t0, y0)
    result.mode_history.append((t0, _snapshot_modes(mode_states)))
    event_callables = [_make_event_callable(spec, cache) for spec in specs]

    loop = _SegmentLoop(idx=0, t_start=t0, y_start=y0.copy())
    while True:
        loop.idx = _fill_gap_samples(result, t_eval, loop.idx, loop.t_start, loop.y_start)
        if loop.idx >= n_points:
            break

        sol = _run_segment(
            solve_ivp,
            model_func,
            tf,
            t_eval,
            loop,
            result,
            method,
            rtol,
            atol,
            step_cap,
            event_callables,
        )
        if not sol.success:
            result.success = False
            break
        if sol.status != 1:
            # Reached t_end with no further event: every remaining grid point
            # was produced by this segment.
            result.status = 0
            break

        t_event, y_event, fired = _resolve_event(sol, specs, min_separation, loop.y_start)
        if t_event is None:
            # Terminal status without a locatable root (defensive).
            result.status = 0
            break

        _apply_discrete_updates(fired, t_event, y_event, cache)
        _record_event(result, mode_states, t_event, fired)
        _update_chatter_guard(result, loop, t_event, fired, min_separation, max_events)
        loop.t_start, loop.y_start = _restart_point(model_func, t_event, y_event, nudge)
        cache.invalidate()

        if result.guard_tripped and loop.events_active:
            _disable_events(result, loop, mode_states)
            if fallback_integrator is not None:
                loop.idx = _finish_fixed_step(
                    result,
                    model_func,
                    t_eval,
                    loop.idx,
                    loop.t_start,
                    loop.y_start,
                    fallback_integrator,
                )
                break

    _truncate_to_produced(result, t_eval, loop.idx)
    return result


def _step_cap(max_step, specs: List[EventSpec]) -> float:
    """Solver step cap: ``max_step`` when some event is non-monotonic, else unbounded.

    A monotonic event (a Step edge, say) cannot hide a round trip through zero
    inside one step, so a diagram whose only discontinuities are scheduled in
    time keeps the solver's full stride.
    """
    step_cap = np.inf
    if max_step and np.isfinite(max_step) and max_step > 0:
        if any(not spec.monotonic for spec in specs):
            step_cap = float(max_step)
    return step_cap


def _seed_modes(specs: List[EventSpec], cache: _SignalCache, t0: float, y0: np.ndarray) -> None:
    """Run every ``on_start`` hook at ``(t0, y0)`` and drop the cached signals.

    Seeds discrete modes from the initial condition before the first segment,
    so a latch whose input already sits past a threshold starts in the right
    mode instead of switching spuriously on the way back.  A diagram with no
    ``on_start`` hooks costs nothing here.
    """
    if not any(spec.on_start is not None for spec in specs):
        return
    try:
        initial_signals = cache.signals(t0, y0)
        for spec in specs:
            if spec.on_start is not None:
                spec.on_start(t0, y0, initial_signals)
    except Exception:  # noqa: BLE001 - seeding is best-effort
        logger.debug("Event mode seeding failed at t0", exc_info=True)
    cache.invalidate()


def _fill_gap_samples(result, t_eval, idx: int, t_start: float, y_start) -> int:
    """Write ``y_start`` into every grid point before ``t_start``; return the new index.

    Grid points inside a restart nudge gap: the state is continuous there, so
    carry the event-time state across the (sub-picosecond) gap rather than
    dropping the sample and shortening the output.
    """
    n_points = int(t_eval.size)
    while idx < n_points and t_eval[idx] < t_start:
        result.y[:, idx] = y_start
        idx += 1
    return idx


def _run_segment(
    solve_ivp,
    model_func,
    tf: float,
    t_eval,
    loop: _SegmentLoop,
    result: EventSolveResult,
    method: str,
    rtol: float,
    atol: float,
    step_cap: float,
    event_callables,
):
    """One ``solve_ivp`` call from ``loop.t_start`` towards ``tf``.

    Writes whatever samples the segment produced into ``result.y`` (advancing
    ``loop.idx``), folds the solver counters and the last message/status into
    ``result``, and returns the raw scipy result so the caller can tell a
    finished run from a terminal event.  With events switched off (after a
    guard trip) the call is a plain, uncapped ``solve_ivp``.
    """
    sol = solve_ivp(
        model_func,
        (loop.t_start, tf),
        loop.y_start,
        t_eval=t_eval[loop.idx :],
        method=method,
        rtol=rtol,
        atol=atol,
        max_step=step_cap if loop.events_active else np.inf,
        events=event_callables if loop.events_active else None,
    )
    result.n_segments += 1
    result.nfev += int(getattr(sol, "nfev", 0) or 0)
    result.njev += int(getattr(sol, "njev", 0) or 0)
    result.nlu += int(getattr(sol, "nlu", 0) or 0)
    result.message = str(getattr(sol, "message", "") or "")
    result.status = int(getattr(sol, "status", 0) or 0)

    # A segment that ends before the next grid point produces no output
    # samples at all -- and scipy then leaves `t`/`y` as the empty *lists*
    # it accumulates into, so normalise before touching `.size`.
    seg_t = np.asarray(sol.t, dtype=float)
    produced = int(seg_t.size)
    if produced:
        result.y[:, loop.idx : loop.idx + produced] = np.asarray(sol.y, dtype=float)
        loop.idx += produced
    return sol


def _apply_discrete_updates(fired: List[EventSpec], t_event: float, y_event, cache) -> None:
    """Run the ``on_event`` hook of every event that fired at ``t_event``.

    All hooks see the same signals, evaluated once at the root; the cache is
    dropped afterwards because a hook may have flipped a latch the signals
    depended on.  A failing hook is logged and skipped -- it must not abort
    the run.
    """
    cache.invalidate()
    signals_at_event = cache.signals(t_event, y_event)
    for spec in fired:
        if spec.on_event is not None:
            try:
                spec.on_event(t_event, y_event, signals_at_event)
            except Exception:  # noqa: BLE001 - a bad update must not abort the run
                logger.warning("Discrete update for event %s failed at t=%.12g", spec.name, t_event)
    cache.invalidate()


def _record_event(result: EventSolveResult, mode_states, t_event: float, fired) -> None:
    """Append the located root to the event log and the post-event modes to the history."""
    result.event_log.append((float(t_event), [spec.name for spec in fired]))
    result.mode_history.append((float(t_event), _snapshot_modes(mode_states)))
    result.n_events += 1


def _update_chatter_guard(
    result: EventSolveResult,
    loop: _SegmentLoop,
    t_event: float,
    fired,
    min_separation: float,
    max_events: int,
) -> None:
    """Advance the chattering streak and trip the guard on a streak or the event cap.

    Sets ``result.guard_tripped`` / ``guard_reason``; the caller decides what
    to do about it (see ``_disable_events``).
    """
    if loop.last_event_t is not None and (t_event - loop.last_event_t) < min_separation:
        loop.chatter_streak += 1
    else:
        loop.chatter_streak = 0
    loop.last_event_t = t_event

    if loop.chatter_streak >= CHATTER_STREAK_LIMIT:
        result.guard_tripped = True
        result.guard_reason = (
            "chattering: {} consecutive events closer than {:.3g}s (around t={:.6g}s, {})".format(
                CHATTER_STREAK_LIMIT,
                min_separation,
                t_event,
                ", ".join(spec.name for spec in fired),
            )
        )
    elif result.n_events >= max_events:
        result.guard_tripped = True
        result.guard_reason = "event cap of {} reached at t={:.6g}s".format(max_events, t_event)


def _restart_point(model_func, t_event: float, y_event, nudge: float):
    """``(t_start, y_start)`` for the segment after a root: a nudge past it in both coordinates."""
    t_start = t_event + nudge
    if not (t_start > t_event):  # nudge lost to rounding at huge |t|: one ulp forward
        t_start = float(np.nextafter(t_event, np.inf))
    y_start = _state_at_restart(model_func, t_event, y_event, t_start - t_event)
    return t_start, y_start


def _disable_events(result: EventSolveResult, loop: _SegmentLoop, mode_states) -> None:
    """Turn event detection off for the rest of the run after a guard trip.

    Logs the reason once, records the instant, and lets latching blocks go
    back to updating their mode from the RHS: with events off nothing else
    would ever advance them.
    """
    logger.warning(
        "Zero-crossing detection disabled for the rest of this run -- %s. "
        "Switching instants after this point are located only to step "
        "accuracy. Add hysteresis to the switching element, or turn "
        "zero-crossing off in Simulation settings to silence this.",
        result.guard_reason,
    )
    loop.events_active = False
    result.events_off_at = float(loop.t_start)
    _reset_mode_states_frozen(mode_states, False)


def _truncate_to_produced(result: EventSolveResult, t_eval, idx: int) -> None:
    """Shorten ``result.t`` / ``result.y`` to the columns actually integrated.

    A failed or truncated solve leaves the tail unfilled; mirror scipy by
    reporting only what was integrated so callers see the short array.
    """
    if idx < int(t_eval.size):
        result.t = t_eval[:idx]
        result.y = result.y[:, :idx]


def _state_at_restart(model_func, t_event, y_event, gap: float) -> np.ndarray:
    """State to restart the next segment with, one explicit Euler step past the root.

    The restart has to move *both* coordinates of the event function.  Nudging
    only ``t`` and carrying ``y`` across unchanged is enough for a guard that is
    a function of time (a ``Step`` edge, ``t - delay``), but not for one written
    on the state: ``Saturation``'s ``u - max`` with ``u`` an integrator output is
    still exactly zero at ``(t_event + nudge, y_event)``, and scipy's
    ``find_active_events`` counts a zero at the start of a step as active and
    brentq returns that same instant as the root.  A single monotone crossing
    then re-fired ``CHATTER_STREAK_LIMIT`` times at ``nudge`` spacing and the run
    finished on the fixed-step fallback with every later switch located only to
    step accuracy.

    Taking the step makes ``g`` at the new segment start ``dg/dt * gap`` away
    from zero -- non-zero for any transversal crossing, and of the sign the
    trajectory is heading in, so the event is behind the segment rather than on
    its edge.  The state is advanced consistently with the time, which the plain
    carry-across was not, and the truncation error is one Euler step over
    ``1e-11`` of the span.

    Genuine chattering is untouched: a sliding relay's ``dy/dt`` at the switching
    surface is zero (or flips straight back), so the restart lands on the root
    again, the streak still builds and the guard still trips.

    Falls back to the unchanged state whenever the RHS cannot be evaluated or
    returns something unusable -- degrading to the old behaviour is always safer
    than aborting a solve.
    """
    y_event = np.asarray(y_event, dtype=float)
    if not gap > 0.0 or y_event.size == 0:
        return y_event
    try:
        dy, _signals = model_func.evaluate(t_event, y_event)
        dy = np.asarray(dy, dtype=float)
    except Exception:  # noqa: BLE001 - a restart must never break a solve
        logger.debug("RHS evaluation for the event restart failed at t=%s", t_event, exc_info=True)
        return y_event
    if dy.shape != y_event.shape or not np.all(np.isfinite(dy)):
        return y_event
    return y_event + gap * dy


def _finish_fixed_step(result, model_func, t_eval, idx, t_start, y_start, integrator) -> int:
    """Fill the rest of the output grid with a fixed-step integration.

    Reached only after the chattering guard gives up.  An adaptive solver
    cannot finish a chattering system -- with no event to stop at, its error
    control drives the step toward machine precision and the run effectively
    never terminates -- so the tail is integrated on the output grid itself,
    which costs exactly one pass and always ends.  Returns the new grid index.
    """
    n_points = int(t_eval.size)
    y_start = np.asarray(y_start, dtype=float)
    while idx < n_points and t_eval[idx] < t_start:
        result.y[:, idx] = y_start
        idx += 1
    if idx >= n_points:
        return idx
    grid = np.concatenate(([t_start], t_eval[idx:]))
    history = np.asarray(integrator(model_func, grid, y_start, "rk4"), dtype=float)
    result.y[:, idx:] = history[:, 1:]
    result.message = "Finished with a fixed step after the zero-crossing guard tripped"
    result.status = 0
    return n_points


def _snapshot_modes(mode_states) -> List[Any]:
    return [holder.get("mode") for holder in (mode_states or ())]


class ModeHistoryReplayer:
    """Drive latch holders through the modes the event solve recorded.

    ``solve_with_events`` freezes every claimed latch and flips it only at
    located roots, then leaves it at its end-of-run mode.  The post-solve
    replay re-executes kernels on the output grid, and a relay cannot re-derive
    its mode from grid samples: in a thermostat loop the error touches the
    thresholds only *at* the located instants, between samples, and turns back
    at once, so a Scope on the relay used to record one flat line while the
    temperature it drove limit-cycled.  Instead the replay calls :meth:`at`
    before each sample and the holders take the mode that held at that time
    (post-event at an event's own instant), frozen so the kernel just reads
    them.  After a chattering-guard trip the latches were free-running in the
    solve, so from ``events_off_at`` on they are unfrozen here too and the
    kernel advances them from the reconstructed signals, as it did then.
    Holders whose block opted out of zero-crossing were never frozen; they are
    reset to their initial mode and left free-running.
    """

    def __init__(self, mode_states, result) -> None:
        self._holders = list(mode_states or ())
        self._history = list(getattr(result, "mode_history", None) or [])
        self._off_at = getattr(result, "events_off_at", None)
        self._idx = -1
        for holder in self._holders:
            if not holder.get("event_driven", False):
                holder["mode"] = holder.get("init")
                holder["frozen"] = False
        self.active = bool(self._holders) and bool(self._history)

    def at(self, t: float) -> None:
        if not self.active:
            return
        history = self._history
        while self._idx + 1 < len(history) and history[self._idx + 1][0] <= t:
            self._idx += 1
        if self._idx < 0:
            return
        free = self._off_at is not None and t >= self._off_at
        for holder, mode in zip(self._holders, history[self._idx][1]):
            if not holder.get("event_driven", False):
                continue
            if free:
                holder["frozen"] = False
            else:
                holder["mode"] = mode
                holder["frozen"] = True


def _reset_mode_states_frozen(mode_states, frozen: bool) -> None:
    """Flip the freeze flag on latch holders without disturbing their values."""
    for holder in mode_states or ():
        holder["frozen"] = bool(frozen)


def _resolve_event(sol, specs, min_separation, y_fallback):
    """Earliest root in a terminated segment, plus every event at that instant.

    Simultaneous events (two limits reached together, a switch and a saturation
    on the same signal) all get their discrete update applied, not just the one
    scipy happened to report last.
    """
    t_event = None
    y_event = None
    t_events = getattr(sol, "t_events", None) or []
    y_events = getattr(sol, "y_events", None) or []

    for i in range(len(specs)):
        roots = t_events[i] if i < len(t_events) else None
        if roots is None or len(roots) == 0:
            continue
        te = float(roots[-1])
        if t_event is None or te < t_event:
            t_event = te
            states = y_events[i] if i < len(y_events) else None
            if states is not None and len(states):
                y_event = np.asarray(states[-1], dtype=float)

    if t_event is None:
        return None, None, []

    if y_event is None:
        seg_y = np.asarray(sol.y, dtype=float)
        y_event = np.asarray(seg_y[:, -1] if seg_y.size else y_fallback, dtype=float)

    tol = max(min_separation, 1e-12)
    fired = []
    for i, spec in enumerate(specs):
        roots = t_events[i] if i < len(t_events) else None
        if roots is None or len(roots) == 0:
            continue
        if abs(float(roots[-1]) - t_event) <= tol:
            fired.append(spec)
    return t_event, y_event, fired
