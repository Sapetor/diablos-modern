"""Transport delays on the compiled path: the method of steps.

``TransportDelay`` makes the diagram a delay differential equation: the RHS at
``t`` reads the delay's input at ``t - tau``. ``solve_ivp`` integrates it
correctly only if that read is a fixed function of ``t``, i.e. if it comes
from a part of the trajectory that is already integrated. So every ``k * tau``
is a breakpoint of ``zero_crossing.solve_with_events`` (merged with the
sample instants): no segment is longer than the shortest delay, and inside a
segment ``t - tau`` always falls into a finished segment. The breakpoints also
put each discontinuity that a delay carries forward from ``t0`` on a segment
boundary, so the solver never steps across one.

:class:`DelayHistory` holds the delayed inputs, one shared object per compile:

* ``finish(sol)`` runs after each segment, before the updates at its end. It
  evaluates the diagram on nodes of the solver's dense output (accepted steps
  only, never trial stages) at most ``dt / 3`` apart and stores a cubic spline
  per segment. One evaluation serves every delay in the diagram. The first
  node sees the updates at the segment's start and the last sits just inside
  its end, so the spline never straddles a jump.
* ``value(i, q)`` is the kernel's read: ``initial_value`` up to ``t0`` (as
  the interpreted block, which also answers ``initial_value`` exactly at its
  first recorded instant), else the spline of the segment holding ``q``. At a
  boundary that is the later segment (right-continuous, as the replay shows
  it), except for the segment being integrated: a segment exactly ``tau``
  long reads its own start at its last stage and gets the left value there,
  so its RHS stays smooth up to its end, as a sampled holder's does. A read
  past the recorded history raises :class:`CompiledDelayError` rather than
  extrapolating.

The history resets at the start of every run (``reset``), which matters on a
compile-cache hit, and stays filled afterwards so the post-solve replay reads
the same trajectory the solve did.
"""

import bisect

import numpy as np
from scipy.interpolate import CubicSpline

from lib.engine.block_names import canonical_fn
from lib.engine.block_params import runtime_params

#: Canonical fn-names of the delay blocks the compiler handles.
DELAY_FNS = frozenset({"Transportdelay"})

#: Spline nodes per output step.
NODES_PER_DT = 3


class CompiledDelayError(RuntimeError):
    """A compiled delay read history that is not integrated yet."""


def is_delay_block(block) -> bool:
    return canonical_fn(getattr(block, "block_fn", "")) in DELAY_FNS


def delay_time(block) -> float:
    """The block's resolved delay in seconds; ``nan`` when unreadable."""
    try:
        return float(runtime_params(block).get("delay_time", 0.1))
    except (TypeError, ValueError):
        return float("nan")


class DelayHistory:
    """Recorded inputs of every delay block in one compiled diagram."""

    def __init__(self):
        self.delays = []  # (name, tau, initial output, input signal key)
        self.model_func = None
        self.reset()

    def add(self, name, tau, initial, source_key) -> int:
        self.delays.append((name, float(tau), np.atleast_1d(float(initial)), source_key))
        return len(self.delays) - 1

    @property
    def taus(self):
        return [tau for _name, tau, _init, _src in self.delays]

    def reset(self, t0=None, dt=None, span=1.0):
        """Forget the recorded history (start of a run)."""
        self._t0 = t0
        self._dt = dt
        self._tol = 1e-9 * (abs(span) or 1.0)
        self._starts, self._ends, self._segments = [], [], []
        self._slices = None
        self._memo = (None, None)

    # -- recording ---------------------------------------------------------

    def _row(self, signals):
        parts = []
        for _name, _tau, _init, key in self.delays:
            value = 0.0 if key is None else signals.get(key, 0.0)
            parts.append(np.ravel(np.asarray(value, dtype=float)))
        if self._slices is None:
            self._slices, i = [], 0
            for part in parts:
                self._slices.append(slice(i, i + part.size))
                i += part.size
        return np.concatenate(parts)

    def _rows_at(self, times, states):
        rows = []
        for k, t in enumerate(times):
            _dy, signals = self.model_func.evaluate(float(t), states[:, k])
            rows.append(self._row(signals))
        return np.asarray(rows)

    def _store(self, start, end, segment):
        self._starts.append(start)
        self._ends.append(end)
        self._segments.append(segment)
        self._memo = (None, None)

    def finish(self, sol):
        """Record the segment ``sol`` just integrated (before its end updates)."""
        dense = sol.sol
        ts = np.asarray(dense.ts, dtype=float)
        if ts.size < 2 or ts[-1] <= ts[0]:
            return
        nodes = []
        for a, b in zip(ts[:-1], ts[1:]):
            n = NODES_PER_DT
            if self._dt:
                n = max(n, int(np.ceil(NODES_PER_DT * (b - a) / self._dt)))
            nodes.append(a + (b - a) * np.arange(n) / n)
        # The end node sits just inside the segment: a time source that jumps
        # at a located event (a Step edge) already reads its new value at the
        # root, and one post-jump node would make the spline ring over the
        # whole segment. The next segment's first node is the right-hand value.
        h_last = ts[-1] - ts[-2]
        nodes.append([ts[-1] - max(1e-9 * h_last, 4 * np.spacing(abs(ts[-1])))])
        nodes = np.unique(np.concatenate(nodes))
        rows = self._rows_at(nodes, np.atleast_2d(dense(nodes)))
        segment = rows[0] if nodes.size < 2 else CubicSpline(nodes, rows, axis=0)
        self._store(float(ts[0]), float(ts[-1]), segment)

    # -- reading -----------------------------------------------------------

    def _lookup(self, q):
        if self._memo[0] == q:
            return self._memo[1]
        j = bisect.bisect_right(self._starts, q) - 1
        if j < 0 or q > self._ends[j] + self._tol:
            raise CompiledDelayError(
                f"delay history read at t={q:.6g} past the integrated "
                f"{self._ends[-1] if self._ends else self._t0!r}"
            )
        segment = self._segments[j]
        if isinstance(segment, np.ndarray):
            row = segment
        else:
            row = segment(min(max(q, self._starts[j]), self._ends[j]))
        self._memo = (q, row)
        return row

    def value(self, i, q):
        """Delay ``i``'s output: its input at ``q = t - tau``."""
        _name, _tau, initial, _key = self.delays[i]
        if self._t0 is None or q <= self._t0 + self._tol:
            return initial.copy()
        row = self._lookup(float(q))
        return np.array(row[self._slices[i]], dtype=float)


def build_delay(ctx, history):
    """Kernel for one TransportDelay: a pure read of ``history``."""
    b_name = ctx.b_name
    tau = float(ctx.params.get("delay_time", 0.1))
    initial = float(ctx.params.get("initial_value", 0.0))
    source = ctx.input_sources[0] if ctx.input_sources else None
    index = history.add(b_name, tau, initial, source)

    def exec_delay(t, y, dy_vec, signals):
        signals[b_name] = history.value(index, t - tau)

    return exec_delay
