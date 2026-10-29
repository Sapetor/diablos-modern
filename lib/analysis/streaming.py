"""Incremental (streaming) reduction of re-simulation runs.

The Monte-Carlo and parameter-sweep runners used to collect every run's Scope
traces and only reduce them once the whole experiment had finished, so memory
grew as ``n_runs * L`` and the 2-D sweep built a full ``(n_points, L)`` matrix
just to compute per-point scalars. This module reduces each run *as it
finishes*:

  * :func:`reduce_run` turns one harvested run into a compact record -- the
    per-run :data:`OUTCOME_METRICS` scalars, the per-signal lengths, and (only
    when the caller still needs them) the traces. It is used identically by the
    in-process loop and by pool worker processes, which is what makes
    ``workers=1`` and ``workers=N`` bit-identical.
  * :class:`EnsembleAccumulator` folds records (in run-index order) into the
    Monte-Carlo result: exact streaming mean/std (Welford) and min/max
    envelopes, per-run metric vectors, and the traces only if asked to keep them.
  * :class:`SweepAccumulator` does the same for 1-D / 2-D sweeps; a 2-D sweep
    never holds a trace beyond the run that produced it.

Percentiles are the one ensemble statistic with no exact fixed-memory streaming
form. They are computed exactly from the full trace matrix when every trace is
kept (``keep_traces=True``, the GUI default) and are reported as ``None``
otherwise -- no silent approximation. Viewers fall back to a mean +/- std band.

Common-length caveat: like the old batch code, results are truncated to the
shortest run (and the signals present in every successful run). Per-run
metrics are computed at each run's own length; if a later run turns out to be
shorter, metrics of earlier runs are recomputed from the kept traces when
available, otherwise they stay at the longer length and a warning is logged.
With fixed ``sim_time``/``sim_dt`` every run has the same length, so this does
not arise in practice.
"""

import logging
import math

import numpy as np

from lib.analysis.resim import OUTCOME_METRICS

logger = logging.getLogger(__name__)

__all__ = [
    "reduce_run",
    "run_metrics",
    "EnsembleAccumulator",
    "SweepAccumulator",
    "trace_sample_indices",
]


def run_metrics(trace):
    """``{metric: float}`` of one 1-D trace, via the shared OUTCOME_METRICS."""
    row = np.asarray(trace, dtype=float)[None, :]
    with np.errstate(all="ignore"):
        return {m: float(fn(row)[0]) for m, fn in OUTCOME_METRICS.items()}


def reduce_run(harvest, keep_signals):
    """Compact record of one run's harvest (``None`` if it produced no signals).

    Returns ``{"timeline", "lengths", "length", "metrics", "signals"}``:
    ``length`` is this run's own common length (timeline and every channel),
    ``metrics`` is ``{signal: {metric: float}}`` computed on the first
    ``length`` samples, and ``signals`` holds the traces only when
    ``keep_signals`` (else ``None``).
    """
    if not harvest or not harvest.get("signals"):
        return None
    timeline = np.asarray(harvest["timeline"], dtype=float)
    sigs = harvest["signals"]
    lengths = {nm: len(arr) for nm, arr in sigs.items()}
    length = min([len(timeline)] + list(lengths.values()))
    metrics = {nm: run_metrics(arr[:length]) for nm, arr in sigs.items()} if length else {}
    return {
        "timeline": timeline,
        "lengths": lengths,
        "length": length,
        "metrics": metrics,
        "signals": dict(sigs) if keep_signals else None,
    }


def trace_sample_indices(n_runs, keep_traces):
    """Which run indices keep their trace.

    ``keep_traces``: ``True`` -> all runs (returns ``None`` = "all"); ``False``
    / ``0`` -> none (empty set); an int ``K`` -> an evenly spaced sample of at
    most ``K`` run indices (the same selection rule the ensemble window uses
    for its faint member traces).
    """
    if keep_traces is True:
        return None
    k = int(keep_traces or 0)
    if k <= 0 or n_runs <= 0:
        return frozenset()
    if k >= n_runs:
        return None
    step = max(1, int(math.ceil(n_runs / k)))
    return frozenset(range(0, n_runs, step))


class _SignalStats:
    """Per-signal streaming state: shifted Welford mean/M2, min/max, metrics.

    The running moments are kept for ``x - shift`` (``shift`` = the first run's
    trace), which keeps the variance accurate when the signal sits on a large
    offset relative to its spread (catastrophic cancellation otherwise).
    """

    __slots__ = ("shift", "mean", "m2", "lo", "hi", "metrics", "traces")

    def __init__(self):
        self.shift = None
        self.mean = None
        self.m2 = None
        self.lo = None
        self.hi = None
        self.metrics = []  # one {metric: float} per ok run, in index order
        self.traces = []  # kept traces (list of 1-D arrays), in index order

    def truncate(self, L):
        if self.mean is not None and self.mean.size > L:
            self.shift = self.shift[:L].copy()
            self.mean = self.mean[:L].copy()
            self.m2 = self.m2[:L].copy()
            self.lo = self.lo[:L].copy()
            self.hi = self.hi[:L].copy()


class EnsembleAccumulator:
    """Fold per-run records into the Monte-Carlo ensemble result, in index order.

    Feed :meth:`add` with ``(run_index, record_or_None)`` in increasing run index
    (gaps allowed after a cancellation); call :meth:`result` at the end.
    """

    def __init__(self, n_runs, keep_traces=True):
        self.keep_traces = keep_traces
        self._keep_idx = trace_sample_indices(n_runs, keep_traces)
        self._all_traces = self._keep_idx is None
        self.n_runs = 0
        self.n_ok = 0
        self.names = None  # signals present in every ok run so far
        self.L = None  # common length so far
        self.timeline = None
        self.stats = {}
        self.kept_indices = []  # run indices whose trace is in stats[*].traces
        self._metric_lengths = []  # length each ok run's metrics were computed at

    def _keeps(self, index):
        return self._all_traces or index in self._keep_idx

    def add(self, index, rec):
        self.n_runs += 1
        if rec is None:
            return
        self.n_ok += 1
        sigs_present = set(rec["lengths"])
        if self.names is None:
            self.names = set(sigs_present)
            self.timeline = rec["timeline"]
            for nm in self.names:
                self.stats[nm] = _SignalStats()
        else:
            for nm in self.names - sigs_present:
                self.stats.pop(nm, None)
            self.names &= sigs_present
        if not self.names:
            return

        # Common length over the timeline and every *common* channel.
        run_L = min([len(rec["timeline"])] + [rec["lengths"][nm] for nm in self.names])
        if self.L is None or run_L < self.L:
            self.L = run_L
            for st in self.stats.values():
                st.truncate(run_L)
        L = self.L
        if L == 0:
            return  # nothing to aggregate (the result comes back empty)
        self._metric_lengths.append(rec["length"])

        if rec.get("signals") is None:
            raise ValueError("EnsembleAccumulator needs the traces of every run")
        keep = self._keeps(index)
        if keep:
            self.kept_indices.append(index)
        n = self.n_ok
        for nm in self.names:
            st = self.stats[nm]
            st.metrics.append(rec["metrics"][nm])
            x = np.asarray(rec["signals"][nm][:L], dtype=float)
            if st.mean is None:
                st.shift = x.copy()
                st.mean = np.zeros_like(x)
                st.m2 = np.zeros_like(x)
                st.lo = x.copy()
                st.hi = x.copy()
            else:
                # Welford update on the shifted data (numerically stable
                # running mean / variance).
                delta = (x - st.shift) - st.mean
                st.mean += delta / n
                st.m2 += delta * ((x - st.shift) - st.mean)
                np.minimum(st.lo, x, out=st.lo)
                np.maximum(st.hi, x, out=st.hi)
            if keep:
                st.traces.append(rec["signals"][nm])

    def result(self):
        """The ensemble-result dict (see ``MonteCarloRunner._aggregate``)."""
        traces_mode = "all" if self._all_traces else ("sample" if self._keep_idx else "none")
        out = {
            "n_runs": self.n_runs,
            "n_ok": self.n_ok,
            "timeline": None,
            "signals": {},
            "traces": traces_mode,
        }
        if not self.n_ok or not self.names or not self.L:
            return out
        L = self.L
        out["timeline"] = np.asarray(self.timeline[:L], dtype=float)
        stale = any(m != L for m in self._metric_lengths)
        for nm in sorted(self.names):
            st = self.stats[nm]
            # Ok runs that predate the signal set narrowing still count: every
            # ok run had this signal (names is an intersection), so n_ok rows.
            n = self.n_ok
            M = np.vstack([t[:L] for t in st.traces]) if st.traces else None
            if stale and M is not None and self._all_traces:
                metrics = {k: fn(M) for k, fn in OUTCOME_METRICS.items()}
            else:
                if stale:
                    logger.warning(
                        "Ensemble runs have different lengths; per-run metrics of %r "
                        "were computed before truncation to the common length",
                        nm,
                    )
                metrics = {
                    k: np.array([m[k] for m in st.metrics], dtype=float) for k in OUTCOME_METRICS
                }
            sig = {
                "runs": M,
                "mean": st.shift[:L] + st.mean[:L],
                "std": np.sqrt(st.m2[:L] / n),
                "p5": None,
                "p50": None,
                "p95": None,
                "min": st.lo[:L].copy(),
                "max": st.hi[:L].copy(),
                "metrics": metrics,
            }
            if self._all_traces and M is not None:
                sig["p5"] = np.percentile(M, 5, axis=0)
                sig["p50"] = np.percentile(M, 50, axis=0)
                sig["p95"] = np.percentile(M, 95, axis=0)
            out["signals"][nm] = sig
        return out


class SweepAccumulator:
    """Fold per-point records into the sweep result's ``timeline``/``signals``.

    1-D sweeps keep each point's trace (the response family) when
    ``keep_traces``; 2-D sweeps never keep traces -- only one scalar per metric
    per grid point, so memory is ``O(n_points)`` rather than ``O(n_points * L)``.
    """

    def __init__(self, mode, shape, keep_traces=True):
        self.mode = mode
        self.shape = tuple(shape)
        self.n_points = int(np.prod(self.shape))
        self.keep_traces = bool(keep_traces) and mode == "1d"
        self.n_ok = 0
        self.names = None
        self.L = None
        self.timeline = None
        self.metrics = {}  # signal -> {metric: ndarray(n_points) NaN-filled}
        self.traces = {}  # signal -> {point_index: trace}
        self._metric_lengths = []

    def add(self, k, rec):
        if rec is None:
            return
        self.n_ok += 1
        present = set(rec["lengths"])
        if self.names is None:
            self.names = set(present)
            self.timeline = rec["timeline"]
            for nm in self.names:
                self.metrics[nm] = {m: np.full(self.n_points, np.nan) for m in OUTCOME_METRICS}
                self.traces[nm] = {}
        else:
            for nm in self.names - present:
                self.metrics.pop(nm, None)
                self.traces.pop(nm, None)
            self.names &= present
        if not self.names:
            return
        run_L = min([len(rec["timeline"])] + [rec["lengths"][nm] for nm in self.names])
        self.L = run_L if self.L is None else min(self.L, run_L)
        self._metric_lengths.append(rec["length"])
        for nm in self.names:
            for m, v in rec["metrics"][nm].items():
                self.metrics[nm][m][k] = v
            if self.keep_traces and rec.get("signals") is not None:
                self.traces[nm][k] = rec["signals"][nm]

    def fill(self, result):
        """Write ``timeline`` and ``signals`` into the sweep ``result`` dict."""
        if not self.n_ok or not self.names or not self.L:
            return result
        L = self.L
        result["timeline"] = np.asarray(self.timeline[:L], dtype=float)
        stale = any(m != L for m in self._metric_lengths)
        for nm in sorted(self.names):
            metrics = self.metrics[nm]
            entry = {}
            if self.keep_traces:
                M = np.full((self.n_points, L), np.nan)
                for k, tr_ in self.traces[nm].items():
                    M[k] = tr_[:L]
                entry["traces"] = M
                if stale:
                    with np.errstate(all="ignore"):
                        metrics = {m: fn(M) for m, fn in OUTCOME_METRICS.items()}
            elif stale:
                logger.warning(
                    "Sweep runs have different lengths; metrics of %r were computed "
                    "before truncation to the common length",
                    nm,
                )
            if self.mode == "2d":
                entry["metrics"] = {m: v.reshape(self.shape) for m, v in metrics.items()}
            else:
                entry["metrics"] = metrics
            result["signals"][nm] = entry
        return result
