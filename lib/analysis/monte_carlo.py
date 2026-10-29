"""Monte-Carlo / ensemble runner for stochastic block diagrams.

Re-runs a diagram N times with per-run seeds (reproducible from a single master
seed) and optional per-run parameter samples, harvests Scope traces, and
aggregates ensemble statistics (mean / std / percentile bands / min-max).

Built for network-effect studies: drop a PacketLoss block, a
VariableTransportDelay driven by a random source, and/or Noise into a diagram,
then run an ensemble and read the mean response with an uncertainty envelope.

The per-run seed injection is the key piece: each stochastic block (any block
exposing a ``seed`` param) gets a fresh sub-seed derived from
(master_seed, run_index, block_name), so every run differs yet the whole
experiment is bit-reproducible from one master seed. Original block params are
restored afterwards so the runner never mutates the user's diagram.
"""

import logging
import zlib

import numpy as np

# Re-exported from the shared re-sim module so existing
# ``from lib.analysis.monte_carlo import OUTCOME_METRICS`` imports keep working.
from lib.analysis.resim import (
    OUTCOME_METRIC_LABELS,
    OUTCOME_METRICS,
    harvest_scope_signals,
    iter_blocks_qualified,
    root_blocks_of,
)
from lib.analysis.parallel import run_indexed
from lib.analysis.streaming import EnsembleAccumulator, reduce_run

logger = logging.getLogger(__name__)

__all__ = ["MonteCarloRunner", "derive_seed", "OUTCOME_METRICS", "OUTCOME_METRIC_LABELS"]


def derive_seed(master_seed, run_index, tag):
    """Deterministic, non-zero 32-bit sub-seed from (master_seed, run_index, tag).

    Reproducible from ``master_seed``; distinct per ``run_index`` and ``tag``.
    Avoids 0, which stochastic blocks treat as "entropy / non-reproducible".
    """
    ss = np.random.SeedSequence(
        [
            int(master_seed) & 0x7FFFFFFF,
            int(run_index) & 0x7FFFFFFF,
            # crc32, not hash(): str hashes are salted per process
            # (PYTHONHASHSEED), which would break cross-process reproducibility.
            zlib.crc32(str(tag).encode("utf-8")) & 0x7FFFFFFF,
        ]
    )
    s = int(ss.generate_state(1, dtype=np.uint32)[0])
    return s if s != 0 else 1


class MonteCarloRunner:
    """Run N seeded simulations of ``dsim`` and aggregate Scope-trace statistics."""

    def __init__(self, dsim):
        self.dsim = dsim

    def run(
        self,
        n_runs,
        master_seed=12345,
        sim_time=None,
        sim_dt=None,
        samplers=None,
        progress_cb=None,
        cancel_cb=None,
        keep_traces=True,
        workers=1,
    ):
        """Run ``n_runs`` simulations and aggregate ensemble statistics.

        Each run is reduced as soon as it finishes (see
        :mod:`lib.analysis.streaming`): mean/std (Welford) and min/max envelopes
        are accumulated incrementally and the per-run outcome metrics are kept
        as scalars, so memory no longer has to grow with ``n_runs * L``.

        Args:
            n_runs: number of Monte-Carlo runs.
            master_seed: experiment seed; the whole ensemble is reproducible from it.
            sim_time, sim_dt: overrides (default: the diagram's current values).
            samplers: optional {(block_name, param_name): (low, high) | callable(rng)->value}
                for per-run parameter Monte-Carlo (uniform draw when a (low, high) tuple).
                Values are drawn in this process, so callables need not be picklable.
            progress_cb: optional callable(done, total), called as runs complete.
            cancel_cb: optional callable() -> bool, polled before each run *and*
                periodically inside each run (it is forwarded to
                ``run_tuning_simulation``); when it returns True the loop stops
                early and the partial ensemble gathered so far is aggregated and
                returned (the diagram is still restored).
            keep_traces: ``True`` (default) keeps every run's trace -- ``runs``
                is the full ``(n_ok, L)`` matrix and the p5/p50/p95 bands are
                exact. ``False`` keeps none (``runs`` and the percentile bands
                are ``None``); an int ``K`` keeps an evenly spaced sample of at
                most ``K`` runs in ``runs`` (percentiles still ``None``: they
                have no exact streaming form). mean/std/min/max and ``metrics``
                are exact in every mode.
            workers: ``1`` (default) runs in this process on ``self.dsim``;
                ``> 1`` runs on a spawn-based process pool (see
                :mod:`lib.analysis.parallel`). Results are bit-identical for the
                same master seed whatever the worker count.

        Returns:
            An ensemble-result dict (see :meth:`_aggregate`).
        """
        dsim = self.dsim
        sim_time = float(sim_time if sim_time is not None else getattr(dsim, "sim_time", 1.0))
        sim_dt = float(sim_dt if sim_dt is not None else getattr(dsim, "sim_dt", 0.01))
        samplers = samplers or {}
        n_runs = int(n_runs)

        # Every block, including those nested in Subsystems, keyed by its
        # qualified (flattened) name.
        blocks_by_name = dict(iter_blocks_qualified(root_blocks_of(dsim)))
        seed_blocks = [
            qn
            for qn, b in blocks_by_name.items()
            if isinstance(getattr(b, "params", None), dict) and "seed" in b.params
        ]

        def assignments_for(i):
            """Per-run seeds + parameter samples, reproducible from master_seed."""
            out = {(qn, "seed"): derive_seed(master_seed, i, qn) for qn in seed_blocks}
            if samplers:
                prng = np.random.default_rng(derive_seed(master_seed, i, "__params__"))
                for (bn, pn), spec in samplers.items():
                    if bn not in blocks_by_name:
                        continue
                    out[(bn, pn)] = (
                        spec(prng) if callable(spec) else float(prng.uniform(spec[0], spec[1]))
                    )
            return out

        # Snapshot originals so we restore the diagram exactly (never mutate it).
        original = {}
        for qn in seed_blocks:
            original[(qn, "seed")] = blocks_by_name[qn].params.get("seed")
        for bn, pn in samplers:
            b = blocks_by_name.get(bn)
            if b is not None:
                original[(bn, pn)] = b.params.get(pn)

        acc = EnsembleAccumulator(n_runs, keep_traces=keep_traces)
        try:
            run_indexed(
                dsim,
                n_runs,
                assignments_for,
                acc.add,
                sim_time,
                sim_dt,
                keep_signals=True,  # per-timestep statistics need every trace once
                workers=workers,
                progress_cb=progress_cb,
                cancel_cb=cancel_cb,
                label="Monte-Carlo run",
            )
        finally:
            for (bn, pn), val in original.items():
                b = blocks_by_name.get(bn)
                if b is None:
                    continue
                if val is None:
                    b.params.pop(pn, None)
                else:
                    b.params[pn] = val
                # Keep exec_params in sync with params (restore the original value
                # rather than dropping the key) so the diagram is left untouched.
                if getattr(b, "exec_params", None) is not None:
                    if val is None:
                        b.exec_params.pop(pn, None)
                    else:
                        b.exec_params[pn] = val

        return acc.result()

    def _harvest(self):
        """Read each Scope's trace(s) into {signal_name: 1-D ndarray} + timeline."""
        return harvest_scope_signals(self.dsim)

    @staticmethod
    def _aggregate(runs, keep_traces=True):
        """Aggregate per-run harvests into ensemble statistics.

        Returns {n_runs, n_ok, timeline, traces, signals: {name: {runs, mean,
        std, p5, p50, p95, min, max, metrics}}}. The per-timestep series
        (mean/std/percentiles/min/max) are length-L arrays over the timeline;
        ``metrics`` is {metric_name: length-n_ok vector} of per-run scalar
        outcomes (see :data:`OUTCOME_METRICS`). Signals are restricted to those
        present in every successful run and truncated to a common length so all
        series align. ``runs`` / ``p5`` / ``p50`` / ``p95`` are ``None`` unless
        every trace is kept (``traces`` is ``"all"``, ``"sample"`` or
        ``"none"``; see ``run(keep_traces=...)``).

        ``run`` streams records into an :class:`EnsembleAccumulator` directly;
        this batch entry point folds a list of harvests the same way.
        """
        acc = EnsembleAccumulator(len(runs), keep_traces=keep_traces)
        for i, h in enumerate(runs):
            acc.add(i, reduce_run(h, keep_signals=True))
        return acc.result()
