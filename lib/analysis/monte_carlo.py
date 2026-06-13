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
import numpy as np

logger = logging.getLogger(__name__)


def derive_seed(master_seed, run_index, tag):
    """Deterministic, non-zero 32-bit sub-seed from (master_seed, run_index, tag).

    Reproducible from ``master_seed``; distinct per ``run_index`` and ``tag``.
    Avoids 0, which stochastic blocks treat as "entropy / non-reproducible".
    """
    ss = np.random.SeedSequence([
        int(master_seed) & 0x7FFFFFFF,
        int(run_index) & 0x7FFFFFFF,
        hash(str(tag)) & 0x7FFFFFFF,
    ])
    s = int(ss.generate_state(1, dtype=np.uint32)[0])
    return s if s != 0 else 1


# Per-run outcome metrics. Each maps an ensemble matrix ``M`` of shape
# (n_ok, L) -- one row per successful run -- to a length-``n_ok`` vector holding
# one scalar summary per run. These power the results window's outcome-metric
# histograms (the distribution of, e.g., final value or peak across the
# ensemble). Insertion order is the order they appear in the metric picker.
OUTCOME_METRICS = {
    "final": lambda M: np.asarray(M)[:, -1],
    "mean": lambda M: np.asarray(M).mean(axis=1),
    "max": lambda M: np.asarray(M).max(axis=1),
    "min": lambda M: np.asarray(M).min(axis=1),
    "peak-to-peak": lambda M: np.ptp(np.asarray(M), axis=1),
    "rms": lambda M: np.sqrt((np.asarray(M) ** 2).mean(axis=1)),
}


class MonteCarloRunner:
    """Run N seeded simulations of ``dsim`` and aggregate Scope-trace statistics."""

    def __init__(self, dsim):
        self.dsim = dsim

    def run(self, n_runs, master_seed=12345, sim_time=None, sim_dt=None,
            samplers=None, progress_cb=None, cancel_cb=None):
        """Run ``n_runs`` simulations and aggregate ensemble statistics.

        Args:
            n_runs: number of Monte-Carlo runs.
            master_seed: experiment seed; the whole ensemble is reproducible from it.
            sim_time, sim_dt: overrides (default: the diagram's current values).
            samplers: optional {(block_name, param_name): (low, high) | callable(rng)->value}
                for per-run parameter Monte-Carlo (uniform draw when a (low, high) tuple).
            progress_cb: optional callable(done, total).
            cancel_cb: optional callable() -> bool, polled before each run; when it
                returns True the loop stops early and the partial ensemble gathered
                so far is aggregated and returned (the diagram is still restored).

        Returns:
            An ensemble-result dict (see :meth:`_aggregate`).
        """
        dsim = self.dsim
        sim_time = float(sim_time if sim_time is not None else getattr(dsim, 'sim_time', 1.0))
        sim_dt = float(sim_dt if sim_dt is not None else getattr(dsim, 'sim_dt', 0.01))
        samplers = samplers or {}

        blocks_by_name = {b.name: b for b in dsim.blocks_list}
        seed_blocks = [b for b in dsim.blocks_list
                       if isinstance(getattr(b, 'params', None), dict) and 'seed' in b.params]

        # Snapshot originals so we restore the diagram exactly (never mutate it).
        original = {}
        for b in seed_blocks:
            original[(b.name, 'seed')] = b.params.get('seed')
        for (bn, pn) in samplers:
            b = blocks_by_name.get(bn)
            if b is not None:
                original[(bn, pn)] = b.params.get(pn)

        def _set(block, name, value):
            block.params[name] = value
            if getattr(block, 'exec_params', None):
                block.exec_params[name] = value

        runs = []
        try:
            for i in range(n_runs):
                if cancel_cb is not None and cancel_cb():
                    logger.info("Monte-Carlo cancelled after %d/%d runs", i, n_runs)
                    break
                # Per-run seed injection for every stochastic block.
                for b in seed_blocks:
                    _set(b, 'seed', derive_seed(master_seed, i, b.name))
                # Per-run parameter samples (reproducible from the master seed).
                if samplers:
                    prng = np.random.default_rng(derive_seed(master_seed, i, '__params__'))
                    for (bn, pn), spec in samplers.items():
                        b = blocks_by_name.get(bn)
                        if b is None:
                            continue
                        val = spec(prng) if callable(spec) else float(prng.uniform(spec[0], spec[1]))
                        _set(b, pn, val)

                ok, err = dsim.run_tuning_simulation(sim_time, sim_dt)
                if ok:
                    runs.append(self._harvest())
                else:
                    runs.append(None)
                    logger.warning("Monte-Carlo run %d failed: %s", i, err)
                if progress_cb:
                    progress_cb(i + 1, n_runs)
        finally:
            for (bn, pn), val in original.items():
                b = blocks_by_name.get(bn)
                if b is None:
                    continue
                if val is None:
                    b.params.pop(pn, None)
                else:
                    b.params[pn] = val
                if getattr(b, 'exec_params', None) is not None:
                    b.exec_params.pop(pn, None)

        return self._aggregate(runs)

    def _harvest(self):
        """Read each Scope's trace(s) into {signal_name: 1-D ndarray} + timeline."""
        dsim = self.dsim
        timeline = getattr(dsim, 'timeline', None)
        if timeline is None or len(np.atleast_1d(timeline)) == 0:
            return None
        blocks = getattr(dsim.engine, 'active_blocks_list', None) or dsim.blocks_list
        signals = {}
        seen = {}

        def put(name, arr):
            if name in seen:
                seen[name] += 1
                name = f"{name}#{seen[name]}"
            else:
                seen[name] = 0
            signals[name] = np.asarray(arr, dtype=float).ravel()

        for b in blocks:
            if b.block_fn != 'Scope':
                continue
            params = getattr(b, 'exec_params', None) or b.params
            vec = params.get('vector')
            if vec is None:
                continue
            arr = np.asarray(vec, dtype=float)
            vec_dim = int(params.get('vec_dim', 1) or 1)
            labels = params.get('vec_labels')
            # Scope stores a flat concatenated buffer; reshape multi-channel data.
            if arr.ndim == 1 and vec_dim > 1 and arr.size % vec_dim == 0:
                arr = arr.reshape(-1, vec_dim)
            if arr.ndim == 1:
                put(labels if isinstance(labels, str) else b.name, arr)
            else:
                for j in range(arr.shape[1]):
                    nm = (labels[j] if isinstance(labels, (list, tuple)) and j < len(labels)
                          else f"{b.name}[{j}]")
                    put(nm, arr[:, j])
        return {'timeline': np.asarray(timeline, dtype=float), 'signals': signals}

    @staticmethod
    def _aggregate(runs):
        """Aggregate per-run harvests into ensemble statistics.

        Returns {n_runs, n_ok, timeline, signals: {name: {runs, mean, std, p5,
        p50, p95, min, max, metrics}}}. The per-timestep series (mean/std/
        percentiles/min/max) are length-L arrays over the timeline; ``metrics``
        is {metric_name: length-n_ok vector} of per-run scalar outcomes (see
        :data:`OUTCOME_METRICS`). Signals are restricted to those present in
        every successful run and truncated to a common length so all series align.
        """
        ok = [r for r in runs if r is not None and r['signals']]
        result = {'n_runs': len(runs), 'n_ok': len(ok), 'timeline': None, 'signals': {}}
        if not ok:
            return result

        names = set(ok[0]['signals'])
        for r in ok[1:]:
            names &= set(r['signals'])
        if not names:
            return result

        lengths = [len(r['timeline']) for r in ok]
        for r in ok:
            for nm in names:
                lengths.append(len(r['signals'][nm]))
        L = min(lengths)
        if L == 0:
            return result

        result['timeline'] = np.asarray(ok[0]['timeline'][:L], dtype=float)
        for nm in sorted(names):
            M = np.vstack([r['signals'][nm][:L] for r in ok])
            result['signals'][nm] = {
                'runs': M,
                'mean': M.mean(axis=0),
                'std': M.std(axis=0),
                'p5': np.percentile(M, 5, axis=0),
                'p50': np.percentile(M, 50, axis=0),
                'p95': np.percentile(M, 95, axis=0),
                'min': M.min(axis=0),
                'max': M.max(axis=0),
                # Per-run scalar outcomes: {metric_name: length-n_ok vector}.
                'metrics': {k: fn(M) for k, fn in OUTCOME_METRICS.items()},
            }
        return result
