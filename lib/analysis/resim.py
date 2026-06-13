"""Shared re-simulation utilities for ensemble / sweep studies.

Both the Monte-Carlo runner (``lib/analysis/monte_carlo.py``) and the parameter
sweep runner (``lib/analysis/parameter_sweep.py``) re-run a diagram many times on
the headless ``DSim.run_tuning_simulation`` path and then reduce the resulting
Scope traces. The two pieces they share live here:

  * :func:`harvest_scope_signals` -- read every Scope block's buffer into a flat
    ``{signal_name: 1-D ndarray}`` dict plus the run timeline.
  * :data:`OUTCOME_METRICS` -- per-run scalar reductions of an ensemble matrix
    (final value, peak, rms, ...), used both to summarise an ensemble and to
    color/heatmap a sweep.

Keeping them in one module guarantees the Monte-Carlo histograms and the sweep
heatmaps speak about *the same* metric definitions.
"""

import numpy as np


# Per-run outcome metrics. Each maps an ensemble matrix ``M`` of shape
# (n_runs, L) -- one row per run -- to a length-``n_runs`` vector holding one
# scalar summary per run. Insertion order is the order they appear in metric
# pickers across the UI.
OUTCOME_METRICS = {
    "final": lambda M: np.asarray(M)[:, -1],
    "mean": lambda M: np.asarray(M).mean(axis=1),
    "max": lambda M: np.asarray(M).max(axis=1),
    "min": lambda M: np.asarray(M).min(axis=1),
    "peak-to-peak": lambda M: np.ptp(np.asarray(M), axis=1),
    "rms": lambda M: np.sqrt((np.asarray(M) ** 2).mean(axis=1)),
}


def harvest_scope_signals(dsim):
    """Read each Scope's trace(s) into ``{'timeline', 'signals'}`` (or ``None``).

    Returns ``{'timeline': 1-D ndarray, 'signals': {name: 1-D ndarray}}`` after a
    completed headless run, or ``None`` if no timeline is available. Multi-channel
    Scope buffers are reshaped and split per channel; duplicate signal names are
    disambiguated with a ``#n`` suffix.
    """
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
