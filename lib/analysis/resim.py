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

from lib.engine.block_params import runtime_params
from lib.i18n import tr_noop


# Per-run outcome metrics. Each maps an ensemble matrix ``M`` of shape
# (n_runs, L) -- one row per run -- to a length-``n_runs`` vector holding one
# scalar summary per run. Insertion order is the order they appear in metric
# pickers across the UI.
#
# The dict keys are identifiers: they are used verbatim as lookup keys into
# ``sig["metrics"]`` dicts and persisted nowhere, but widgets read them back
# (e.g. via ``QComboBox`` item data), so they must never be translated.
OUTCOME_METRICS = {
    "final": lambda M: np.asarray(M)[:, -1],
    "mean": lambda M: np.asarray(M).mean(axis=1),
    "max": lambda M: np.asarray(M).max(axis=1),
    "min": lambda M: np.asarray(M).min(axis=1),
    "peak-to-peak": lambda M: np.ptp(np.asarray(M), axis=1),
    "rms": lambda M: np.sqrt((np.asarray(M) ** 2).mean(axis=1)),
}

# Human-readable label for each :data:`OUTCOME_METRICS` key, shown in combo
# boxes and plot titles/axes. Declared here (next to the keys they describe)
# but translated at *display time* by the widget via ``tr(label)`` -- this
# module stays Qt-free, so it only marks the literals for extraction with
# ``tr_noop`` (see ``lib.i18n.tr_noop``) rather than translating them itself.
OUTCOME_METRIC_LABELS = {
    "final": tr_noop("Final value"),
    "mean": tr_noop("Mean"),
    "max": tr_noop("Maximum"),
    "min": tr_noop("Minimum"),
    "peak-to-peak": tr_noop("Peak-to-peak"),
    "rms": tr_noop("RMS"),
}


def iter_blocks_qualified(blocks, prefix=""):
    """Yield ``(qualified_name, block)`` for ``blocks`` and everything nested in them.

    Descends into Subsystems via ``sub_blocks``.  A nested block's qualified
    name is ``"<Subsystem>/<block>"`` -- the same name the Flattener gives its
    primitive copy -- so it identifies the same block in results and in the
    ``derive_seed`` tag.  Top-level blocks keep their plain name.  Container
    blocks are yielded too, before their children.
    """
    for b in blocks:
        full = f"{prefix}{b.name}"
        yield full, b
        children = getattr(b, "sub_blocks", None)
        if isinstance(children, (list, tuple)) and children:
            yield from iter_blocks_qualified(children, f"{full}/")


def harvest_scope_signals(dsim):
    """Read each Scope's trace(s) into ``{'timeline', 'signals'}`` (or ``None``).

    Returns ``{'timeline': 1-D ndarray, 'signals': {name: 1-D ndarray}}`` after a
    completed headless run, or ``None`` if no timeline is available. Every
    channel is keyed by its Scope label (``vec_labels[j]``: the user's
    ``labels`` entry, or the ``<scope>-<j>`` default), on both execution paths.
    Only a channel without a label falls back to the block name (``<scope>``
    for a single channel, ``<scope>[j]`` otherwise); duplicate names are
    disambiguated with a ``#n`` suffix.
    """
    timeline = getattr(dsim, "timeline", None)
    if timeline is None or len(np.atleast_1d(timeline)) == 0:
        return None
    blocks = getattr(dsim.engine, "active_blocks_list", None) or dsim.blocks_list
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
        if b.block_fn != "Scope":
            continue
        params = runtime_params(b)
        vec = params.get("vector")
        if vec is None:
            continue
        arr = np.asarray(vec, dtype=float)
        vec_dim = int(params.get("vec_dim", 1) or 1)
        # The interpreter keeps a flat, vec_dim-strided sample buffer; the
        # compiled replay preallocates ``(n_samples, vec_dim)``. Normalise both
        # to 2-D so the per-channel loop below is the only naming rule -- a
        # single-channel Scope must come back under the same key either way.
        if arr.ndim == 1:
            if vec_dim > 1 and arr.size % vec_dim == 0:
                arr = arr.reshape(-1, vec_dim)
            else:
                arr = arr.reshape(-1, 1)
        labels = params.get("vec_labels")
        if isinstance(labels, str):
            labels = [labels]
        elif not isinstance(labels, (list, tuple)):
            labels = []
        n_channels = arr.shape[1]
        for j in range(n_channels):
            label = labels[j] if j < len(labels) else None
            if not isinstance(label, str) or not label:
                label = b.name if n_channels == 1 else f"{b.name}[{j}]"
            put(label, arr[:, j])
    return {"timeline": np.asarray(timeline, dtype=float), "signals": signals}
