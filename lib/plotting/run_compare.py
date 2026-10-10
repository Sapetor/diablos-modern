"""Parameter snapshots for comparing a Scope run against a kept reference run.

``snapshot_params`` records the user-facing block parameters of a diagram at
the time of a run; ``param_changes`` lists what differs between two snapshots,
so the Scope can say what changed between the reference and the current run.
"""

from __future__ import annotations

import copy
from typing import Dict, Iterable, List

import numpy as np


def snapshot_params(blocks: Iterable) -> Dict[str, Dict[str, object]]:
    """``{block label: {param: value}}`` without internal ``_name_`` keys.

    Keyed by the label shown on the canvas (``username``), falling back to the
    internal name, so change lines read like the diagram.
    """
    snap = {}
    for block in blocks:
        params = getattr(block, "params", None) or {}
        snap[getattr(block, "username", None) or block.name] = {
            k: copy.deepcopy(v) for k, v in params.items() if not str(k).startswith("_")
        }
    return snap


def _same(a, b) -> bool:
    try:
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            return np.array_equal(np.asarray(a), np.asarray(b))
        return bool(a == b)
    except Exception:
        return repr(a) == repr(b)


def _fmt(v, limit=24) -> str:
    text = repr(v.tolist()) if isinstance(v, np.ndarray) else str(v)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def param_changes(reference: Dict, current: Dict) -> List[str]:
    """Readable ``"Block.param: old -> new"`` lines, in ``current`` order.

    Blocks added or removed since the reference are listed as such.
    """
    lines = []
    for name, params in current.items():
        ref = reference.get(name)
        if ref is None:
            lines.append(f"{name}: added")
            continue
        for key, value in params.items():
            if key in ref and not _same(ref[key], value):
                lines.append(f"{name}.{key}: {_fmt(ref[key])} -> {_fmt(value)}")
    for name in reference:
        if name not in current:
            lines.append(f"{name}: removed")
    return lines
