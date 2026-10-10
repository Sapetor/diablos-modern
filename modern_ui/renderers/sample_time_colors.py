"""Sample-time colors shared by the block rate dot and the wires.

The engine resolves rates only when a run starts
(``SimulationEngine`` sample-time propagation), so a diagram that was never
run would show no wire colors. ``wire_sample_times`` repeats the same rule
on the canvas side: a block's declared rate comes from
``DBlock.resolve_sample_time()``; an inherited block (0) takes the fastest
discrete rate among its inputs, or becomes continuous; a wire carries the
rate of its source block.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Optional

from PyQt6.QtGui import QColor

INHERITED_COLOR = QColor(128, 128, 128)


def rate_color(sample_time: float) -> Optional[QColor]:
    """Color for a sample period: None for continuous (< 0), gray for
    inherited (0), red (1 ms, fast) to blue (1 s, slow) on a log scale."""
    if sample_time < 0:
        return None
    if sample_time == 0:
        return QColor(INHERITED_COLOR)
    log_min, log_max = math.log10(0.001), math.log10(1.0)
    log_sample = math.log10(max(0.001, min(1.0, sample_time)))
    t = max(0.0, min(1.0, (log_sample - log_min) / (log_max - log_min)))
    r = int(255 * (1 - t))
    g = int(100 * (1 - abs(t - 0.5) * 2))  # green peak in the middle
    b = int(255 * t)
    return QColor(r, g, b)


def _declared(block) -> float:
    try:
        return float(block.resolve_sample_time())
    except (AttributeError, TypeError, ValueError):
        return -1.0


def _output_rate(block) -> float:
    """Rate of a block's output: a continuous-running block with an
    ``output_sample_time`` (RateTransition) emits at that rate."""
    try:
        rate = float(block.params.get("output_sample_time", -1.0))
    except (AttributeError, TypeError, ValueError):
        return -1.0
    return rate if rate > 0 else -1.0


def block_sample_times(blocks: Iterable, lines: Iterable) -> Dict[str, float]:
    """Effective sample time per block name (-1 continuous, > 0 discrete)."""
    blocks = list(blocks)
    rates = {b.name: _declared(b) for b in blocks}
    sources: Dict[str, list] = {b.name: [] for b in blocks}
    for line in lines:
        if line.dstblock in sources:
            sources[line.dstblock].append(line.srcblock)
    inherited = [name for name, r in rates.items() if r == 0]
    for _ in range(len(inherited) + 1):
        changed = False
        for name in inherited:
            fastest = -1.0
            for src in sources[name]:
                r = rates.get(src, -1.0)
                if r > 0 and (fastest < 0 or r < fastest):
                    fastest = r
            if fastest != rates[name]:
                rates[name] = fastest
                changed = True
        if not changed:
            break
    return rates


def wire_sample_times(blocks: Iterable, lines: Iterable) -> Dict[int, float]:
    """Sample time carried by each wire, keyed by ``id(line)``."""
    blocks = list(blocks)
    lines = list(lines)
    rates = block_sample_times(blocks, lines)
    by_name = {b.name: b for b in blocks}
    out = {}
    for line in lines:
        rate = rates.get(line.srcblock, -1.0)
        if rate <= 0 and line.srcblock in by_name:
            rate = _output_rate(by_name[line.srcblock])
        out[id(line)] = rate
    return out
