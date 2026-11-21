"""Sample-time resolution shared by the engine, the exporter and the canvas.

Two pure steps, kept free of engine state so every consumer applies the same
rule:

* :func:`declared_sample_time` reads the period a block *declares*
  (``sampling_time`` / ``sample_time``): ``> 0`` a discrete rate, ``0`` inherit,
  ``< 0`` continuous.  A literal, a workspace variable name and an expression
  must give the same number, so a string that is not a literal is read from
  the block's *resolved* parameters (``exec_params``).  A period that cannot be
  resolved to a finite real number raises :class:`SampleTimeError` -- it used
  to fall back to continuous silently, which changed the modelled system.

* :func:`propagate_sample_times` turns declared rates into effective ones.  An
  inheriting block (0) takes the fastest (smallest positive) rate among its
  inputs; continuous inputs do not count.  The result is a fixed point over the
  whole graph, so it does not depend on the order the blocks are stored in:
  an inheriting block is "unknown" until every pass agrees, and only a block
  that still has no discrete rate reachable through inheriting predecessors
  falls back to continuous (-1).
"""

import math
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np

#: Parameter names a block may declare its period under, in lookup order.
SAMPLE_TIME_KEYS = ("sampling_time", "sample_time")

CONTINUOUS = -1.0
INHERITED = 0.0


class SampleTimeError(ValueError):
    """A block's sample period is not a finite real number."""

    def __init__(self, message: str, block: str = ""):
        super().__init__(message)
        self.block = block


def _as_period(value: Any) -> Optional[float]:
    """``value`` as a float period, or None when it is not a real scalar."""
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value)
        if arr.size != 1 or arr.dtype == object:
            return None
        value = arr.ravel()[0]
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def declared_sample_time(
    params: Optional[Mapping[str, Any]],
    resolved: Optional[Mapping[str, Any]] = None,
    block_name: str = "",
) -> float:
    """The sample period a block declares (-1 continuous, 0 inherit, > 0 rate).

    Args:
        params: the block's stored parameters (may hold an expression string).
        resolved: the same parameters resolved against the workspace (and any
            enclosing mask), e.g. ``exec_params``.  Consulted only when the
            stored value is not itself a number, so a stale resolution never
            overrides a literal the user has since typed.
        block_name: used in the error message.

    Raises:
        SampleTimeError: the period is a name/expression that did not resolve,
            or resolves to something that is not a finite real scalar.
    """
    params = params or {}
    key = next((k for k in SAMPLE_TIME_KEYS if k in params), None)
    if key is None:
        return CONTINUOUS
    raw = params[key]
    period = _as_period(raw)
    if period is None and isinstance(raw, str) and resolved is not None and key in resolved:
        value = resolved[key]
        if not isinstance(value, str):
            period = _as_period(value)
    label = f"Block '{block_name}': " if block_name else ""
    if period is None:
        if isinstance(raw, str):
            raise SampleTimeError(
                f"{label}sample period {key}={raw!r} could not be resolved to a number "
                f"(undefined workspace variable or invalid expression)",
                block_name,
            )
        raise SampleTimeError(
            f"{label}sample period {key}={raw!r} is not a real number", block_name
        )
    if not math.isfinite(period):
        raise SampleTimeError(
            f"{label}sample period {key}={raw!r} is not finite ({period})", block_name
        )
    return period


def propagate_sample_times(
    declared: Mapping[str, float], sources: Mapping[str, Iterable[str]]
) -> Dict[str, float]:
    """Effective sample time per block from declared rates and the graph.

    Args:
        declared: ``{block: declared rate}`` (see :func:`declared_sample_time`).
        sources: ``{block: names of the blocks feeding its inputs}``.

    Returns:
        ``{block: effective rate}``: declared non-zero rates are kept; each
        inheriting block gets the smallest positive rate reachable backwards
        through inheriting blocks, or -1 when there is none.
    """
    rates: Dict[str, float] = {name: float(rate) for name, rate in declared.items()}
    inherited = [name for name, rate in rates.items() if rate == INHERITED]
    srcs = {name: [s for s in sources.get(name, ()) if s in rates] for name in inherited}
    # Unknown = +inf; values only ever decrease, so this terminates and the
    # fixed point (the minimum over reachable discrete rates) is independent
    # of the iteration order.
    best = {name: math.inf for name in inherited}

    def current(name: str) -> float:
        rate = rates[name]
        if rate == INHERITED:
            return best[name]
        return rate if rate > 0 else math.inf

    changed = True
    while changed:
        changed = False
        for name in inherited:
            fastest = min((current(s) for s in srcs[name]), default=math.inf)
            if fastest < best[name]:
                best[name] = fastest
                changed = True

    for name in inherited:
        rates[name] = best[name] if math.isfinite(best[name]) else CONTINUOUS
    return rates
