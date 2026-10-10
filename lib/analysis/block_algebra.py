"""Closed-loop transfer function between two blocks, with reduction steps.

Pure numeric rational-function algebra (numpy polynomials in ``s``; no SymPy,
no engine imports). Every supported block has numeric coefficients, so the
result is an exact rational function ``G(s) = N(s)/D(s)`` up to float
round-off.

The diagram becomes a signal-flow graph: nodes are block output signals
``(block, port)``; a linear block adds an edge from the signal on each input
port to its output, weighted by that port's gain. The chosen source's output
is ``U`` (its own dynamics are ignored); every other independent source is
zeroed (superposition). Mason's gain formula then gives

    G = sum_k P_k * Delta_k / Delta,
    Delta = 1 - sum(L) + sum(non-touching pairs) - ...

and the forward paths ``P_k``, loops ``L``, ``Delta`` and ``Delta_k`` are the
"reduction steps" shown to the user.

Supported blocks: Gain (scalar), Sum, TranFn, StateSpace (SISO), Integrator,
Deriv, PID (``Kp + Ki/s + Kd N s / (s + N)``, both the error-input and the
setpoint/measurement wiring; output limits and anti-windup are ignored), all
continuous (``sampling_time <= 0``). Any other block on a path or loop
between the source and the output raises ``AlgebraError`` naming it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

MAX_LOOPS = 100
Signal = Tuple[str, int]  # (block name, output port)


class AlgebraError(ValueError):
    """The transfer function cannot be formed (unsupported block, no path...)."""


# ----------------------------------------------------------------- rationals
def _trim(p):
    p = np.atleast_1d(np.asarray(p, dtype=float))
    nz = np.flatnonzero(np.abs(p) > 1e-14 * max(1.0, np.max(np.abs(p))))
    return p[nz[0] :] if nz.size else np.zeros(1)


@dataclass(frozen=True)
class Rational:
    """``num(s) / den(s)``, coefficients highest power first."""

    num: tuple
    den: tuple

    @staticmethod
    def of(num, den=(1.0,)) -> "Rational":
        num, den = _trim(num), _trim(den)
        if not np.any(den):
            raise AlgebraError("zero denominator")
        return Rational(tuple(num / den[0]), tuple(den / den[0]))

    def __add__(self, o):
        o = _as_rational(o)
        return Rational.of(
            np.polyadd(np.polymul(self.num, o.den), np.polymul(o.num, self.den)),
            np.polymul(self.den, o.den),
        )

    __radd__ = __add__

    def __neg__(self):
        return Rational.of(-np.asarray(self.num), self.den)

    def __sub__(self, o):
        return self + (-_as_rational(o))

    def __rsub__(self, o):
        return _as_rational(o) - self

    def __mul__(self, o):
        o = _as_rational(o)
        return Rational.of(np.polymul(self.num, o.num), np.polymul(self.den, o.den))

    __rmul__ = __mul__

    def __truediv__(self, o):
        o = _as_rational(o)
        if not np.any(o.num):
            raise AlgebraError("division by zero (Delta = 0: the loop is degenerate)")
        return Rational.of(np.polymul(self.num, o.den), np.polymul(self.den, o.num))

    def is_zero(self) -> bool:
        return not np.any(self.num)

    def minreal(self, tol=1e-6) -> "Rational":
        """Cancel numerator/denominator roots that coincide within ``tol``."""
        num, den = np.asarray(self.num), np.asarray(self.den)
        if num.size < 2 or den.size < 2:
            return self
        zeros, poles = list(np.roots(num)), list(np.roots(den))
        kept_z = []
        for z in zeros:
            match = next(
                (i for i, p in enumerate(poles) if abs(z - p) <= tol * max(1.0, abs(p))), None
            )
            if match is None:
                kept_z.append(z)
            else:
                poles.pop(match)
        gain = num[0] / den[0]
        return Rational.of(gain * np.real(np.poly(kept_z)), np.real(np.poly(poles)))

    def dc_gain(self) -> float:
        d = self.den[-1]
        if abs(d) < 1e-12:
            return float("inf") if abs(self.num[-1]) > 1e-12 else float("nan")
        return float(self.num[-1] / d)


def _as_rational(x) -> Rational:
    return x if isinstance(x, Rational) else Rational.of([float(x)])


ONE = Rational.of([1.0])
S = Rational.of([1.0, 0.0])


# ------------------------------------------------------------- block models
def _param(block, key, default=None):
    params = getattr(block, "params", None) or {}
    return params.get(key, default)


def _continuous(block):
    try:
        ts = float(_param(block, "sampling_time", -1.0))
    except (TypeError, ValueError):
        ts = -1.0
    if ts > 0:
        raise AlgebraError(
            f"{_label(block)} is sampled (sampling_time = {ts}); only continuous blocks are supported"
        )


def _scalar(value, what, block) -> float:
    arr = np.atleast_1d(np.asarray(value, dtype=float))
    if arr.size != 1:
        raise AlgebraError(f"{_label(block)}: {what} must be a scalar for a SISO transfer function")
    return float(arr.ravel()[0])


def _label(block) -> str:
    user = getattr(block, "username", "") or block.name
    return f'{block.block_fn} "{user}"'


def port_gains(block, n_inputs: int) -> Dict[int, Rational]:
    """Gain from each input port to output port 0. Raises for unsupported blocks."""
    fn = block.block_fn
    if fn == "Gain":
        return {0: Rational.of([_scalar(_param(block, "gain", 1.0), "gain", block)])}
    if fn == "Sum":
        signs = str(_param(block, "signs", _param(block, "sign", "++")))
        return {
            i: Rational.of([1.0 if (signs[i] if i < len(signs) else "+") == "+" else -1.0])
            for i in range(max(n_inputs, len(signs)))
            if (signs[i] if i < len(signs) else "+") in "+-"
        }
    if fn == "TranFn":
        _continuous(block)
        return {
            0: Rational.of(
                _param(block, "numerator", [1.0]), _param(block, "denominator", [1.0, 1.0])
            )
        }
    if fn == "StateSpace":
        from scipy import signal

        a, b, c, d = (np.atleast_2d(np.asarray(_param(block, k), dtype=float)) for k in "ABCD")
        if b.shape[1] != 1 or c.shape[0] != 1:
            raise AlgebraError(f"{_label(block)} is not single-input single-output")
        num, den = signal.ss2tf(a, b, c, d)
        return {0: Rational.of(np.ravel(num), den)}
    if fn == "Integrator":
        _continuous(block)
        return {0: Rational.of([1.0], [1.0, 0.0])}
    if fn == "Deriv":
        _continuous(block)
        return {0: S}
    if fn == "PID":
        _continuous(block)
        kp, ki, kd = (float(_param(block, k, 0.0) or 0.0) for k in ("Kp", "Ki", "Kd"))
        n = float(_param(block, "N", 20.0) or 20.0)
        c = Rational.of([kp])
        if ki:
            c = c + Rational.of([ki], [1.0, 0.0])
        if kd:
            c = c + Rational.of([kd * n, 0.0], [1.0, n])
        return {0: c, 1: -c}
    raise AlgebraError(f"{_label(block)} is not a supported linear block")


# -------------------------------------------------------------- the graph
@dataclass
class Steps:
    """Mason's formula, term by term (for display)."""

    paths: List[Tuple[List[str], Rational]] = field(default_factory=list)
    loops: List[Tuple[List[str], Rational]] = field(default_factory=list)
    delta: Optional[Rational] = None
    path_deltas: List[Rational] = field(default_factory=list)


@dataclass
class Result:
    tf: Rational
    steps: Steps
    source: str
    output: str

    @property
    def poles(self):
        return np.roots(self.tf.den) if len(self.tf.den) > 1 else np.array([])

    @property
    def zeros(self):
        return np.roots(self.tf.num) if len(self.tf.num) > 1 else np.array([])


def _graph(blocks, lines, source: Signal):
    """Edges ``{src_signal: [(dst_signal, gain), ...]}``; unsupported blocks
    become poisoned nodes (error raised only if they are actually reached)."""
    by_name = {b.name: b for b in blocks}
    feeds: Dict[str, Dict[int, List[Signal]]] = {}
    for ln in lines:
        feeds.setdefault(ln.dstblock, {}).setdefault(int(ln.dstport), []).append(
            (ln.srcblock, int(ln.srcport))
        )
    edges: Dict[Signal, List[Tuple[Signal, Rational]]] = {}
    poisoned: Dict[Signal, str] = {}
    for name, ports in feeds.items():
        block = by_name.get(name)
        if block is None or name == source[0]:
            continue
        out = (name, 0)
        try:
            gains = port_gains(block, max(ports) + 1)
        except AlgebraError as exc:
            poisoned[out] = str(exc)
            gains = {p: ONE for p in ports}  # keep connectivity; never evaluated
        for port, srcs in ports.items():
            if port not in gains:
                continue
            for src in srcs:
                edges.setdefault(src, []).append((out, gains[port]))
    return edges, poisoned


def _forward_paths(edges, start, goal):
    paths, stack = [], [(start, [start])]
    while stack:
        node, path = stack.pop()
        if node == goal and len(path) > 1:
            paths.append(path)
            continue
        for nxt, _ in edges.get(node, []):
            if nxt == goal or nxt not in path:
                stack.append((nxt, path + [nxt]))
    return sorted(paths, key=lambda p: (len(p), [str(x) for x in p]))


def _loops(edges, nodes):
    """Simple cycles, each reported once (rotated to start at its least node)."""
    order = {n: i for i, n in enumerate(sorted(nodes, key=str))}
    found = []
    for start in sorted(nodes, key=str):
        stack = [(start, [start])]
        while stack:
            node, path = stack.pop()
            for nxt, _ in edges.get(node, []):
                if nxt not in order:
                    continue
                if nxt == start:
                    found.append(path)
                    if len(found) > MAX_LOOPS:
                        raise AlgebraError(
                            f"more than {MAX_LOOPS} feedback loops; simplify the diagram"
                        )
                elif order[nxt] > order[start] and nxt not in path:
                    stack.append((nxt, path + [nxt]))
    return found


def _gain(edges, cycle_or_path, closed):
    total = ONE
    seq = list(cycle_or_path) + ([cycle_or_path[0]] if closed else [])
    for a, b in zip(seq, seq[1:]):
        g = Rational.of([0.0])
        for dst, gain in edges.get(a, []):
            if dst == b:
                g = g + gain  # parallel wires between the same signals add up
        total = total * g
    return total


def _delta(loops: Sequence[Tuple[frozenset, Rational]]) -> Rational:
    delta, sign = ONE, -1
    for k in range(1, len(loops) + 1):
        term = Rational.of([0.0])
        any_set = False
        for combo in combinations(loops, k):
            sets = [c[0] for c in combo]
            if all(not (x & y) for x, y in combinations(sets, 2)):
                prod = ONE
                for c in combo:
                    prod = prod * c[1]
                term = term + prod
                any_set = True
        if not any_set:
            break
        delta = delta + sign * term
        sign = -sign
    return delta


def transfer_function(blocks, lines, source: str, output: Signal) -> Result:
    """``G(s)`` from ``source``'s output 0 to signal ``output`` = (block, port).

    For a sink (no outputs) pass the signal feeding the input of interest,
    see ``sink_input_signal``.
    """
    src = (source, 0)
    if output == src:  # the output is the source signal itself
        steps = Steps(paths=[([src], ONE)], delta=ONE, path_deltas=[ONE])
        return Result(tf=ONE, steps=steps, source=source, output=f"{output[0]}:{output[1]}")
    edges, poisoned = _graph(blocks, lines, src)
    paths = _forward_paths(edges, src, output)
    if not paths:
        raise AlgebraError("no signal path from the source to the output")
    relevant = {n for p in paths for n in p}
    # Loops matter only if they touch a node from which the output is reachable
    # and which the source reaches; take every node on any source->output walk.
    reach_fwd, stack = {src}, [src]
    while stack:
        for nxt, _ in edges.get(stack.pop(), []):
            if nxt not in reach_fwd:
                reach_fwd.add(nxt)
                stack.append(nxt)
    rev: Dict[Signal, List[Signal]] = {}
    for a, outs in edges.items():
        for b, _ in outs:
            rev.setdefault(b, []).append(a)
    reach_back, stack = {output}, [output]
    while stack:
        for prv in rev.get(stack.pop(), []):
            if prv not in reach_back:
                reach_back.add(prv)
                stack.append(prv)
    core = (reach_fwd & reach_back) | relevant
    loops = _loops(edges, core - {src})
    for node in sorted(core, key=str):
        if node in poisoned:
            raise AlgebraError(poisoned[node])

    loop_terms = [(frozenset(c), _gain(edges, c, closed=True)) for c in loops]
    delta = _delta(loop_terms)
    steps = Steps(delta=delta)
    total = Rational.of([0.0])
    for path in paths:
        p_gain = _gain(edges, path, closed=False)
        touching = set(path)
        dk = _delta([lt for lt in loop_terms if not (lt[0] & touching)])
        steps.paths.append((path, p_gain))
        steps.path_deltas.append(dk)
        total = total + p_gain * dk
    steps.loops = [(c, g) for c, (_, g) in zip(loops, loop_terms)]
    tf = (total / delta).minreal()
    return Result(tf=tf, steps=steps, source=source, output=f"{output[0]}:{output[1]}")


def sink_input_signal(lines, sink: str, port: int = 0) -> Signal:
    """The signal wired into input ``port`` of block ``sink``."""
    for ln in lines:
        if ln.dstblock == sink and int(ln.dstport) == port:
            return (ln.srcblock, int(ln.srcport))
    raise AlgebraError(f"nothing is connected to input {port} of {sink}")
