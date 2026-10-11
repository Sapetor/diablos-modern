"""Algebraic loops inside the compiled right-hand side.

The compiled RHS evaluates the middle group (algebraic blocks and feedthrough
state blocks) in dataflow order, every block reading ``signals`` written
earlier in the same evaluation. A cycle among those blocks -- ``y = u - k*y``,
or a loop through a proper transfer function -- has no such order. Each cycle
is one strongly connected component (:func:`lib.engine.topo.cyclic_components`)
and runs as a unit: :func:`order_units` places it where a single block would
go, and :func:`make_loop_executor` solves it at every RHS evaluation.

The unknowns are the loop's internal signals (one per member output port that
feeds another member). ``F(z)`` writes ``z`` into those signal keys, runs the
member kernels once, and reads the keys back; ``scipy.optimize.root`` (hybr)
solves ``F(z) = z`` to ``XTOL``, warm-started from the previous evaluation's
solution. The warm start only picks the initial guess, so as long as the loop
has a unique solution the RHS stays a function of ``(t, y)`` to solver
precision, which the adaptive integrator and the linearizer's finite
differences both need. The members then run once more on the solution: that
pass writes their outputs and any ``dy_vec`` entries (no kernel accumulates
into ``dy_vec``).

A loop that does not converge raises :class:`CompiledLoopError`; the compiled
run fails and ``DSim`` falls through to the interpreter, which has its own
loop solver and reports the failure as before.
"""

import numpy as np
from scipy.optimize import root

from lib.engine.topo import cyclic_components, kahn_topological_order

XTOL = 1e-12


class CompiledLoopError(RuntimeError):
    """An algebraic loop on the compiled path did not converge."""


def _signal_key(src, port):
    return src if not port else f"{src}_out{port}"


def _flush_tiny(z):
    """Zero out subnormal-scale entries of a warm start.

    MINPACK's forward-difference step is ``sqrt(eps) * |x_j|`` (``sqrt(eps)``
    only when ``x_j == 0``), so a guess like ``-5e-324`` left over from a
    solution that was zero yields a zero-width difference, a singular
    Jacobian and "not making good progress".
    """
    z = np.array(z, dtype=float)
    z[np.abs(z) < 1e-100] = 0.0
    return z


def order_units(middle, input_map):
    """Middle-group execution units in dataflow order.

    Returns a list whose items are blocks, or lists of blocks for each
    algebraic loop (members in ``middle`` order). Only middle-to-middle edges
    constrain, as in ``system_compiler._dataflow_order``.
    """
    names = [b.name for b in middle]
    present = set(names)
    succ = {n: [] for n in names}
    for dst in names:
        for src, _port in input_map.get(dst, {}).values():
            if src in present:
                succ[src].append(dst)
    loops = cyclic_components(names, succ)
    unit_of = {n: n for n in names}
    for i, comp in enumerate(loops):
        for n in comp:
            unit_of[n] = ("loop", i)
    unit_nodes = []
    for n in names:
        if unit_of[n] not in unit_nodes:
            unit_nodes.append(unit_of[n])
    unit_succ = {u: [] for u in unit_nodes}
    for src in names:
        for dst in succ[src]:
            if unit_of[src] != unit_of[dst]:
                unit_succ[unit_of[src]].append(unit_of[dst])
    order, rest = kahn_topological_order(unit_nodes, unit_succ)
    by_name = {b.name: b for b in middle}
    units = []
    for u in order + rest:
        if isinstance(u, tuple):
            units.append([by_name[n] for n in loops[u[1]]])
        else:
            units.append(by_name[u])
    return units


def make_loop_executor(members, executors, input_map):
    """One executor that solves the loop ``members`` (see module docstring)."""
    member_names = [b.name for b in members]
    in_loop = set(member_names)
    slots = sorted(
        {
            (src, port)
            for dst in member_names
            for src, port in input_map.get(dst, {}).values()
            if src in in_loop
        }
    )
    keys = [_signal_key(src, port) for src, port in slots]
    state = {"shapes": None, "z": None}

    def run(t, y, dy_vec, signals, z_parts):
        for key, value in zip(keys, z_parts):
            signals[key] = value
        for executor in executors:
            executor(t, y, dy_vec, signals)
        return [np.asarray(signals.get(key, 0.0), dtype=float) for key in keys]

    def ravel(parts):
        return np.concatenate([np.ravel(p) for p in parts])

    def unravel(flat):
        parts, i = [], 0
        for shape in state["shapes"]:
            n = int(np.prod(shape)) if shape else 1
            chunk = flat[i : i + n]
            parts.append(chunk.reshape(shape) if shape else float(chunk[0]))
            i += n
        return parts

    def exec_loop(t, y, dy_vec, signals):
        if state["shapes"] is None:
            first = run(t, y, dy_vec, signals, [0.0] * len(keys))
            state["shapes"] = [np.shape(v) for v in first]
            state["z"] = ravel(first)

        def residual(z):
            return ravel(run(t, y, dy_vec, signals, unravel(z))) - z

        z0 = _flush_tiny(state["z"])
        r0 = residual(z0)
        if np.all(np.abs(r0) <= XTOL * (1.0 + np.abs(z0))):
            z = z0
        else:
            # A retry from the fixed-point image F(z0) rescues a warm start
            # that is far off after a jump in the loop's inputs.
            for guess in (z0, _flush_tiny(z0 + r0)):
                sol = root(residual, guess, method="hybr", options={"xtol": XTOL})
                z = sol.x
                r = residual(z)
                ok = np.all(np.isfinite(r)) and np.max(np.abs(r)) <= 1e-9 * (
                    1.0 + np.max(np.abs(z))
                )
                if ok:
                    break
            else:
                raise CompiledLoopError(
                    f"Algebraic loop {member_names} did not converge at t={t:.6g} "
                    f"(residual {np.max(np.abs(r)):.3g}): {sol.message}"
                )
        state["z"] = z
        run(t, y, dy_vec, signals, unravel(z))

    exec_loop.loop_members = member_names
    return exec_loop
