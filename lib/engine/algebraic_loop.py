"""Algebraic-loop solving for the interpreter.

An algebraic loop is a cycle of blocks with direct feedthrough and no memory
block to break it, e.g. ``y = u - k*y``. Each loop is one strongly connected
component (:func:`lib.engine.topo.cyclic_components`) and is executed as a
unit: the unknowns ``z`` are the outputs of the member ports that feed other
members, and each step solves the fixed point ``z = F(z)``, where ``F`` runs
every member once with its loop inputs taken from ``z``.

Members may be evaluated many times per step, so every evaluation starts from
a snapshot of the members' ``params``/``exec_params``. The block contract keeps
all cross-step state there (see CLAUDE.md, "Critical rule"), so trial
evaluations leave no trace, and a feedthrough block with state (a proper
transfer function) is safe. After convergence the members run once more from
the snapshot with the solved inputs, which is the step's real execution.

Not supported (the engine keeps rejecting these, naming the block): members
with a discrete sample time, whose sample schedule lives on the DBlock rather
than in params, and external (stub) blocks.
"""

import copy
import logging

import numpy as np
from scipy.optimize import root

logger = logging.getLogger(__name__)

RTOL = 1e-10
ATOL = 1e-12


class AlgebraicLoopError(Exception):
    """A loop that cannot be solved; ``block`` names the member to highlight."""

    def __init__(self, message, block=""):
        super().__init__(message)
        self.block = block


def _optional_inputs(block):
    instance = getattr(block, "block_instance", None)
    if instance is not None and hasattr(instance, "optional_inputs"):
        return set(instance.optional_inputs)
    return set()


class AlgebraicLoop:
    """One strongly connected algebraic component, solved as a unit."""

    def __init__(self, members, lines):
        self.members = list(members)
        self.names = [b.name for b in self.members]
        name_set = set(self.names)
        self.internal = [
            (line.srcblock, line.srcport, line.dstblock, line.dstport)
            for line in lines
            if line.srcblock in name_set and line.dstblock in name_set
        ]
        # One unknown per distinct (member, output port) that feeds the loop.
        self.slots = sorted({(src, port) for src, port, _, _ in self.internal})
        internal_ports = {(dst, port) for _, _, dst, port in self.internal}
        # Required ports fed from outside the loop: non-optional, no internal wire.
        self._external_required = {
            b.name: sum(
                1
                for p in range(b.in_ports)
                if p not in _optional_inputs(b) and (b.name, p) not in internal_ports
            )
            for b in self.members
        }
        self._shapes = None
        self._z = None  # warm start: the previous step's solution

    def __repr__(self):
        return f"AlgebraicLoop({self.names})"

    def check_supported(self):
        """Raise AlgebraicLoopError for a member the solver cannot iterate."""
        for b in self.members:
            if getattr(b, "external", False):
                raise AlgebraicLoopError(
                    f"Algebraic loop {self.names} contains an external block ({b.name})", b.name
                )
            if getattr(b, "effective_sample_time", -1) > 0:
                raise AlgebraicLoopError(
                    f"Algebraic loop {self.names} contains the discrete block {b.name}; "
                    "loops through discrete-rate blocks are not supported. "
                    "Break the loop with a Delay or a strictly proper block.",
                    b.name,
                )

    def is_pending(self):
        return any(not b.computed_data for b in self.members)

    def is_ready(self):
        """Every member has received all of its inputs from outside the loop."""
        return all(b.data_received >= self._external_required[b.name] for b in self.members)

    # ------------------------------------------------------------------ solve

    def solve(self, execute):
        """Solve the loop; return ``{member_name: out_value}`` from the real run.

        ``execute(block) -> out dict`` runs one block (SimulationEngine.execute_block).
        Raises AlgebraicLoopError when a member fails or Newton does not converge.
        """
        snapshot = {
            b.name: (copy.deepcopy(b.params), copy.deepcopy(b.exec_params)) for b in self.members
        }
        ext_inputs = {b.name: dict(b.input_queue) for b in self.members}

        def run(z_parts):
            outs = {}
            for b in self.members:
                # Restore in place: the engine may hold references to these dicts.
                for live, saved in zip((b.params, b.exec_params), snapshot[b.name]):
                    if live is None:
                        continue
                    live.clear()
                    live.update(copy.deepcopy(saved))
                b.input_queue = dict(ext_inputs[b.name])
            for src, sport, dst, dport in self.internal:
                self._block(dst).input_queue[dport] = z_parts[(src, sport)]
            for b in self.members:
                out = execute(b)
                if not isinstance(out, dict) or out.get("E"):
                    msg = out.get("error", "block error") if isinstance(out, dict) else repr(out)
                    raise AlgebraicLoopError(
                        f"{b.name} failed inside algebraic loop: {msg}", b.name
                    )
                outs[b.name] = out
            return outs

        def feedback(outs):
            return {
                (src, port): np.asarray(outs[src][port], dtype=float) for src, port in self.slots
            }

        if self._shapes is None:
            # Discover signal shapes with one fixed-point step from zero; it
            # doubles as the first initial guess.
            first = feedback(run({slot: 0.0 for slot in self.slots}))
            self._shapes = {slot: np.shape(v) for slot, v in first.items()}
            self._z = self._ravel(first)

        def residual(zflat):
            return self._ravel(feedback(run(self._unravel(zflat)))) - zflat

        z0 = self._z if self._z is not None else np.zeros(self._size())
        # Subnormal leftovers of a zero solution make MINPACK's difference
        # step zero-width (see lib/engine/compiled_loops._flush_tiny).
        z0 = np.where(np.abs(z0) < 1e-100, 0.0, z0)
        r0 = residual(z0)
        if np.all(np.abs(r0) <= ATOL + RTOL * np.abs(z0)):
            z = z0
        else:
            sol = root(residual, z0, method="hybr", options={"xtol": RTOL})
            z = sol.x
            r = residual(z)
            if not np.all(np.isfinite(r)) or np.max(np.abs(r)) > 1e-8 * (1.0 + np.max(np.abs(z))):
                raise AlgebraicLoopError(
                    f"Algebraic loop {self.names} did not converge "
                    f"(residual {np.max(np.abs(r)):.3g}): {sol.message}",
                    self.names[0],
                )
        self._z = z
        return run(self._unravel(z))

    # ------------------------------------------------------------------ helpers

    def _block(self, name):
        return self.members[self.names.index(name)]

    def _size(self):
        return int(sum(int(np.prod(s)) if s else 1 for s in self._shapes.values()))

    def _ravel(self, parts):
        return np.concatenate([np.ravel(parts[slot]).astype(float) for slot in self.slots])

    def _unravel(self, zflat):
        parts, i = {}, 0
        for slot in self.slots:
            shape = self._shapes[slot]
            n = int(np.prod(shape)) if shape else 1
            chunk = zflat[i : i + n]
            parts[slot] = chunk.reshape(shape) if shape else np.array([chunk[0]])
            i += n
        return parts
