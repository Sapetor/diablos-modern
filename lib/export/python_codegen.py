"""Standalone Python script export ("code generation") for DiaBloS diagrams.

Turns a loaded diagram -- the same ``DBlock`` / ``DLine`` structures the
headless runner (``lib/cli.py``) and
:class:`~lib.engine.system_compiler.SystemCompiler` consume -- into the source
text of a self-contained ``.py`` file that depends only on numpy and scipy
(matplotlib is imported lazily, only when the script actually plots).

The generated script mirrors the **compiled** solver, not the interpreter: the
whole diagram becomes one ODE ``rhs(t, x)`` integrated by
``scipy.integrate.solve_ivp``.  Its ``evaluate(t, x)`` reproduces the compiled
path's three-group evaluation order documented in ``CLAUDE.md`` and implemented
in ``SystemCompiler.compile_system``:

1. outputs of strictly-proper (D = 0) state blocks are pre-populated from the
   state vector, so feedback loops resolve;
2. source blocks run;
3. algebraic blocks and feedthrough (D != 0) state blocks run in topological
   order over the feedthrough edges;
4. the strictly-proper state blocks contribute their derivatives last.

Every emitted expression is a transcription of the corresponding kernel in
``lib/engine/compiler_kernels/`` (param keys, defaults and quirks included), so
a generated script and a compiled run of the same diagram agree to solver
tolerance.  Scope traces are rebuilt by re-evaluating the diagram on the saved
time grid -- the same trick ``SimulationEngine._replay_compiled_signals`` uses
-- and are labelled exactly like ``lib/analysis/resim.harvest_scope_signals``
names them, so the script's ``--out`` CSV/NPZ lines up column-for-column with
``python -m lib.cli run``.

Only the block families that can be emitted faithfully are supported (see
:data:`SUPPORTED_FNS`); anything else raises :class:`CodegenUnsupportedError`
listing the offending blocks rather than emitting broken code.
"""

import keyword
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import signal as _scipy_signal

from lib.engine.block_names import canonical_fn
from lib.engine.topo import kahn_topological_order

__all__ = [
    "CodegenError",
    "CodegenUnsupportedError",
    "PythonCodeGenerator",
    "generate_python_script",
    "SUPPORTED_FNS",
    "SUPPORTED_BLOCK_NAMES",
]


# --------------------------------------------------------------------------
# Supported block set
# --------------------------------------------------------------------------

# Canonical fn-names (see lib.engine.block_names.canonical_fn) this exporter
# knows how to write out as plain Python.  Deliberately a subset of
# SystemCompiler.COMPILABLE_BLOCKS: a block belongs here only when its compiled
# kernel is a short, state-free expression that can be transcribed literally.
# Excluded on purpose: Noise/PRBS/WaveGenerator and Hysteresis/StateVariable
# (path- or RNG-dependent), the PDE/Field families (large method-of-lines
# stencils), MathFunction (arbitrary user expressions), RateLimiter (the
# compiled stiff-chase approximation), Exponential/Switch/Selector/Deadband/
# LogicalOperator/MatrixGain (emittable, but not yet transcribed).
SUPPORTED_FNS = frozenset(
    {
        # sources
        "Constant",
        "Step",
        "Sine",
        "Ramp",
        # algebraic
        "Gain",
        "Sum",
        "Product",
        "Sgprod",
        "Sigproduct",
        "Abs",
        "Absblock",
        "Saturation",
        # routing
        "Mux",
        "Demux",
        # state
        "Integrator",
        "TransferFcn",
        "StateSpace",
        "PID",
        # sinks
        "Scope",
        "Terminator",
        "Display",
    }
)

#: Blocks the compiled path runs first, before anything else.
SOURCE_FNS = frozenset({"Constant", "Step", "Sine", "Ramp"})

#: Blocks that allocate ODE state.
STATE_FNS = frozenset({"Integrator", "TransferFcn", "StateSpace", "PID"})

#: Blocks that consume signals but produce none.
SINK_FNS = frozenset({"Scope", "Terminator", "Display"})

#: User-facing spelling of the supported set, for error messages (SUPPORTED_FNS
#: holds canonical dispatch names such as "Sgprod").
SUPPORTED_BLOCK_NAMES = (
    "Abs",
    "Constant",
    "Demux",
    "Display",
    "Gain",
    "Integrator",
    "Mux",
    "PID",
    "Product",
    "Ramp",
    "Saturation",
    "Scope",
    "SgProd",
    "Sine",
    "StateSpace",
    "Step",
    "Sum",
    "Terminator",
    "TranFn",
)

_FIXED_STEP_METHODS = ("Euler", "RK4")
_SCIPY_METHODS = ("RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA")


class CodegenError(Exception):
    """Raised when a diagram cannot be turned into a standalone script."""


class CodegenUnsupportedError(CodegenError):
    """Raised when a diagram contains blocks the exporter cannot emit.

    ``blocks`` is a list of ``(block_name, block_fn, reason)`` triples so a UI
    can list the offenders without re-parsing the message.
    """

    def __init__(self, blocks: List[Tuple[str, str, str]]):
        self.blocks = list(blocks)
        lines = ["This diagram cannot be exported as a standalone Python script."]
        lines.append("")
        lines.append("Unsupported block(s):")
        for name, fn, reason in self.blocks:
            lines.append("  - {} ({}){}".format(name, fn, ": " + reason if reason else ""))
        lines.append("")
        lines.append("Supported blocks: " + ", ".join(SUPPORTED_BLOCK_NAMES))
        super().__init__("\n".join(lines))


# --------------------------------------------------------------------------
# Literal formatting
# --------------------------------------------------------------------------


def _num(value: Any) -> str:
    """Format a scalar as a Python literal (``np.inf``/``np.nan`` spelled out)."""
    try:
        val = float(value)
    except (TypeError, ValueError):
        raise CodegenError("Expected a number, got {!r}".format(value))
    if math.isnan(val):
        return "np.nan"
    if math.isinf(val):
        return "np.inf" if val > 0 else "-np.inf"
    return repr(val)


def _array_literal(arr: np.ndarray) -> str:
    """Format an ndarray as ``np.array([...])`` source."""
    arr = np.asarray(arr, dtype=float)

    def _nest(a):
        if a.ndim == 0:
            return _num(a)
        return "[" + ", ".join(_nest(sub) for sub in a) + "]"

    return "np.array({})".format(_nest(arr))


# --------------------------------------------------------------------------
# Name allocation
# --------------------------------------------------------------------------

_RESERVED_NAMES = set(keyword.kwlist) | {
    "np",
    "os",
    "sys",
    "argparse",
    "solve_ivp",
    "plt",
    "t",
    "x",
    "dx",
    "signals",
    "evaluate",
    "rhs",
    "simulate",
    "scope_traces",
    "flatten_traces",
    "write_output",
    "plot_traces",
    "main",
    "SCOPES",
    "X0",
    "T_END",
    "DT",
    "METHOD",
    "RTOL",
    "ATOL",
    "abs",
    "float",
    "int",
    "len",
    "list",
    "max",
    "min",
    "print",
    "range",
    "str",
    "sum",
    "type",
}


def _sanitize(name: str) -> str:
    """Turn an arbitrary block/user name into a candidate Python identifier."""
    ident = re.sub(r"\W", "_", str(name)).strip("_")
    if not ident or ident[0].isdigit():
        ident = "b_" + ident
    return ident


class _NameAllocator:
    """Hands out unique, valid Python identifiers."""

    def __init__(self):
        self._used = set(_RESERVED_NAMES)

    def take(self, preferred: str) -> str:
        base = _sanitize(preferred)
        if base in self._used:
            i = 2
            while "{}_{}".format(base, i) in self._used:
                i += 1
            base = "{}_{}".format(base, i)
        self._used.add(base)
        return base


# --------------------------------------------------------------------------
# Block bookkeeping
# --------------------------------------------------------------------------


class _Block:
    """Everything the emitter needs to know about one diagram block."""

    __slots__ = (
        "obj",
        "name",
        "fn",
        "raw_fn",
        "label",
        "params",
        "var",
        "in_ports",
        "out_ports",
        "state",
        "matrices",
        "port_vars",
    )

    def __init__(self, obj: Any, params: Dict[str, Any]):
        self.obj = obj
        self.name = getattr(obj, "name", "")
        self.raw_fn = getattr(obj, "block_fn", "") or ""
        fn = canonical_fn(self.raw_fn)
        # canonical_fn only spells 'PID' when block_fn is exactly "PID"; the
        # compiler's allowlist also accepts "PiD", so fold that spelling here.
        self.fn = "PID" if fn == "Pid" else fn
        self.label = getattr(obj, "username", "") or self.name
        self.params = params
        self.var = ""
        self.in_ports = int(getattr(obj, "in_ports", 1) or 0)
        self.out_ports = int(getattr(obj, "out_ports", 1) or 0)
        self.state: Optional[Tuple[int, int]] = None  # (start, size)
        self.matrices: Optional[Tuple[np.ndarray, ...]] = None  # (A, B, C, D)
        self.port_vars: Dict[int, str] = {}  # output port -> variable name

    @property
    def title(self) -> str:
        """Human-readable ``Gain 'Kp' (gain2)`` style tag for comments."""
        if self.label and self.label != self.name:
            return "{} '{}' ({})".format(self.raw_fn, self.label, self.name)
        return "{} ({})".format(self.raw_fn, self.name)


def _resolved_params(block: Any) -> Dict[str, Any]:
    """Params with workspace variables resolved, the way the engine reads them.

    Prefers ``exec_params`` when the diagram has been run (that is what
    ``SystemCompiler`` compiles from); otherwise resolves ``params`` through the
    workspace manager so ``gain = "Kp"`` exports as its numeric value.
    """
    exec_params = getattr(block, "exec_params", None)
    if exec_params:
        return dict(exec_params)
    params = dict(getattr(block, "params", None) or {})
    try:
        from lib.workspace import WorkspaceManager

        return dict(WorkspaceManager().resolve_params(params))
    except Exception:  # noqa: BLE001 - workspace is optional for export
        return params


def _declared_sample_time(block: Any) -> float:
    """Sample time declared on a block (>0 means a discrete rate)."""
    resolver = getattr(block, "resolve_sample_time", None)
    if callable(resolver):
        try:
            return float(resolver())
        except (TypeError, ValueError):
            return -1.0
    params = getattr(block, "params", None) or {}
    try:
        return float(params.get("sampling_time", params.get("sample_time", -1.0)))
    except (TypeError, ValueError):
        return -1.0


# --------------------------------------------------------------------------
# Generator
# --------------------------------------------------------------------------


class PythonCodeGenerator:
    """Generate a standalone simulation script from a diagram.

    Args:
        blocks: the diagram's blocks (``DBlock``-like: ``name``, ``block_fn``,
            ``params``, ``in_ports``, ``out_ports``).
        lines: the connections (``DLine``-like: ``srcblock``, ``srcport``,
            ``dstblock``, ``dstport``).
        sim_time: simulation duration baked in as the script's default.
        sim_dt: output/step grid spacing baked in as the script's default.
        solver: solver method (``RK45`` ... or the fixed-step ``Euler``/``RK4``).
        rtol / atol: solve_ivp tolerances (engine defaults 1e-9 / 1e-12).
        diagram_name: name shown in the generated docstring.
    """

    def __init__(
        self,
        blocks: List[Any],
        lines: List[Any],
        sim_time: float = 10.0,
        sim_dt: float = 0.01,
        solver: str = "RK45",
        rtol: float = 1e-9,
        atol: float = 1e-12,
        diagram_name: Optional[str] = None,
    ):
        self.raw_blocks = list(blocks or [])
        self.raw_lines = list(lines or [])
        self.sim_time = float(sim_time)
        self.sim_dt = float(sim_dt)
        solver = str(solver or "RK45")
        # Same fallback the engine applies to an unknown solver name.
        self.solver = solver if solver in _SCIPY_METHODS + _FIXED_STEP_METHODS else "RK45"
        self.rtol = float(rtol)
        self.atol = float(atol)
        self.diagram_name = diagram_name or "untitled diagram"

        self._names = _NameAllocator()
        self._blocks: List[_Block] = []
        self._by_name: Dict[str, _Block] = {}
        self._lines: List[Any] = []
        self._input_map: Dict[str, Dict[int, Tuple[str, int]]] = {}
        # (block order, block title, const name, literal, comment)
        self._consts: List[Tuple[int, str, str, str, str]] = []
        self._block_index: Dict[str, int] = {}
        self._matrix_names: Dict[str, Dict[str, str]] = {}
        self._helpers = set()
        self._n_states = 0
        self._x0: List[float] = []

    # -- public API --------------------------------------------------------

    @classmethod
    def from_dsim(cls, dsim: Any, diagram_name: Optional[str] = None) -> "PythonCodeGenerator":
        """Build a generator from a live ``DSim`` facade (GUI / CLI runner)."""
        return cls(
            getattr(dsim, "blocks_list", []) or [],
            getattr(dsim, "line_list", []) or [],
            sim_time=getattr(dsim, "sim_time", 10.0) or 10.0,
            sim_dt=getattr(dsim, "sim_dt", 0.01) or 0.01,
            solver=getattr(dsim, "solver_method", "RK45") or "RK45",
            rtol=getattr(dsim, "rtol", 1e-9),
            atol=getattr(dsim, "atol", 1e-12),
            diagram_name=diagram_name or getattr(dsim, "filename", None),
        )

    def generate(self) -> str:
        """Return the source text of the standalone script."""
        blocks, lines = self._flatten()
        self._prepare(blocks, lines)
        self._check_supported()
        self._allocate_states()
        order = self._execution_order()
        body = self._emit_evaluate(order)
        return self._assemble(body)

    # -- preparation -------------------------------------------------------

    def _flatten(self):
        """Expand subsystems the way the engine does before compiling."""
        blocks, lines = self.raw_blocks, self.raw_lines
        has_subsystems = any(
            getattr(b, "block_type", "") == "Subsystem" or getattr(b, "block_fn", "") == "Subsystem"
            for b in blocks
        )
        if not has_subsystems:
            return blocks, lines
        try:
            from lib.engine.flattener import Flattener

            return Flattener().flatten(blocks, lines)
        except Exception as exc:  # noqa: BLE001 - report as an unsupported block
            raise CodegenUnsupportedError(
                [
                    (
                        getattr(b, "name", "?"),
                        "Subsystem",
                        "could not be flattened ({})".format(exc),
                    )
                    for b in blocks
                    if getattr(b, "block_fn", "") == "Subsystem"
                ]
            )

    def _prepare(self, blocks, lines):
        for obj in blocks:
            blk = _Block(obj, _resolved_params(obj))
            self._block_index[blk.name] = len(self._blocks)
            self._blocks.append(blk)
            self._by_name[blk.name] = blk

        # Variable names: sinks get one too (kept out of the signals dict) so
        # a Scope named like a signal cannot shadow it.
        for blk in self._blocks:
            blk.var = self._names.take(blk.label or blk.name)
            blk.port_vars[0] = blk.var

        self._input_map = {b.name: {} for b in self._blocks}
        for line in lines:
            dst = getattr(line, "dstblock", None)
            src = getattr(line, "srcblock", None)
            if dst not in self._input_map or src not in self._by_name:
                continue
            self._input_map[dst][int(getattr(line, "dstport", 0) or 0)] = (
                src,
                int(getattr(line, "srcport", 0) or 0),
            )
        self._lines = [
            line
            for line in lines
            if getattr(line, "srcblock", None) in self._by_name
            and getattr(line, "dstblock", None) in self._by_name
        ]

    def _check_supported(self):
        bad: List[Tuple[str, str, str]] = []
        for blk in self._blocks:
            if blk.fn not in SUPPORTED_FNS:
                bad.append((blk.name, blk.raw_fn, "block type is not supported by the exporter"))
                continue
            if blk.fn == "Step" and blk.params.get("type") == "impulse":
                bad.append(
                    (
                        blk.name,
                        blk.raw_fn,
                        "impulse steps run on the interpreter, not the compiled path",
                    )
                )
            elif _declared_sample_time(blk.obj) > 0:
                bad.append(
                    (
                        blk.name,
                        blk.raw_fn,
                        "discrete sample time (sampled-data blocks need the interpreter)",
                    )
                )
        if bad:
            raise CodegenUnsupportedError(bad)

    # -- state allocation (mirrors SystemCompiler.compile_system, section 2) --

    def _allocate_states(self):
        idx = 0
        for blk in self._blocks:
            params = blk.params
            if blk.fn == "Integrator":
                ic = np.atleast_1d(np.array(params.get("init_conds", 0.0), dtype=float)).flatten()
                blk.state = (idx, ic.size)
                self._x0.extend(ic.tolist())
                idx += ic.size
            elif blk.fn == "StateSpace":
                try:
                    A = np.array(params["A"], dtype=float)
                    B = np.atleast_2d(np.array(params["B"], dtype=float))
                    C = np.atleast_2d(np.array(params["C"], dtype=float))
                    D = np.atleast_2d(np.array(params["D"], dtype=float))
                except (KeyError, TypeError, ValueError) as exc:
                    raise CodegenError(
                        "StateSpace block {} has unusable A/B/C/D matrices: {}".format(
                            blk.name, exc
                        )
                    )
                n = A.shape[0] if A.ndim > 1 else 1
                A = A.reshape(n, n)
                if B.shape[0] != n:
                    B = B.reshape(n, -1)
                blk.matrices = (A, B, C, D)
                blk.state = (idx, n)
                self._x0.extend(self._init_conds(params, n))
                idx += n
            elif blk.fn == "TransferFcn":
                num = params.get("numerator", [1.0])
                den = params.get("denominator", [1.0, 1.0])
                try:
                    A, B, C, D = _scipy_signal.tf2ss(num, den)
                except Exception as exc:  # noqa: BLE001 - surface as a codegen error
                    raise CodegenError(
                        "TransferFcn block {} could not be converted to state space: {}".format(
                            blk.name, exc
                        )
                    )
                A = np.atleast_2d(A)
                blk.matrices = (A, np.atleast_2d(B), np.atleast_2d(C), np.atleast_2d(D))
                n = A.shape[0]
                blk.state = (idx, n)
                self._x0.extend(self._init_conds(params, n))
                idx += n
            elif blk.fn == "PID":
                blk.state = (idx, 2)
                self._x0.extend([0.0, 0.0])
                idx += 2
        self._n_states = idx

    @staticmethod
    def _init_conds(params: Dict[str, Any], n: int) -> List[float]:
        """Initial state padded/truncated to ``n`` (as compile_system does)."""
        ic = np.atleast_1d(np.array(params.get("init_conds", [0.0] * n), dtype=float)).flatten()
        if ic.size < n:
            padded = np.zeros(n)
            padded[: ic.size] = ic
            ic = padded
        elif ic.size > n:
            ic = ic[:n]
        return ic.tolist()

    def _is_d0_state(self, blk: _Block) -> bool:
        """True for strictly-proper state blocks (output = C x, no D u)."""
        if blk.fn not in STATE_FNS:
            return False
        if blk.matrices is not None:
            return not np.any(blk.matrices[3] != 0)
        return blk.fn == "Integrator"  # PID always feeds through

    # -- ordering ----------------------------------------------------------

    def _execution_order(self) -> List[_Block]:
        """sources -> algebraic/feedthrough -> strictly-proper state blocks."""
        adjacency: Dict[str, List[str]] = {b.name: [] for b in self._blocks}
        for line in self._lines:
            dst = self._by_name[line.dstblock]
            if self._is_d0_state(dst):
                continue  # not a feedthrough edge: output is known from x
            adjacency[line.srcblock].append(dst.name)

        order_names, cyclic = kahn_topological_order(
            (b.name for b in self._blocks), adjacency, key=lambda n: n
        )
        if cyclic:
            raise CodegenError(
                "Algebraic loop detected through: {}. Break it with an Integrator or a "
                "strictly-proper transfer function before exporting.".format(", ".join(cyclic))
            )

        ordered = [self._by_name[n] for n in order_names]
        sources = [b for b in ordered if b.fn in SOURCE_FNS]
        d0_state = [b for b in ordered if self._is_d0_state(b)]
        d0_names = {b.name for b in d0_state}
        middle = [b for b in ordered if b.fn not in SOURCE_FNS and b.name not in d0_names]
        return sources + middle + d0_state

    # -- input lookup ------------------------------------------------------

    def _input_expr(self, blk: _Block, port: int) -> Optional[str]:
        """Variable holding the signal wired into ``port`` (None = unconnected)."""
        dep = self._input_map.get(blk.name, {}).get(port)
        if dep is None:
            return None
        src_name, src_port = dep
        src = self._by_name[src_name]
        return src.port_vars.get(src_port, src.var)

    def _input_or_zero(self, blk: _Block, port: int) -> str:
        return self._input_expr(blk, port) or "0.0"

    def _n_wired_inputs(self, blk: _Block) -> int:
        deps = self._input_map.get(blk.name, {})
        return (max(deps) + 1) if deps else 0

    # -- constants ---------------------------------------------------------

    def _const(self, blk: _Block, param: str, literal: str, comment: str = "") -> str:
        name = self._names.take("{}_{}".format(blk.var, param).upper())
        self._consts.append((self._block_index.get(blk.name, 0), blk.title, name, literal, comment))
        return name

    def _num_const(self, blk: _Block, param: str, value: Any, comment: str = "") -> str:
        return self._const(blk, param, _num(value), comment)

    # -- emission ----------------------------------------------------------

    def _emit_evaluate(self, order: List[_Block]) -> List[str]:
        """Body of the generated ``evaluate(t, x)`` (returns source lines)."""
        out: List[str] = []
        emitted: List[_Block] = []

        # 1. Strictly-proper state outputs, straight from the state vector.
        d0 = [b for b in order if self._is_d0_state(b)]
        out.append("    # --- Strictly-proper state outputs (D = 0), read from x ---")
        if not d0:
            out.append("    # (none)")
        for blk in d0:
            out.append("    # {}".format(blk.title))
            out.extend(self._emit_state_output(blk))
            emitted.append(blk)

        # 2. Sources.
        out.append("")
        out.append("    # --- Sources ---")
        sources = [b for b in order if b.fn in SOURCE_FNS]
        if not sources:
            out.append("    # (none)")
        for blk in sources:
            out.extend(self._emit_block(blk))
            emitted.append(blk)

        # 3. Algebraic blocks and feedthrough state blocks, in topological order.
        out.append("")
        out.append("    # --- Algebraic blocks and feedthrough (D != 0) state blocks ---")
        middle = [b for b in order if b.fn not in SOURCE_FNS and not self._is_d0_state(b)]
        emitted_any = False
        for blk in middle:
            chunk = self._emit_block(blk)
            if chunk:
                out.extend(chunk)
                emitted_any = True
            if blk.fn not in SINK_FNS:
                emitted.append(blk)
        if not emitted_any:
            out.append("    # (none)")

        # 4. Derivatives of the strictly-proper state blocks.
        out.append("")
        out.append("    # --- State derivatives (strictly-proper blocks) ---")
        if not d0:
            out.append("    # (none)")
        for blk in d0:
            out.append("    # {}".format(blk.title))
            out.extend(self._emit_state_derivative(blk))

        # 5. Signal dict (block outputs at this instant).
        out.append("")
        out.append("    signals = {")
        for blk in emitted:
            for port in sorted(blk.port_vars):
                var = blk.port_vars[port]
                out.append('        "{}": {},'.format(var, var))
        out.append("    }")
        out.append("    return dx, signals")
        return out

    # -- per-block emitters -------------------------------------------------

    def _emit_block(self, blk: _Block) -> List[str]:
        fn = blk.fn
        if fn in SINK_FNS:
            return []
        emitter = getattr(self, "_emit_" + fn.lower(), None)
        if emitter is None:
            raise CodegenError("No emitter for supported block {}".format(blk.raw_fn))
        lines = emitter(blk)
        return ["    # {}".format(blk.title)] + lines

    # sources ---------------------------------------------------------------

    def _emit_constant(self, blk: _Block) -> List[str]:
        raw = blk.params.get("value", 0.0)
        if isinstance(raw, (list, tuple, np.ndarray)):
            literal = _array_literal(np.atleast_1d(np.asarray(raw, dtype=float)))
        else:
            literal = _num(raw)
        const = self._const(blk, "value", literal)
        return ["    {} = {}".format(blk.var, const)]

    def _emit_step(self, blk: _Block) -> List[str]:
        # Mirrors compiler_kernels.sources.build_step: only 'impulse' is special
        # (and rejected above); every other type is a rising step at `delay`.
        delay = self._num_const(blk, "delay", blk.params.get("delay", 0.0))
        value = self._num_const(blk, "value", blk.params.get("value", 1.0))
        return ["    {} = {} if t >= {} else 0.0".format(blk.var, value, delay)]

    def _emit_sine(self, blk: _Block) -> List[str]:
        params = blk.params
        amp = self._num_const(blk, "amplitude", params.get("amplitude", 1.0))
        freq = self._num_const(
            blk, "frequency", params.get("frequency", params.get("omega", 1.0)), "rad/s"
        )
        phase = self._num_const(blk, "phase", params.get("phase", params.get("init_angle", 0.0)))
        bias = self._num_const(blk, "bias", params.get("bias", 0.0))
        return ["    {} = {} * np.sin({} * t + {}) + {}".format(blk.var, amp, freq, phase, bias)]

    def _emit_ramp(self, blk: _Block) -> List[str]:
        slope_val = float(blk.params.get("slope", 1.0))
        slope = self._num_const(blk, "slope", slope_val)
        delay = self._num_const(blk, "delay", blk.params.get("delay", 0.0))
        if slope_val > 0:
            return ["    {} = max(0.0, {} * (t - {}))".format(blk.var, slope, delay)]
        if slope_val < 0:
            return ["    {} = min(0.0, {} * (t - {}))".format(blk.var, slope, delay)]
        return ["    {} = 0.0".format(blk.var)]

    # algebraic -------------------------------------------------------------

    def _emit_gain(self, blk: _Block) -> List[str]:
        gain = self._num_const(blk, "gain", blk.params.get("gain", 1.0))
        return ["    {} = {} * {}".format(blk.var, self._input_or_zero(blk, 0), gain)]

    def _emit_sum(self, blk: _Block) -> List[str]:
        signs = blk.params.get("sign", blk.params.get("inputs", "++"))
        n_terms = max(len(signs), self._n_wired_inputs(blk))
        terms = []
        for i in range(n_terms):
            sign = signs[i] if i < len(signs) else "+"
            src = self._input_expr(blk, i)
            if src is None:
                continue  # unconnected port contributes 0.0
            terms.append(("-" if sign == "-" else "+", src))
        if not terms:
            return ["    {} = 0.0".format(blk.var)]
        first_sign, first_src = terms[0]
        expr = ("-" + first_src) if first_sign == "-" else first_src
        for sign, src in terms[1:]:
            expr += " {} {}".format(sign, src)
        return ["    {} = {}".format(blk.var, expr)]

    def _emit_product(self, blk: _Block) -> List[str]:
        ops = blk.params.get("ops", "**")
        n_terms = max(len(ops), self._n_wired_inputs(blk))
        expr = "1.0"
        for i in range(n_terms):
            op = ops[i] if i < len(ops) else "*"
            src = self._input_or_zero(blk, i)
            if op == "/":
                self._helpers.add("_safe_div")
                expr = "_safe_div({}, {})".format(expr, src)
            else:
                expr = "{} * {}".format(expr, src)
        if expr.startswith("1.0 * "):
            expr = expr[len("1.0 * ") :]
        return ["    {} = {}".format(blk.var, expr)]

    def _emit_sgprod(self, blk: _Block) -> List[str]:
        n_terms = self._n_wired_inputs(blk)
        if not n_terms:
            return ["    {} = 1.0".format(blk.var)]
        factors = [self._input_or_zero(blk, i) for i in range(n_terms)]
        return ["    {} = {}".format(blk.var, " * ".join(factors))]

    _emit_sigproduct = _emit_sgprod

    def _emit_abs(self, blk: _Block) -> List[str]:
        return ["    {} = abs({})".format(blk.var, self._input_or_zero(blk, 0))]

    _emit_absblock = _emit_abs

    def _emit_saturation(self, blk: _Block) -> List[str]:
        lower = self._num_const(blk, "min", blk.params.get("min", -np.inf))
        upper = self._num_const(blk, "max", blk.params.get("max", np.inf))
        return [
            "    {} = np.clip({}, {}, {})".format(
                blk.var, self._input_or_zero(blk, 0), lower, upper
            )
        ]

    # routing ---------------------------------------------------------------

    def _emit_mux(self, blk: _Block) -> List[str]:
        n_terms = self._n_wired_inputs(blk) or blk.in_ports
        parts = [self._input_or_zero(blk, i) for i in range(n_terms)]
        return ["    {} = np.array([{}])".format(blk.var, ", ".join(parts) if parts else "0.0")]

    def _emit_demux(self, blk: _Block) -> List[str]:
        width = int(blk.params.get("output_shape", 1) or 1)
        width = max(width, 1)
        n_out = int(blk.params.get("_outputs_", blk.out_ports) or 1)
        n_out = max(n_out, 1)
        tmp = "_{}_in".format(blk.var)
        lines = [
            "    {} = np.atleast_1d(np.asarray({}, dtype=float)).ravel()".format(
                tmp, self._input_or_zero(blk, 0)
            )
        ]
        for port in range(n_out):
            if port not in blk.port_vars:
                blk.port_vars[port] = self._names.take("{}_out{}".format(blk.var, port))
            lines.append(
                "    {} = {}[{}:{}]".format(
                    blk.port_vars[port], tmp, port * width, (port + 1) * width
                )
            )
        return lines

    # state blocks -----------------------------------------------------------

    def _slice(self, blk: _Block) -> str:
        start, size = blk.state
        return "x[{}:{}]".format(start, start + size)

    def _emit_state_output(self, blk: _Block) -> List[str]:
        """Output of a strictly-proper state block, pre-populated from x."""
        start, size = blk.state
        if blk.fn == "Integrator":
            if size == 1:
                return ["    {} = x[{}]".format(blk.var, start)]
            return ["    {} = {}".format(blk.var, self._slice(blk))]
        C = self._matrix_const(blk, "C")
        expr = "{} @ {}.reshape(-1, 1)".format(C, self._slice(blk))
        if blk.matrices[2].shape[0] == 1:
            return ["    {} = ({}).item()".format(blk.var, expr)]
        return ["    {} = ({}).ravel()".format(blk.var, expr)]

    def _emit_state_derivative(self, blk: _Block) -> List[str]:
        start, size = blk.state
        if blk.fn == "Integrator":
            src = self._input_or_zero(blk, 0)
            if size == 1:
                self._helpers.add("_scalar")
                return ["    dx[{}] = _scalar({})".format(start, src)]
            return ["    dx[{}:{}] = np.atleast_1d({}).ravel()".format(start, start + size, src)]
        A = self._matrix_const(blk, "A")
        B = self._matrix_const(blk, "B")
        u_var = "_{}_u".format(blk.var)
        lines = self._emit_input_vector(blk, u_var)
        lines.append(
            "    dx[{}:{}] = ({} @ {}.reshape(-1, 1) + {} @ {}).ravel()".format(
                start, start + size, A, self._slice(blk), B, u_var
            )
        )
        return lines

    def _emit_statespace(self, blk: _Block) -> List[str]:
        """Feedthrough (D != 0) state block: output and derivative together."""
        start, size = blk.state
        A = self._matrix_const(blk, "A")
        B = self._matrix_const(blk, "B")
        C = self._matrix_const(blk, "C")
        D = self._matrix_const(blk, "D")
        u_var = "_{}_u".format(blk.var)
        x_var = "_{}_x".format(blk.var)
        lines = self._emit_input_vector(blk, u_var)
        lines.append("    {} = {}.reshape(-1, 1)".format(x_var, self._slice(blk)))
        out_expr = "{} @ {} + {} @ {}".format(C, x_var, D, u_var)
        if blk.matrices[2].shape[0] == 1:
            lines.append("    {} = ({}).item()".format(blk.var, out_expr))
        else:
            lines.append("    {} = ({}).ravel()".format(blk.var, out_expr))
        lines.append(
            "    dx[{}:{}] = ({} @ {} + {} @ {}).ravel()".format(
                start, start + size, A, x_var, B, u_var
            )
        )
        return lines

    _emit_transferfcn = _emit_statespace

    def _emit_input_vector(self, blk: _Block, u_var: str) -> List[str]:
        """Assemble the ``u`` column vector for a state-space style block."""
        B = blk.matrices[1]
        n_inputs = B.shape[1]
        if n_inputs == 1:
            return [
                "    {} = np.atleast_1d({}).reshape(-1, 1)".format(
                    u_var, self._input_or_zero(blk, 0)
                )
            ]
        self._helpers.add("_pack_inputs")
        n_ports = max(self._n_wired_inputs(blk), blk.in_ports, n_inputs)
        parts = [self._input_expr(blk, i) or "None" for i in range(n_ports)]
        return ["    {} = _pack_inputs([{}], {})".format(u_var, ", ".join(parts), n_inputs)]

    def _emit_pid(self, blk: _Block) -> List[str]:
        start, _ = blk.state
        params = blk.params
        kp = self._num_const(blk, "kp", params.get("Kp", 1.0))
        ki = self._num_const(blk, "ki", params.get("Ki", 0.0))
        kd = self._num_const(blk, "kd", params.get("Kd", 0.0))
        n_filt = self._num_const(blk, "n", params.get("N", 20.0), "derivative filter pole")
        u_min = self._num_const(blk, "umin", params.get("u_min", -np.inf))
        u_max = self._num_const(blk, "umax", params.get("u_max", np.inf))

        self._helpers.add("_scalar")
        sp = self._input_expr(blk, 0)
        meas = self._input_expr(blk, 1)
        v = blk.var
        lines = [
            "    _{}_e = {} - {}".format(
                v,
                "float(_scalar({}))".format(sp) if sp else "0.0",
                "float(_scalar({}))".format(meas) if meas else "0.0",
            ),
            "    _{}_xi = x[{}]".format(v, start),
            "    _{}_xd = x[{}]".format(v, start + 1),
            "    _{v}_dxi = _{v}_e".format(v=v),
            "    _{v}_dxd = {n} * (_{v}_e - _{v}_xd)".format(v=v, n=n_filt),
            "    _{v}_u = {kp} * _{v}_e + {ki} * _{v}_xi + {kd} * _{v}_dxd".format(
                v=v, kp=kp, ki=ki, kd=kd
            ),
            "    {} = np.clip(_{}_u, {}, {})".format(v, v, u_min, u_max),
            "    # anti-windup: freeze the integrator while the output is saturated",
            "    if (_{v}_u > {hi} and _{v}_e > 0) or (_{v}_u < {lo} and _{v}_e < 0):".format(
                v=v, hi=u_max, lo=u_min
            ),
            "        _{}_dxi = 0.0".format(v),
            "    dx[{}] = _{}_dxi".format(start, v),
            "    dx[{}] = _{}_dxd".format(start + 1, v),
        ]
        return lines

    def _matrix_const(self, blk: _Block, which: str) -> str:
        """Emit (once) the A/B/C/D constant for a state block and return its name."""
        cache = self._matrix_names.setdefault(blk.name, {})
        if which in cache:
            return cache[which]
        idx = "ABCD".index(which)
        mat = np.atleast_2d(np.asarray(blk.matrices[idx], dtype=float))
        name = self._names.take("{}_{}".format(which, blk.var))
        self._consts.append(
            (
                self._block_index.get(blk.name, 0),
                blk.title,
                name,
                _array_literal(mat),
                "{} matrix".format(which),
            )
        )
        cache[which] = name
        return name

    # -- scopes --------------------------------------------------------------

    def _scope_specs(self) -> List[Dict[str, Any]]:
        specs = []
        for blk in self._blocks:
            if blk.fn != "Scope":
                continue
            n_inputs = blk.in_ports or 1
            sources = [self._input_expr(blk, port) for port in range(n_inputs)]
            raw = blk.params.get("labels", "")
            labels: List[str] = []
            if isinstance(raw, str) and raw and raw != "default":
                # The engine strips *all* spaces before splitting; mirror it so
                # the exported column names match a headless run exactly.
                labels = [lab.strip() for lab in raw.replace(" ", "").split(",") if lab.strip()]
            specs.append(
                {
                    "name": blk.name,
                    "title": str(blk.params.get("title", "") or blk.label or blk.name),
                    "sources": sources,
                    "labels": labels,
                }
            )
        return specs

    # -- assembly ------------------------------------------------------------

    def _assemble(self, evaluate_body: List[str]) -> str:
        out: List[str] = []
        add = out.append

        n_blocks = len(self._blocks)
        n_lines = len(self._lines)
        diagram = os.path.basename(str(self.diagram_name))
        script_name = (os.path.splitext(diagram)[0] or "model") + ".py"

        add('"""Standalone simulation exported from DiaBloS Modern.')
        add("")
        add("Source diagram: {} ({} blocks, {} connections).".format(diagram, n_blocks, n_lines))
        add("")
        add("The diagram is compiled to a single ODE and integrated with")
        add("scipy.integrate.solve_ivp, matching DiaBloS' compiled (fast) solver.")
        add("Requires numpy and scipy; matplotlib is only needed for the plots.")
        add("")
        add("Usage:")
        add("    python {}                # simulate and plot".format(script_name))
        add(
            "    python {} --no-plot      # headless (also honours DIABLOS_NO_PLOT=1)".format(
                script_name
            )
        )
        add(
            "    python {} --out run.csv  # write the Scope traces to CSV or NPZ".format(
                script_name
            )
        )
        add('"""')
        add("")
        add("import argparse")
        add("import os")
        add("import sys")
        add("")
        add("import numpy as np")
        add("from scipy.integrate import solve_ivp")
        add("")
        add("")
        add("# ==========================================================================")
        add("# Simulation settings")
        add("# ==========================================================================")
        add("")
        add("T_END = {}      # simulation duration [s]".format(_num(self.sim_time)))
        add("DT = {}         # output/step grid spacing [s]".format(_num(self.sim_dt)))
        add('METHOD = "{}"'.format(self.solver))
        add("RTOL = {}".format(_num(self.rtol)))
        add("ATOL = {}".format(_num(self.atol)))
        add("")
        add("")
        add("# ==========================================================================")
        add("# Block parameters")
        add("# ==========================================================================")
        add("")
        if not self._consts:
            add("# (this diagram has no numeric parameters)")
        last_title = None
        for _, title, name, literal, comment in sorted(self._consts, key=lambda c: c[0]):
            if title != last_title:
                if last_title is not None:
                    add("")
                add("# {}".format(title))
                last_title = title
            add("{} = {}{}".format(name, literal, "  # " + comment if comment else ""))
        add("")
        add("")
        add("# ==========================================================================")
        add("# Initial state")
        add("# ==========================================================================")
        add("#")
        add("# State vector layout:")
        if self._n_states:
            for blk in self._blocks:
                if blk.state is None:
                    continue
                start, size = blk.state
                add("#   x[{}:{}]  {}".format(start, start + size, blk.title))
        else:
            add("#   (no continuous states -- the diagram is purely algebraic)")
        add("")
        add("X0 = np.array([{}])".format(", ".join(_num(v) for v in self._x0)))
        add("")
        add("")

        helpers = self._helper_source()
        if helpers:
            add("# ==========================================================================")
            add("# Helpers (transcribed from the DiaBloS compiled kernels)")
            add("# ==========================================================================")
            add("")
            out.extend(helpers)
            add("")

        add("# ==========================================================================")
        add("# Diagram")
        add("# ==========================================================================")
        add("")
        add("")
        add("def evaluate(t, x):")
        add('    """Evaluate every block at time ``t`` and state ``x``.')
        add("")
        add("    Returns ``(dx, signals)``: the state derivative and each block's")
        add("    output signal. The evaluation order mirrors the DiaBloS compiled")
        add("    solver -- strictly-proper state outputs first (so feedback loops")
        add("    resolve), then sources, then the algebraic blocks, then the")
        add("    remaining state derivatives.")
        add('    """')
        add("    dx = np.zeros_like(x)")
        add("")
        out.extend(evaluate_body)
        add("")
        add("")
        add("def rhs(t, x):")
        add('    """State derivative for scipy.integrate.solve_ivp."""')
        add("    return evaluate(t, x)[0]")
        add("")
        add("")
        add("# ==========================================================================")
        add("# Simulation")
        add("# ==========================================================================")
        add("")
        add("")
        out.extend(self._simulate_source())
        add("")
        add("")
        out.extend(self._scope_source())
        add("")
        add("")
        out.extend(self._io_source())
        add("")
        add("")
        out.extend(self._main_source())
        return "\n".join(out) + "\n"

    def _helper_source(self) -> List[str]:
        src: List[str] = []
        if "_scalar" in self._helpers:
            src += [
                "def _scalar(value):",
                '    """First element of a possibly-vector signal."""',
                "    return np.ravel(value)[0] if np.ndim(value) else value",
                "",
                "",
            ]
        if "_safe_div" in self._helpers:
            src += [
                "def _safe_div(num, den):",
                '    """Element-wise divide with the solver\'s guards (inf -> +-1e308, nan -> 0)."""',
                "    num = np.asarray(num, dtype=float)",
                "    den = np.asarray(den, dtype=float)",
                '    with np.errstate(divide="ignore", invalid="ignore"):',
                "        res = num / den",
                "        res = np.where(np.isinf(res), np.sign(res) * 1e308, res)",
                "        res = np.where(np.isnan(res), 0.0, res)",
                "    return res",
                "",
                "",
            ]
        if "_pack_inputs" in self._helpers:
            src += [
                "def _pack_inputs(values, n_inputs):",
                '    """Pack several ports into one column input vector (vectors unpacked)."""',
                "    u = np.zeros((n_inputs, 1))",
                "    idx = 0",
                "    for val in values:",
                "        if val is None:",
                "            idx += 1",
                "            continue",
                "        flat = np.atleast_1d(val).ravel()",
                "        for j in range(len(flat)):",
                "            if idx < n_inputs:",
                "                u[idx, 0] = flat[j]",
                "                idx += 1",
                "    return u",
                "",
                "",
            ]
        return src

    def _simulate_source(self) -> List[str]:
        src = [
            "def simulate(t_end=T_END, dt=DT):",
            '    """Integrate the diagram; returns (t, x_hist) with x_hist shape (n_states, n)."""',
            "    t_grid = np.arange(0.0, t_end + dt, dt)",
            "    t_grid = t_grid[t_grid <= t_end + 1e-12]",
            "    t_grid[-1] = min(t_grid[-1], t_end)",
            "    if X0.size == 0:",
            "        return t_grid, np.zeros((0, len(t_grid)))",
        ]
        if self.solver in _FIXED_STEP_METHODS:
            scheme = "rk4" if self.solver == "RK4" else "euler"
            src += [
                "    # Fixed-step integration on the output grid (METHOD = {}).".format(
                    self.solver
                ),
                "    x_hist = np.zeros((X0.size, len(t_grid)))",
                "    xk = X0.copy()",
                "    x_hist[:, 0] = xk",
                "    for k in range(1, len(t_grid)):",
                "        tk = t_grid[k - 1]",
                "        h = t_grid[k] - tk",
            ]
            if scheme == "rk4":
                src += [
                    "        k1 = np.asarray(rhs(tk, xk), dtype=float)",
                    "        k2 = np.asarray(rhs(tk + h / 2.0, xk + h / 2.0 * k1), dtype=float)",
                    "        k3 = np.asarray(rhs(tk + h / 2.0, xk + h / 2.0 * k2), dtype=float)",
                    "        k4 = np.asarray(rhs(tk + h, xk + h * k3), dtype=float)",
                    "        xk = xk + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)",
                ]
            else:
                src += [
                    "        xk = xk + h * np.asarray(rhs(tk, xk), dtype=float)",
                ]
            src += [
                "        x_hist[:, k] = xk",
                "    return t_grid, x_hist",
            ]
        else:
            src += [
                "    sol = solve_ivp(",
                "        rhs, (0.0, t_end), X0, t_eval=t_grid, method=METHOD, rtol=RTOL, atol=ATOL",
                "    )",
                "    if not sol.success:",
                '        raise RuntimeError("solve_ivp failed: " + str(sol.message))',
                "    return sol.t, sol.y",
            ]
        return src

    def _scope_source(self) -> List[str]:
        specs = self._scope_specs()
        src = [
            "# Scope blocks: which signals each one records, in port order.",
            "SCOPES = [",
        ]
        for spec in specs:
            sources = ", ".join(('"{}"'.format(s) if s else "None") for s in spec["sources"]) or ""
            labels = ", ".join('"{}"'.format(lab) for lab in spec["labels"])
            src += [
                "    {",
                '        "name": "{}",'.format(spec["name"]),
                '        "title": "{}",'.format(spec["title"].replace('"', "'")),
                '        "sources": [{}],'.format(sources),
                '        "labels": [{}],'.format(labels),
                "    },",
            ]
        src.append("]")
        src += [
            "",
            "",
            "def scope_traces(t_grid, x_hist):",
            '    """Rebuild each Scope\'s channels by re-evaluating the diagram.',
            "",
            "    Returns ``[(title, {label: trace}), ...]`` in diagram order. Channel",
            "    labels follow the Scope's ``labels`` parameter, padded with",
            '    ``"<scope>-<index>"`` exactly as DiaBloS names them.',
            '    """',
            "    buffers = [[] for _ in SCOPES]",
            "    for i, ti in enumerate(t_grid):",
            "        _, signals = evaluate(ti, x_hist[:, i])",
            "        for spec, buf in zip(SCOPES, buffers):",
            "            parts = [",
            "                np.atleast_1d(np.asarray(signals[key] if key else 0.0, dtype=float)).ravel()",
            '                for key in spec["sources"]',
            "            ]",
            "            buf.append(np.concatenate(parts) if parts else np.zeros(1))",
            "    results = []",
            "    for spec, buf in zip(SCOPES, buffers):",
            "        data = np.array(buf) if buf else np.zeros((len(t_grid), 1))",
            '        labels = list(spec["labels"])',
            "        while len(labels) < data.shape[1]:",
            '            labels.append("{}-{}".format(spec["name"], len(labels)))',
            "        labels = labels[: data.shape[1]]",
            '        results.append((spec["title"], {lab: data[:, j] for j, lab in enumerate(labels)}))',
            "    return results",
            "",
            "",
            "def flatten_traces(results):",
            '    """Flatten per-scope traces into one ordered {label: trace} mapping."""',
            "    flat = {}",
            "    seen = {}",
            "    for _, traces in results:",
            "        for label, values in traces.items():",
            "            name = label",
            "            if name in seen:",
            "                seen[name] += 1",
            '                name = "{}#{}".format(label, seen[label])',
            "            else:",
            "                seen[name] = 0",
            "            flat[name] = values",
            "    return flat",
        ]
        return src

    def _io_source(self) -> List[str]:
        return [
            "def write_output(t_grid, results, path):",
            '    """Write the Scope traces to ``path`` (.npz or .csv)."""',
            "    flat = flatten_traces(results)",
            '    if path.lower().endswith(".npz"):',
            "        np.savez(path, t=t_grid, **flat)",
            "    else:",
            "        columns = [t_grid] + [flat[k] for k in flat]",
            '        header = ",".join(["t"] + list(flat))',
            "        np.savetxt(",
            '            path, np.column_stack(columns), delimiter=",", header=header, comments=""',
            "        )",
            "    return len(flat)",
            "",
            "",
            "def plot_traces(t_grid, results):",
            '    """Plot one figure per Scope block."""',
            "    import matplotlib.pyplot as plt",
            "",
            "    for title, traces in results:",
            "        fig, ax = plt.subplots()",
            "        for label, values in traces.items():",
            "            ax.plot(t_grid, values, label=label)",
            "        ax.set_title(title)",
            '        ax.set_xlabel("time [s]")',
            "        ax.grid(True, alpha=0.3)",
            "        if traces:",
            "            ax.legend()",
            "        fig.tight_layout()",
            "    plt.show()",
        ]

    def _main_source(self) -> List[str]:
        return [
            "def main(argv=None):",
            "    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])",
            '    parser.add_argument("--time", type=float, default=T_END,',
            '                        help="simulation duration in seconds")',
            '    parser.add_argument("--dt", type=float, default=DT,',
            '                        help="output grid spacing in seconds")',
            '    parser.add_argument("--out", default=None,',
            '                        help="write the Scope traces to this .csv or .npz file")',
            '    parser.add_argument("--no-plot", action="store_true",',
            '                        help="skip the plots (also set by DIABLOS_NO_PLOT=1)")',
            "    args = parser.parse_args(argv)",
            "",
            "    t_grid, x_hist = simulate(args.time, args.dt)",
            "    results = scope_traces(t_grid, x_hist)",
            "",
            "    if args.out:",
            "        n = write_output(t_grid, results, args.out)",
            '        print("wrote {} signal(s) to {}".format(n, args.out))',
            "",
            '    no_plot = args.no_plot or os.environ.get("DIABLOS_NO_PLOT") == "1"',
            "    if not no_plot and results:",
            "        plot_traces(t_grid, results)",
            "    return 0",
            "",
            "",
            'if __name__ == "__main__":',
            "    sys.exit(main())",
        ]


def generate_python_script(blocks, lines, **kwargs) -> str:
    """Convenience wrapper: build a :class:`PythonCodeGenerator` and run it."""
    return PythonCodeGenerator(blocks, lines, **kwargs).generate()
