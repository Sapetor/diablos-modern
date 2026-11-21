"""
Symbolic Engine for Equation Extraction and Analysis

Provides symbolic computation capabilities for block diagrams:
- Automatic equation extraction from block diagrams
- Transfer function computation between signals
- Linearization at operating points
- LaTeX/MathML equation export

Uses SymPy for symbolic mathematics.
"""

from __future__ import annotations

import importlib.util
import logging
from typing import Dict, List, Any, Tuple, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from sympy import Symbol

logger = logging.getLogger(__name__)

# SymPy is optional and slow to import (~0.2 s), so it is only checked for here
# and imported when a SymbolicEngine is actually built.
SYMPY_AVAILABLE = importlib.util.find_spec("sympy") is not None
if not SYMPY_AVAILABLE:
    logger.warning("SymPy not installed. Symbolic features disabled.")


class SymbolicEngine:
    """
    Engine for symbolic analysis of block diagrams.

    Traces signal flow through the diagram symbolically,
    composing expressions through each block.
    """

    def __init__(self, dsim=None):
        """
        Initialize the symbolic engine.

        Args:
            dsim: Reference to DSim instance
        """
        if not SYMPY_AVAILABLE:
            raise ImportError(
                "SymPy is required for symbolic features. Install with: pip install sympy"
            )
        import sympy

        self.dsim = dsim
        self.blocks = []
        self.lines = []
        self.block_map = {}  # name -> block
        self.input_map = {}  # block_name -> {port: (src_block, src_port)}
        self.output_map = {}  # block_name -> {port: [(dst_block, dst_port), ...]}

        # Symbolic state
        self.symbolic_outputs = {}  # block_name -> {port: sympy_expr}
        self.state_variables = {}  # block_name -> [sympy_symbols]
        self.s = sympy.Symbol("s")  # Laplace variable

    def build_graph(self, blocks: List = None, lines: List = None):
        """
        Build the block diagram graph for symbolic analysis.

        Args:
            blocks: List of blocks (uses dsim if not provided)
            lines: List of connections (uses dsim if not provided)
        """
        if blocks is None and self.dsim is not None:
            blocks = self.dsim.blocks_list
        if lines is None and self.dsim is not None:
            lines = self.dsim.line_list

        self.blocks = blocks
        self.lines = lines

        # Build block map
        self.block_map = {b.name: b for b in blocks}

        # Build connection maps
        self.input_map = {b.name: {} for b in blocks}
        self.output_map = {b.name: {} for b in blocks}

        for line in lines:
            if hasattr(line, "hidden") and line.hidden:
                continue

            src = line.srcblock
            src_port = line.srcport
            dst = line.dstblock
            dst_port = line.dstport

            self.input_map[dst][dst_port] = (src, src_port)

            if src_port not in self.output_map[src]:
                self.output_map[src][src_port] = []
            self.output_map[src][src_port].append((dst, dst_port))

    def create_input_symbols(self, input_blocks: List[str] = None) -> Dict[str, Symbol]:
        """
        Create symbolic inputs for specified blocks.

        Args:
            input_blocks: List of block names to treat as inputs.
                         If None, automatically detect source blocks.

        Returns:
            Dict mapping block names to input symbols
        """
        import sympy

        input_symbols = {}

        if input_blocks is None:
            # Find source blocks (no inputs)
            input_blocks = []
            for block in self.blocks:
                block_type = getattr(block, "block_fn", "")
                if block_type in ("Step", "Sine", "Constant", "Ramp", "Inport"):
                    input_blocks.append(block.name)

        for name in input_blocks:
            # Create symbol with nice name
            sym_name = name.replace(" ", "_")
            input_symbols[name] = sympy.Symbol(sym_name)

        return input_symbols

    def _resolve_outputs(
        self, input_symbols: Dict[str, Symbol], zero_sources: bool = False
    ) -> Dict[str, Dict[int, Any]]:
        """Output expressions of every block, with feedback loops solved.

        Every wire gets an unknown ``<block>_<port>`` and each block is evaluated
        on the unknowns of its input wires. Chains are then substituted forward
        and whatever still refers to itself (a loop) is solved as one system, so
        a loop yields its closed-loop expression instead of a dangling symbol.
        Blocks in ``input_symbols`` output their symbol. With ``zero_sources``,
        every other block without input wires outputs 0 (superposition, as a
        transfer function needs). A loop SymPy cannot solve (nonlinear blocks)
        keeps its unknowns, with a warning.

        Returns {block_name: {port: expr}}. Unsupported blocks are left out, so
        their outputs stay as unknowns downstream.
        """
        import sympy

        unknowns = {}  # (block, port) -> Symbol

        def is_zeroed(name):
            return zero_sources and name not in input_symbols and not self.input_map.get(name)

        def signal(src, port):
            if src in input_symbols:
                return input_symbols[src]
            if is_zeroed(src):
                return sympy.Integer(0)
            if (src, port) not in unknowns:
                unknowns[(src, port)] = sympy.Symbol(f"{src}_{port}".replace(" ", "_"))
            return unknowns[(src, port)]

        raw = {}
        for block in self.blocks:
            name = block.name
            if name in input_symbols:
                raw[name] = {0: input_symbols[name]}
            elif is_zeroed(name):
                raw[name] = {p: sympy.Integer(0) for p in self.output_map.get(name) or [0]}
            else:
                inputs = {
                    port: signal(src, src_port)
                    for port, (src, src_port) in self.input_map.get(name, {}).items()
                }
                outputs = self._compute_block_symbolic(block, inputs)
                if outputs:
                    raw[name] = outputs

        pending = {sym: raw[b][p] for (b, p), sym in unknowns.items() if b in raw and p in raw[b]}
        resolved = {}
        progress = True
        while progress and pending:
            progress = False
            for sym, expr in list(pending.items()):
                expr = expr.xreplace(resolved)
                pending[sym] = expr
                if not expr.free_symbols & pending.keys():
                    resolved[sym] = expr
                    del pending[sym]
                    progress = True
        if pending:  # loops, plus anything downstream of them
            try:
                found = sympy.solve(
                    [sympy.Eq(sym, expr) for sym, expr in pending.items()],
                    list(pending),
                    dict=True,
                )
            except Exception as exc:  # noqa: BLE001 -- e.g. matrix gains in a loop
                logger.debug("Solving the loop equations failed: %s", exc)
                found = []
            if len(found) == 1:
                resolved.update(found[0])
            else:
                logger.warning(
                    "Could not solve the feedback loop(s) through %s symbolically "
                    "(%d solutions); those signals stay as symbols.",
                    ", ".join(sorted(str(sym) for sym in pending)),
                    len(found),
                )

        def tidy(expr):
            expr = expr.xreplace(resolved)
            if isinstance(expr, sympy.Expr) and expr.has(self.s):
                expr = sympy.cancel(expr)
            return expr

        self.symbolic_outputs = {
            name: {port: tidy(expr) for port, expr in outputs.items()}
            for name, outputs in raw.items()
        }
        return self.symbolic_outputs

    def _signal_expr(self, src: str, port: int) -> Any:
        """Resolved expression on the wire leaving ``src`` at ``port``."""
        import sympy

        value = self.symbolic_outputs.get(src, {}).get(port)
        if value is None:
            return sympy.Symbol(f"{src}_{port}".replace(" ", "_"))
        return value

    def _block_expr(self, name: str, port: int = 0) -> Any:
        """Output ``port`` of a block, or for a sink (no outputs) the signal on
        its input ``port``."""
        if port in self.symbolic_outputs.get(name, {}):
            return self.symbolic_outputs[name][port]
        if not self.output_map.get(name) and port in self.input_map.get(name, {}):
            return self._signal_expr(*self.input_map[name][port])
        return self._signal_expr(name, port)

    def _compute_block_symbolic(self, block, input_exprs: Dict) -> Optional[Dict]:
        """
        Compute symbolic outputs for a block.

        Args:
            block: The block object
            input_exprs: Dict of input port -> symbolic expression

        Returns:
            Dict of output port -> symbolic expression
        """
        import sympy

        block_type = getattr(block, "block_fn", "")
        params = block.params

        # Check if block has symbolic_execute
        if getattr(block, "block_instance", None) is not None:
            try:
                result = block.block_instance.symbolic_execute(input_exprs, params)
            except Exception as exc:  # noqa: BLE001 -- a bad param must not abort the trace
                logger.warning(
                    "symbolic_execute of '%s' failed: %s", getattr(block, "name", "?"), exc
                )
                result = None
            if result is not None:
                return result

        # Default symbolic implementations for common blocks
        if block_type == "Gain":
            K = params.get("gain", 1.0)
            if isinstance(K, (int, float)):
                K = sympy.Float(K)
            u = input_exprs.get(0, sympy.Symbol("u"))
            return {0: K * u}

        elif block_type == "Sum":
            signs = params.get("sign", params.get("inputs", "++"))
            # Keep only valid sign characters; ignore spaces / count specs / etc.
            valid_signs = [c for c in str(signs) if c in "+-"]
            if len(valid_signs) != len(str(signs)):
                logger.warning(
                    "Sum block '%s' sign spec %r contains non-sign characters; ignoring them.",
                    getattr(block, "name", "?"),
                    signs,
                )
            # Warn if the sign spec does not cover every connected input port.
            connected_ports = set(input_exprs.keys())
            if connected_ports and len(valid_signs) < (max(connected_ports) + 1):
                logger.warning(
                    "Sum block '%s' has connected input port(s) up to index %d "
                    "but only %d sign(s) %r; uncovered inputs are dropped from "
                    "the symbolic expression.",
                    getattr(block, "name", "?"),
                    max(connected_ports),
                    len(valid_signs),
                    signs,
                )
            result = sympy.Integer(0)
            for i, sign in enumerate(valid_signs):
                u = input_exprs.get(i, sympy.Integer(0))
                if sign == "+":
                    result = result + u
                else:
                    result = result - u
            return {0: result}

        elif block_type == "Integrator":
            # In Laplace domain: Y(s) = U(s) / s
            u = input_exprs.get(0, sympy.Symbol("u"))
            return {0: u / self.s}

        elif block_type in ("Deriv", "Derivative"):
            # In Laplace domain: Y(s) = s * U(s)
            u = input_exprs.get(0, sympy.Symbol("u"))
            return {0: self.s * u}

        elif block_type in ("TransferFcn", "TranFn"):
            num = params.get("numerator", [1])
            den = params.get("denominator", [1, 1])

            # Build transfer function
            num_poly = sum(coef * self.s**i for i, coef in enumerate(reversed(num)))
            den_poly = sum(coef * self.s**i for i, coef in enumerate(reversed(den)))

            u = input_exprs.get(0, sympy.Symbol("u"))
            return {0: (num_poly / den_poly) * u}

        elif block_type == "StateSpace":
            # State space in Laplace: Y(s) = C(sI - A)^(-1)B + D * U(s)
            A = np.array(params.get("A", [[0]]))
            B = np.array(params.get("B", [[1]]))
            C = np.array(params.get("C", [[1]]))
            D = np.array(params.get("D", [[0]]))

            n = A.shape[0]

            # Convert to SymPy matrices
            A_sym = sympy.Matrix(A.tolist())
            B_sym = sympy.Matrix(B.tolist())
            C_sym = sympy.Matrix(C.tolist())
            D_sym = sympy.Matrix(D.tolist())

            # (sI - A)^(-1)
            sI = self.s * sympy.eye(n)
            resolvent = (sI - A_sym).inv()

            # Transfer function matrix: C * (sI-A)^(-1) * B + D
            G = C_sym * resolvent * B_sym + D_sym

            # G has shape (n_outputs, n_inputs). Build a per-input column vector
            # from the connected input ports so MIMO systems are handled
            # correctly; a single scalar would be dimensionally wrong here.
            n_inputs = G.shape[1]
            u_vec = sympy.Matrix(
                [input_exprs.get(i, sympy.Symbol(f"u{i}")) for i in range(n_inputs)]
            )
            Y = G * u_vec  # shape (n_outputs, 1)

            if Y.shape == (1, 1):
                return {0: Y[0, 0]}
            # Expose each output as its own port.
            return {row: Y[row, 0] for row in range(Y.shape[0])}

        elif block_type == "PID":
            # Coerce gains to exact sympy Floats so numpy scalars/arrays do not
            # leak into the symbolic expression (matches the Gain block above).
            Kp = sympy.Float(float(params.get("Kp", 1.0)))
            Ki = sympy.Float(float(params.get("Ki", 0.0)))
            Kd = sympy.Float(float(params.get("Kd", 0.0)))
            N = sympy.Float(float(params.get("N", 20.0)))

            sp = input_exprs.get(0, sympy.Symbol("sp"))
            meas = input_exprs.get(1, sympy.Symbol("meas"))
            e = sp - meas

            # P + I/s + filtered-derivative term D*N*s/(s+N).
            C_pid = Kp + Ki / self.s
            if Kd != 0:
                if N <= 0:
                    # Ideal (unfiltered) derivative when the filter is disabled;
                    # avoids the degenerate s/(s+0) = 1 the filtered form gives.
                    logger.warning(
                        "PID block '%s' has derivative filter N=%s <= 0; "
                        "using ideal derivative Kd*s.",
                        getattr(block, "name", "?"),
                        N,
                    )
                    C_pid = C_pid + Kd * self.s
                else:
                    C_pid = C_pid + Kd * N * self.s / (self.s + N)
            return {0: sympy.simplify(C_pid * e)}

        elif block_type in ("Constant", "Step"):
            value = params.get("value", 1.0)
            return {0: sympy.Float(value)}

        elif block_type == "Saturation":
            # Saturation is nonlinear - return input for small-signal
            u = input_exprs.get(0, sympy.Symbol("u"))
            return {0: u}  # Linear approximation

        # Unknown block - return as symbol
        return None

    def extract_transfer_function(
        self, from_block: str, to_block: str, from_port: int = 0, to_port: int = 0
    ) -> Any:
        """
        Extract transfer function between two signals.

        The output of ``from_block`` is replaced by the input U(s) and every
        other source is zeroed, so loops through the path are closed and the
        result is the closed-loop G(s) = Y(s)/U(s).

        Args:
            from_block: Input block name
            to_block: Output block name; for a sink (e.g. a Scope) the signal
                on its input ``to_port`` is the output
            from_port: Input port (default 0)
            to_port: Output port (default 0)

        Returns:
            SymPy transfer function expression G(s)
        """
        import sympy

        U = sympy.Symbol("U")
        self._resolve_outputs({from_block: U}, zero_sources=True)
        Y = self._block_expr(to_block, to_port)

        # A transfer function only exists if Y depends *linearly* on U, i.e.
        # dY/dU is independent of U. Other independent sources (Constant/Step
        # return numeric values) or nonlinear blocks (Saturation) break this,
        # making the naive Y/U a U-dependent, incorrect "transfer function".
        try:
            dYdU = sympy.simplify(sympy.diff(Y, U))
            if dYdU.has(U):
                logger.warning(
                    "Transfer function from '%s' to '%s' is not well defined: "
                    "the output is nonlinear in the input (dY/dU still depends "
                    "on the input). The returned G(s) is unreliable.",
                    from_block,
                    to_block,
                )
            elif Y.has(U) is False:
                logger.warning(
                    "Transfer function from '%s' to '%s' is not well defined: "
                    "the output does not depend on the input.",
                    from_block,
                    to_block,
                )
            else:
                # Affine offset (independent of U) from other sources also
                # invalidates the plain Y/U ratio.
                offset = sympy.simplify(Y - dYdU * U)
                if offset != 0:
                    logger.warning(
                        "Transfer function from '%s' to '%s' has an "
                        "input-independent offset (other sources contribute); "
                        "G(s) = Y/U includes that offset and may be incorrect.",
                        from_block,
                        to_block,
                    )
        except Exception as exc:
            logger.debug("Linearity check for transfer function failed: %s", exc)

        # G(s) = Y(s) / U(s)
        return sympy.cancel(Y / U)

    def get_all_equations(self, input_symbols: Dict[str, Symbol] = None) -> Dict[str, Any]:
        """
        Get symbolic equations for all blocks.

        Args:
            input_symbols: Map of input block names to symbols

        Returns:
            Dict of block_name -> expression of its output 0 (for a sink, of
            the signal on its input 0), in terms of the input symbols
        """
        if input_symbols is None:
            input_symbols = self.create_input_symbols()

        self._resolve_outputs(input_symbols)
        return {block.name: self._block_expr(block.name) for block in self.blocks}

    def linearize_at_point(
        self, operating_point: Dict[str, float], input_block: str, output_block: str
    ) -> Tuple:
        """
        Linearize the system at an operating point.

        Args:
            operating_point: Dict of block_name -> operating value
            input_block: Input block name
            output_block: Output block name

        Returns:
            Tuple of (A, B, C, D) state space matrices on success, or None if
            the transfer function could not be converted to state space (e.g.
            it is non-rational, has symbolic/non-numeric coefficients, or
            scipy's tf2ss raises). Callers must check for None before unpacking.
        """
        import sympy

        # Get transfer function
        G = self.extract_transfer_function(input_block, output_block)

        # Convert to state space (if rational)
        try:
            num, den = sympy.fraction(G)

            # Get coefficients
            num_poly = sympy.Poly(num, self.s)
            den_poly = sympy.Poly(den, self.s)

            num_coeffs = [float(c) for c in num_poly.all_coeffs()]
            den_coeffs = [float(c) for c in den_poly.all_coeffs()]

            from scipy import signal as sig

            A, B, C, D = sig.tf2ss(num_coeffs, den_coeffs)

            return (A, B, C, D)

        except Exception as e:
            logger.warning(
                "Could not convert transfer function from '%s' to '%s' into "
                "state space (non-rational, non-numeric coefficients, or "
                "tf2ss failure): %s",
                input_block,
                output_block,
                e,
            )
            return None

    def to_latex(self, expr, simplified: bool = True) -> str:
        """
        Convert symbolic expression to LaTeX.

        Args:
            expr: SymPy expression
            simplified: Whether to simplify first

        Returns:
            LaTeX string
        """
        import sympy

        if simplified:
            expr = sympy.simplify(expr)
        return sympy.latex(expr)

    def export_equations_latex(self, equations: Dict = None, filename: str = None) -> str:
        """
        Export all equations to LaTeX.

        Args:
            equations: Dict of block_name -> expression
            filename: Output file path (if None, returns string)

        Returns:
            LaTeX document string
        """
        if equations is None:
            equations = self.get_all_equations()

        lines = [
            r"\documentclass{article}",
            r"\usepackage{amsmath}",
            r"\begin{document}",
            r"\section{Block Diagram Equations}",
            r"",
        ]

        for name, expr in equations.items():
            if expr is not None:
                try:
                    latex_expr = self.to_latex(expr)
                    lines.append(r"\subsection{%s}" % name.replace("_", r"\_"))
                    lines.append(r"\begin{equation}")
                    lines.append(f"y = {latex_expr}")
                    lines.append(r"\end{equation}")
                    lines.append("")
                except Exception as exc:
                    logger.warning("Skipping LaTeX export for block '%s': %s", name, exc)

        lines.append(r"\end{document}")

        result = "\n".join(lines)

        if filename:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(result)

        return result
