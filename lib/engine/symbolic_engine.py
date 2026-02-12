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

import logging
from typing import Dict, List, Any, Tuple, Optional, Set, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from sympy import Symbol

logger = logging.getLogger(__name__)

# Import SymPy (optional dependency)
try:
    import sympy
    from sympy import Symbol, symbols, simplify, expand, factor
    from sympy import Matrix, eye, zeros
    from sympy import latex, mathml
    from sympy import Eq, solve, Function
    from sympy import diff, integrate
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
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
            raise ImportError("SymPy is required for symbolic features. "
                            "Install with: pip install sympy")

        self.dsim = dsim
        self.blocks = []
        self.lines = []
        self.block_map = {}  # name -> block
        self.input_map = {}  # block_name -> {port: (src_block, src_port)}
        self.output_map = {}  # block_name -> {port: [(dst_block, dst_port), ...]}

        # Symbolic state
        self.symbolic_outputs = {}  # block_name -> {port: sympy_expr}
        self.state_variables = {}  # block_name -> [sympy_symbols]
        self.s = Symbol('s')  # Laplace variable

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
            if hasattr(line, 'hidden') and line.hidden:
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
        input_symbols = {}

        if input_blocks is None:
            # Find source blocks (no inputs)
            input_blocks = []
            for block in self.blocks:
                block_type = getattr(block, 'block_fn', '')
                if block_type in ('Step', 'Sine', 'Constant', 'Ramp', 'Inport'):
                    input_blocks.append(block.name)

        for name in input_blocks:
            # Create symbol with nice name
            sym_name = name.replace(' ', '_')
            input_symbols[name] = Symbol(sym_name)

        return input_symbols

    def trace_signal(self, start_block: str, start_port: int = 0,
                    input_symbols: Dict[str, Symbol] = None,
                    visited: Set[str] = None) -> Any:
        """
        Trace a signal backward through the diagram symbolically.

        Args:
            start_block: Block name to start from
            start_port: Input port to trace
            input_symbols: Map of input block names to symbols
            visited: Set of already visited blocks (for loop detection)

        Returns:
            SymPy expression for the signal
        """
        if input_symbols is None:
            input_symbols = {}
        if visited is None:
            visited = set()

        # Check if this block is an input
        if start_block in input_symbols:
            return input_symbols[start_block]

        # Check for loops
        if start_block in visited:
            # Algebraic loop - create feedback variable
            return Symbol(f'{start_block}_fb')

        visited = visited | {start_block}

        # Get the block
        block = self.block_map.get(start_block)
        if block is None:
            return Symbol(f'{start_block}_unknown')

        # Get input connections
        deps = self.input_map.get(start_block, {})

        # Trace inputs recursively
        input_exprs = {}
        for port, (src_block, src_port) in deps.items():
            input_exprs[port] = self._get_block_output(
                src_block, src_port, input_symbols, visited
            )

        # Get symbolic output from this block
        return self._compute_block_symbolic(block, input_exprs)

    def _get_block_output(self, block_name: str, port: int,
                         input_symbols: Dict[str, Symbol],
                         visited: Set[str]) -> Any:
        """Get symbolic output expression for a block's port."""

        # Check if already computed
        if block_name in self.symbolic_outputs:
            outputs = self.symbolic_outputs[block_name]
            if port in outputs:
                return outputs[port]

        # Check if input
        if block_name in input_symbols:
            return input_symbols[block_name]

        # Compute the block's outputs
        block = self.block_map.get(block_name)
        if block is None:
            return Symbol(f'{block_name}_{port}')

        # Get this block's inputs
        deps = self.input_map.get(block_name, {})

        input_exprs = {}
        for in_port, (src_block, src_port) in deps.items():
            if src_block in visited:
                # Loop detected
                input_exprs[in_port] = Symbol(f'{src_block}_{src_port}_fb')
            else:
                input_exprs[in_port] = self._get_block_output(
                    src_block, src_port, input_symbols, visited | {block_name}
                )

        # Compute symbolic output
        outputs = self._compute_block_symbolic(block, input_exprs)

        if outputs is not None:
            self.symbolic_outputs[block_name] = outputs
            return outputs.get(port, Symbol(f'{block_name}_{port}'))

        return Symbol(f'{block_name}_{port}')

    def _compute_block_symbolic(self, block, input_exprs: Dict) -> Optional[Dict]:
        """
        Compute symbolic outputs for a block.

        Args:
            block: The block object
            input_exprs: Dict of input port -> symbolic expression

        Returns:
            Dict of output port -> symbolic expression
        """
        block_type = getattr(block, 'block_fn', '')
        params = block.params

        # Check if block has symbolic_execute
        if hasattr(block, 'block_instance') and block.block_instance:
            result = block.block_instance.symbolic_execute(input_exprs, params)
            if result is not None:
                return result

        # Default symbolic implementations for common blocks
        if block_type == 'Gain':
            K = params.get('gain', 1.0)
            if isinstance(K, (int, float)):
                K = sympy.Float(K)
            u = input_exprs.get(0, Symbol('u'))
            return {0: K * u}

        elif block_type == 'Sum':
            signs = params.get('sign', params.get('inputs', '++'))
            result = sympy.Integer(0)
            for i, sign in enumerate(signs):
                u = input_exprs.get(i, sympy.Integer(0))
                if sign == '+':
                    result = result + u
                else:
                    result = result - u
            return {0: result}

        elif block_type == 'Integrator':
            # In Laplace domain: Y(s) = U(s) / s
            u = input_exprs.get(0, Symbol('u'))
            return {0: u / self.s}

        elif block_type == 'Derivative':
            # In Laplace domain: Y(s) = s * U(s)
            u = input_exprs.get(0, Symbol('u'))
            return {0: self.s * u}

        elif block_type in ('TransferFcn', 'TranFn'):
            num = params.get('numerator', [1])
            den = params.get('denominator', [1, 1])

            # Build transfer function
            num_poly = sum(coef * self.s**i for i, coef in enumerate(reversed(num)))
            den_poly = sum(coef * self.s**i for i, coef in enumerate(reversed(den)))

            u = input_exprs.get(0, Symbol('u'))
            return {0: (num_poly / den_poly) * u}

        elif block_type == 'StateSpace':
            # State space in Laplace: Y(s) = C(sI - A)^(-1)B + D * U(s)
            A = np.array(params.get('A', [[0]]))
            B = np.array(params.get('B', [[1]]))
            C = np.array(params.get('C', [[1]]))
            D = np.array(params.get('D', [[0]]))

            n = A.shape[0]

            # Convert to SymPy matrices
            A_sym = Matrix(A.tolist())
            B_sym = Matrix(B.tolist())
            C_sym = Matrix(C.tolist())
            D_sym = Matrix(D.tolist())

            # (sI - A)^(-1)
            sI = self.s * eye(n)
            resolvent = (sI - A_sym).inv()

            # Transfer function matrix: C * (sI-A)^(-1) * B + D
            G = C_sym * resolvent * B_sym + D_sym

            u = input_exprs.get(0, Symbol('u'))
            if G.shape == (1, 1):
                return {0: G[0, 0] * u}
            else:
                return {0: G * u}

        elif block_type == 'PID':
            Kp = params.get('Kp', 1.0)
            Ki = params.get('Ki', 0.0)
            Kd = params.get('Kd', 0.0)
            N = params.get('N', 20.0)

            # PID transfer function: Kp + Ki/s + Kd*N*s/(s + N)
            sp = input_exprs.get(0, Symbol('sp'))
            meas = input_exprs.get(1, Symbol('meas'))
            e = sp - meas

            # P + I/s + D*N*s/(s+N)
            C_pid = Kp + Ki/self.s + Kd*N*self.s/(self.s + N)
            return {0: simplify(C_pid * e)}

        elif block_type in ('Constant', 'Step'):
            value = params.get('value', 1.0)
            return {0: sympy.Float(value)}

        elif block_type == 'Saturation':
            # Saturation is nonlinear - return input for small-signal
            u = input_exprs.get(0, Symbol('u'))
            return {0: u}  # Linear approximation

        # Unknown block - return as symbol
        return None

    def extract_transfer_function(self, from_block: str, to_block: str,
                                  from_port: int = 0, to_port: int = 0) -> Any:
        """
        Extract transfer function between two signals.

        Args:
            from_block: Input block name
            to_block: Output block name
            from_port: Input port (default 0)
            to_port: Output port (default 0)

        Returns:
            SymPy transfer function expression G(s)
        """
        self.symbolic_outputs = {}  # Reset

        # Create input symbol
        U = Symbol('U')
        input_symbols = {from_block: U}

        # Trace to output
        Y = self._get_block_output(to_block, to_port, input_symbols, set())

        # G(s) = Y(s) / U(s)
        G = simplify(Y / U)

        return G

    def get_all_equations(self, input_symbols: Dict[str, Symbol] = None) -> Dict[str, Any]:
        """
        Get symbolic equations for all blocks.

        Args:
            input_symbols: Map of input block names to symbols

        Returns:
            Dict of block_name -> symbolic output expression
        """
        if input_symbols is None:
            input_symbols = self.create_input_symbols()

        self.symbolic_outputs = {}

        equations = {}
        for block in self.blocks:
            # Skip if already computed
            if block.name in self.symbolic_outputs:
                equations[block.name] = self.symbolic_outputs[block.name]
                continue

            # Compute output
            output = self._get_block_output(block.name, 0, input_symbols, set())
            equations[block.name] = output

        return equations

    def linearize_at_point(self, operating_point: Dict[str, float],
                          input_block: str, output_block: str) -> Tuple:
        """
        Linearize the system at an operating point.

        Args:
            operating_point: Dict of block_name -> operating value
            input_block: Input block name
            output_block: Output block name

        Returns:
            Tuple of (A, B, C, D) state space matrices
        """
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
            logger.warning(f"Could not convert to state space: {e}")
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
        if simplified:
            expr = simplify(expr)
        return latex(expr)

    def to_mathml(self, expr, simplified: bool = True) -> str:
        """
        Convert symbolic expression to MathML.

        Args:
            expr: SymPy expression
            simplified: Whether to simplify first

        Returns:
            MathML string
        """
        if simplified:
            expr = simplify(expr)
        return mathml(expr)

    def export_equations_latex(self, equations: Dict = None,
                              filename: str = None) -> str:
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
            r'\documentclass{article}',
            r'\usepackage{amsmath}',
            r'\begin{document}',
            r'\section{Block Diagram Equations}',
            r''
        ]

        for name, expr in equations.items():
            if expr is not None:
                try:
                    latex_expr = self.to_latex(expr)
                    lines.append(r'\subsection{%s}' % name.replace('_', r'\_'))
                    lines.append(r'\begin{equation}')
                    lines.append(f'y = {latex_expr}')
                    lines.append(r'\end{equation}')
                    lines.append('')
                except Exception:
                    pass

        lines.append(r'\end{document}')

        result = '\n'.join(lines)

        if filename:
            with open(filename, 'w') as f:
                f.write(result)

        return result
