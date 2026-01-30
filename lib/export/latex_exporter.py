"""
LaTeX Exporter - Export equations and analysis to LaTeX/MathML

Provides export capabilities for:
- Block diagram equations
- Transfer functions
- State-space models
- Bode/frequency response data
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class LaTeXExporter:
    """
    Exports symbolic equations and analysis results to LaTeX.
    """

    def __init__(self):
        """Initialize the LaTeX exporter."""
        # Check for SymPy
        try:
            import sympy
            self.sympy = sympy
            self.sympy_available = True
        except ImportError:
            self.sympy_available = False
            logger.warning("SymPy not available for LaTeX export")

    def to_latex(self, expr) -> str:
        """
        Convert a SymPy expression to LaTeX.

        Args:
            expr: SymPy expression or numeric value

        Returns:
            LaTeX string
        """
        if not self.sympy_available:
            return str(expr)

        try:
            return self.sympy.latex(expr)
        except Exception:
            return str(expr)

    def matrix_to_latex(self, M: np.ndarray, name: str = "A",
                       precision: int = 4) -> str:
        """
        Convert a numpy matrix to LaTeX.

        Args:
            M: Numpy array
            name: Matrix name
            precision: Decimal precision

        Returns:
            LaTeX string
        """
        if M is None:
            return ""

        M = np.atleast_2d(M)
        rows, cols = M.shape

        latex = f"{name} = \\begin{{bmatrix}}\n"

        for i in range(rows):
            row_str = " & ".join([f"{M[i, j]:.{precision}g}" for j in range(cols)])
            if i < rows - 1:
                latex += f"  {row_str} \\\\\n"
            else:
                latex += f"  {row_str}\n"

        latex += "\\end{bmatrix}"

        return latex

    def transfer_function_to_latex(self, num: List[float], den: List[float],
                                   name: str = "G") -> str:
        """
        Convert transfer function to LaTeX.

        Args:
            num: Numerator coefficients (highest power first)
            den: Denominator coefficients (highest power first)
            name: Transfer function name

        Returns:
            LaTeX string
        """
        def poly_to_latex(coeffs):
            terms = []
            n = len(coeffs) - 1

            for i, c in enumerate(coeffs):
                power = n - i

                if abs(c) < 1e-10:
                    continue

                # Format coefficient
                if abs(c - 1.0) < 1e-10 and power > 0:
                    coef_str = ""
                elif abs(c + 1.0) < 1e-10 and power > 0:
                    coef_str = "-"
                else:
                    coef_str = f"{c:.4g}"

                # Format power
                if power == 0:
                    term = coef_str if coef_str else "1"
                elif power == 1:
                    term = f"{coef_str}s"
                else:
                    term = f"{coef_str}s^{{{power}}}"

                terms.append(term)

            if not terms:
                return "0"

            result = terms[0]
            for term in terms[1:]:
                if term.startswith("-"):
                    result += f" {term}"
                else:
                    result += f" + {term}"

            return result

        num_latex = poly_to_latex(num)
        den_latex = poly_to_latex(den)

        return f"{name}(s) = \\frac{{{num_latex}}}{{{den_latex}}}"

    def state_space_to_latex(self, A: np.ndarray, B: np.ndarray,
                            C: np.ndarray, D: np.ndarray,
                            precision: int = 4) -> str:
        """
        Convert state-space model to LaTeX.

        Args:
            A, B, C, D: State-space matrices
            precision: Decimal precision

        Returns:
            LaTeX string
        """
        latex = "\\begin{align}\n"
        latex += "\\dot{x} &= Ax + Bu \\\\\n"
        latex += "y &= Cx + Du\n"
        latex += "\\end{align}\n\n"
        latex += "\\text{where:}\n\n"

        latex += self.matrix_to_latex(A, "A", precision) + "\n\n"
        latex += self.matrix_to_latex(B, "B", precision) + "\n\n"
        latex += self.matrix_to_latex(C, "C", precision) + "\n\n"
        latex += self.matrix_to_latex(D, "D", precision)

        return latex

    def eigenvalues_to_latex(self, eigenvalues: np.ndarray) -> str:
        """
        Format eigenvalues as LaTeX.

        Args:
            eigenvalues: Array of eigenvalues (may be complex)

        Returns:
            LaTeX string
        """
        latex = "\\lambda = \\{"

        formatted = []
        for ev in eigenvalues:
            real = np.real(ev)
            imag = np.imag(ev)

            if abs(imag) < 1e-10:
                formatted.append(f"{real:.4g}")
            else:
                if imag > 0:
                    formatted.append(f"{real:.4g} + {imag:.4g}j")
                else:
                    formatted.append(f"{real:.4g} - {abs(imag):.4g}j")

        latex += ", ".join(formatted)
        latex += "\\}"

        return latex

    def export_document(self, content: Dict, filename: str = None) -> str:
        """
        Export a complete LaTeX document.

        Args:
            content: Dict with sections and equations
            filename: Output file path (optional)

        Returns:
            LaTeX document string
        """
        doc = [
            r'\documentclass{article}',
            r'\usepackage{amsmath}',
            r'\usepackage{amssymb}',
            r'\usepackage{graphicx}',
            r'\usepackage{booktabs}',
            r'\usepackage[margin=1in]{geometry}',
            r'',
            r'\title{Block Diagram Analysis}',
            r'\author{Generated by Diablos}',
            r'\date{\today}',
            r'',
            r'\begin{document}',
            r'\maketitle',
            r'',
        ]

        # Add sections
        if 'title' in content:
            doc.append(r'\section{%s}' % content['title'])
            doc.append('')

        if 'description' in content:
            doc.append(content['description'])
            doc.append('')

        if 'transfer_function' in content:
            doc.append(r'\section{Transfer Function}')
            doc.append(r'\begin{equation}')
            doc.append(content['transfer_function'])
            doc.append(r'\end{equation}')
            doc.append('')

        if 'state_space' in content:
            doc.append(r'\section{State-Space Model}')
            doc.append(content['state_space'])
            doc.append('')

        if 'eigenvalues' in content:
            doc.append(r'\section{Eigenvalue Analysis}')
            doc.append(r'\begin{equation}')
            doc.append(content['eigenvalues'])
            doc.append(r'\end{equation}')
            doc.append('')

        if 'stability' in content:
            doc.append(r'\subsection{Stability}')
            doc.append(content['stability'])
            doc.append('')

        if 'equations' in content:
            doc.append(r'\section{Block Equations}')
            for name, eq in content['equations'].items():
                doc.append(r'\subsection{%s}' % name.replace('_', r'\_'))
                doc.append(r'\begin{equation}')
                doc.append(f'y = {eq}')
                doc.append(r'\end{equation}')
                doc.append('')

        doc.append(r'\end{document}')

        result = '\n'.join(doc)

        if filename:
            with open(filename, 'w') as f:
                f.write(result)
            logger.info(f"LaTeX document exported to {filename}")

        return result

    def export_equations_only(self, equations: Dict[str, str],
                             filename: str = None) -> str:
        """
        Export only the equations (no document preamble).

        Args:
            equations: Dict of name -> LaTeX equation
            filename: Output file (optional)

        Returns:
            LaTeX fragment
        """
        lines = []

        for name, eq in equations.items():
            lines.append(f"% {name}")
            lines.append(r'\begin{equation}')
            lines.append(f'  {name} = {eq}')
            lines.append(r'\end{equation}')
            lines.append('')

        result = '\n'.join(lines)

        if filename:
            with open(filename, 'w') as f:
                f.write(result)

        return result


class MathMLExporter:
    """
    Exports equations to MathML format.
    """

    def __init__(self):
        """Initialize MathML exporter."""
        try:
            import sympy
            self.sympy = sympy
            self.sympy_available = True
        except ImportError:
            self.sympy_available = False

    def to_mathml(self, expr) -> str:
        """
        Convert expression to MathML.

        Args:
            expr: SymPy expression

        Returns:
            MathML string
        """
        if not self.sympy_available:
            return f"<mi>{str(expr)}</mi>"

        try:
            from sympy.printing.mathml import mathml
            return mathml(expr)
        except Exception:
            return f"<mi>{str(expr)}</mi>"

    def wrap_mathml(self, content: str) -> str:
        """
        Wrap MathML content in math tags.

        Args:
            content: MathML content

        Returns:
            Complete MathML element
        """
        return f'<math xmlns="http://www.w3.org/1998/Math/MathML">{content}</math>'


class MathematicaExporter:
    """
    Exports equations to Mathematica format.
    """

    def __init__(self):
        """Initialize Mathematica exporter."""
        pass

    def to_mathematica(self, expr) -> str:
        """
        Convert SymPy expression to Mathematica code.

        Args:
            expr: SymPy expression

        Returns:
            Mathematica code string
        """
        try:
            import sympy
            from sympy.printing.mathematica import mathematica_code
            return mathematica_code(expr)
        except ImportError:
            return str(expr)

    def transfer_function_to_mathematica(self, num: List[float],
                                        den: List[float]) -> str:
        """
        Convert transfer function to Mathematica TransferFunctionModel.

        Args:
            num: Numerator coefficients
            den: Denominator coefficients

        Returns:
            Mathematica code
        """
        num_str = "{" + ", ".join(str(c) for c in num) + "}"
        den_str = "{" + ", ".join(str(c) for c in den) + "}"

        return f"TransferFunctionModel[{{{num_str}, {den_str}}}, s]"

    def state_space_to_mathematica(self, A: np.ndarray, B: np.ndarray,
                                   C: np.ndarray, D: np.ndarray) -> str:
        """
        Convert state-space to Mathematica StateSpaceModel.
        """
        def array_to_mma(arr):
            if arr.ndim == 1:
                return "{" + ", ".join(str(x) for x in arr) + "}"
            return "{" + ", ".join("{" + ", ".join(str(x) for x in row) + "}"
                                  for row in arr) + "}"

        return f"StateSpaceModel[{{{array_to_mma(A)}, {array_to_mma(B)}, {array_to_mma(C)}, {array_to_mma(D)}}}]"
