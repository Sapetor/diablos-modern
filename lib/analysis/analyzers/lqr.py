import logging
import ast
import numpy as np
from scipy.linalg import solve_continuous_are
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QGroupBox, QApplication, QFormLayout
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from .base_analyzer import BaseAnalyzer

logger = logging.getLogger(__name__)


class LQRAnalyzer(BaseAnalyzer):
    """Analyzer that solves the continuous LQR problem and displays results."""

    def analyze(self, source_block, canvas, **kwargs):
        """Compute LQR gain from block parameters A, B, Q, R.

        If the LQR block's input is connected to a StateSpace block,
        A and B are read from that block automatically.
        """
        logger.debug(f"LQRAnalyzer called for {source_block.name}")

        # Check for connected StateSpace block → read A, B from it
        ss_block = self._find_connected_statespace(source_block, canvas)

        params = source_block.params
        try:
            if ss_block:
                ss_params = ss_block.params
                A = self._resolve_matrix(ss_params.get('A', '[[0]]'), canvas)
                B = self._resolve_matrix(ss_params.get('B', '[[0]]'), canvas)
                logger.info(f"LQR: reading A, B from connected StateSpace '{ss_block.name}'")
            else:
                A = self._resolve_matrix(params.get('A', '[[0]]'), canvas)
                B = self._resolve_matrix(params.get('B', '[[0]]'), canvas)
            Q = self._resolve_matrix(params.get('Q', '[[1]]'), canvas)
            R = self._resolve_matrix(params.get('R', '[[1]]'), canvas)
        except Exception as e:
            logger.error(f"Failed to parse LQR matrices: {e}")
            self._show_error(f"Failed to parse matrices:\n{e}")
            return None

        # Validate dimensions
        n = A.shape[0]
        if A.shape != (n, n):
            self._show_error(f"A must be square (got {A.shape})")
            return None
        if B.shape[0] != n:
            self._show_error(f"B must have {n} rows to match A (got {B.shape})")
            return None
        m = B.shape[1]
        if Q.shape != (n, n):
            self._show_error(f"Q must be {n}×{n} to match A (got {Q.shape})")
            return None
        if R.shape != (m, m):
            self._show_error(f"R must be {m}×{m} to match B columns (got {R.shape})")
            return None

        # Solve CARE: A'P + PA - PBR⁻¹B'P + Q = 0
        try:
            P = solve_continuous_are(A, B, Q, R)
            K = np.linalg.solve(R, B.T @ P)
            A_cl = A - B @ K
            eig_cl = np.linalg.eigvals(A_cl)
            eig_ol = np.linalg.eigvals(A)
        except np.linalg.LinAlgError as e:
            self._show_error(f"LQR computation failed:\n{e}")
            return None
        except Exception as e:
            self._show_error(f"Unexpected error:\n{e}")
            return None

        # Check stability
        stable = all(e.real < 0 for e in eig_cl)

        # Show results dialog
        plant_source = ss_block.name if ss_block else None
        dialog = LQRResultDialog(
            K=K, P=P, A=A, B=B, Q=Q, R=R,
            eig_cl=eig_cl, eig_ol=eig_ol,
            stable=stable, block_name=source_block.name,
            plant_source=plant_source
        )
        dialog.show()
        return dialog

    def _find_connected_statespace(self, lqr_block, canvas):
        """Find a StateSpace block connected to the LQR block's input."""
        try:
            line_list = canvas.dsim.model.line_list
        except AttributeError:
            return None

        for line in line_list:
            if line.dstblock == lqr_block.name:
                try:
                    src = canvas.dsim.model.get_block_by_name(line.srcblock)
                    if src and src.block_fn in ("StateSpace", "DiscreteStateSpace"):
                        return src
                except AttributeError:
                    pass
        return None

    def _resolve_matrix(self, value, canvas):
        """Parse a matrix parameter: literal, list, or workspace variable name."""
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, list):
            return np.array(value, dtype=float)

        if isinstance(value, str):
            value = value.strip()
            # Try literal eval first (handles "[[1,0],[0,1]]")
            try:
                parsed = ast.literal_eval(value)
                return np.atleast_2d(np.array(parsed, dtype=float))
            except (ValueError, SyntaxError):
                pass

            # Try workspace variable
            try:
                from lib.workspace import WorkspaceManager
                ws = WorkspaceManager()
                ws_val = ws.get_variable(value)
                if ws_val is not None:
                    return np.atleast_2d(np.array(ws_val, dtype=float))
            except Exception:
                pass

            raise ValueError(f"Cannot parse '{value}' as matrix or workspace variable")

        # dict with 'default' key (param dict)
        if isinstance(value, dict) and 'default' in value:
            return self._resolve_matrix(value['default'], canvas)

        return np.atleast_2d(np.array(value, dtype=float))

    def _show_error(self, message):
        """Show error in a message box."""
        from PyQt5.QtWidgets import QMessageBox
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("LQR Error")
        msg.setText(message)
        msg.exec_()


class LQRResultDialog(QDialog):
    """Dialog showing LQR computation results with copy buttons."""

    def __init__(self, K, P, A, B, Q, R, eig_cl, eig_ol, stable, block_name,
                 plant_source=None, parent=None):
        super().__init__(parent)
        self.K = K
        self.P = P
        self.setWindowTitle(f"LQR Result: {block_name}")
        self.setMinimumWidth(500)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        self._build_ui(K, P, A, B, Q, R, eig_cl, eig_ol, stable, plant_source)

    def _build_ui(self, K, P, A, B, Q, R, eig_cl, eig_ol, stable, plant_source):
        layout = QVBoxLayout(self)
        mono = QFont("Menlo, Consolas, monospace", 11)

        # Plant source info
        if plant_source:
            src_label = QLabel(f"Plant: A, B from '{plant_source}'")
            src_label.setStyleSheet("color: #336699; font-style: italic;")
            layout.addWidget(src_label)

        # Gain K — the main result
        k_group = QGroupBox("Optimal Gain K  (u = −Kx)")
        k_layout = QVBoxLayout(k_group)
        k_text = QTextEdit()
        k_text.setFont(mono)
        k_text.setReadOnly(True)
        k_text.setPlainText(self._format_matrix(K))
        k_text.setMaximumHeight(80 + 20 * K.shape[0])
        k_layout.addWidget(k_text)

        btn_row = QHBoxLayout()
        copy_k_btn = QPushButton("Copy K (Python list)")
        copy_k_btn.clicked.connect(lambda: self._copy_to_clipboard(self._to_python_list(K)))
        btn_row.addWidget(copy_k_btn)

        copy_k_flat = QPushButton("Copy K (flat)")
        copy_k_flat.clicked.connect(lambda: self._copy_to_clipboard(self._format_matrix(K)))
        btn_row.addWidget(copy_k_flat)
        btn_row.addStretch()
        k_layout.addLayout(btn_row)
        layout.addWidget(k_group)

        # Eigenvalues
        eig_group = QGroupBox("Closed-Loop Eigenvalues of (A − BK)")
        eig_layout = QVBoxLayout(eig_group)
        eig_text = QLabel(self._format_eigenvalues(eig_cl))
        eig_text.setFont(mono)
        eig_text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        eig_layout.addWidget(eig_text)

        status = QLabel("✓ Stable" if stable else "✗ UNSTABLE")
        status.setStyleSheet(
            "color: green; font-weight: bold;" if stable
            else "color: red; font-weight: bold;"
        )
        eig_layout.addWidget(status)

        # Open-loop eigenvalues for comparison
        ol_label = QLabel(f"Open-loop eigenvalues: {self._format_eigenvalues(eig_ol)}")
        ol_label.setFont(QFont("Menlo, Consolas, monospace", 10))
        ol_label.setStyleSheet("color: gray;")
        ol_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        eig_layout.addWidget(ol_label)
        layout.addWidget(eig_group)

        # Cost matrix P
        p_group = QGroupBox("Riccati Solution P")
        p_layout = QVBoxLayout(p_group)
        p_text = QTextEdit()
        p_text.setFont(mono)
        p_text.setReadOnly(True)
        p_text.setPlainText(self._format_matrix(P))
        p_text.setMaximumHeight(80 + 20 * P.shape[0])
        p_layout.addWidget(p_text)

        copy_p_btn = QPushButton("Copy P (Python list)")
        copy_p_btn.clicked.connect(lambda: self._copy_to_clipboard(self._to_python_list(P)))
        p_layout.addWidget(copy_p_btn)
        layout.addWidget(p_group)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)

    def _format_matrix(self, M):
        """Pretty-print matrix with aligned columns."""
        if M.shape[0] == 1 and M.shape[1] == 1:
            return f"{M[0, 0]:.6g}"
        rows = []
        for i in range(M.shape[0]):
            row = "  ".join(f"{M[i, j]:12.6g}" for j in range(M.shape[1]))
            rows.append(f"[ {row} ]")
        return "\n".join(rows)

    def _format_eigenvalues(self, eigs):
        """Format eigenvalues as readable string."""
        parts = []
        for e in sorted(eigs, key=lambda x: x.real):
            if abs(e.imag) < 1e-10:
                parts.append(f"{e.real:.4g}")
            else:
                parts.append(f"{e.real:.4g} ± {abs(e.imag):.4g}j")
        # Deduplicate conjugate pairs
        seen = set()
        unique = []
        for p in parts:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        return ",  ".join(unique)

    def _to_python_list(self, M, digits=6):
        """Convert matrix to Python list literal pasteable into DiaBloS block params.

        Rounds to `digits` significant figures and strips trailing zeros
        so the output is clean (e.g. ``[[1.0, 1.73205]]`` not
        ``[[1.0000000000000002, 1.7320508075688772]]``).
        """
        def _fmt(x):
            s = f"{x:.{digits}g}"
            # Ensure floats keep a decimal point (so ast.literal_eval → float)
            if '.' not in s and 'e' not in s:
                s += '.0'
            return s

        if M.ndim == 1:
            return '[' + ', '.join(_fmt(v) for v in M) + ']'
        return '[' + ', '.join(
            '[' + ', '.join(_fmt(v) for v in row) + ']'
            for row in M
        ) + ']'

    def _copy_to_clipboard(self, text):
        """Copy text to system clipboard."""
        clipboard = QApplication.clipboard()
        clipboard.setText(text)
