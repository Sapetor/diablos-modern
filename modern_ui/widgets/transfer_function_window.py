"""Result window for Analysis > Transfer Function Between Blocks.

Shows ``G(s)``, its poles, zeros and DC gain, and the reduction steps of
Mason's gain formula (forward paths, loops, Delta, Delta_k) as selectable
plain text, one section per output signal.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget

from lib.analysis.sysid import format_tf
from lib.i18n import tr
from modern_ui.themes.theme_manager import get_mono_font


def _num(z) -> str:
    z = complex(z)
    if abs(z.imag) < 1e-9 * max(1.0, abs(z.real)):
        return f"{z.real:.6g}"
    return f"{z.real:.6g} {'+' if z.imag >= 0 else '-'} {abs(z.imag):.6g}j"


def _rat(r) -> str:
    text = format_tf(r.num, r.den, digits=6)
    # "(2 s + 1) / (1)" -> "2 s + 1": a plain polynomial needs no fraction.
    if len(r.den) == 1 and text.endswith(" / (1)"):
        text = text[: -len(" / (1)")]
        if text.startswith("(") and text.endswith(")"):
            text = text[1:-1]
    return text


def describe(result, names) -> str:
    """Plain-text report for one ``block_algebra.Result``.

    ``names`` maps block names to the labels shown on the canvas.
    """

    def node(sig):
        return names.get(sig[0], sig[0])

    tf = result.tf
    lines = [f"G(s) = {_rat(tf)}", ""]
    poles = sorted(result.poles, key=lambda z: (z.real, z.imag))
    zeros = sorted(result.zeros, key=lambda z: (z.real, z.imag))
    lines.append(tr("Poles: {values}", values=", ".join(_num(p) for p in poles) or "-"))
    lines.append(tr("Zeros: {values}", values=", ".join(_num(z) for z in zeros) or "-"))
    lines.append(tr("DC gain: {value}", value=f"{tf.dc_gain():.6g}"))
    stable = bool(np.all(np.real(poles) < 0)) if len(poles) else True
    lines.append(tr("Stable: yes") if stable else tr("Stable: no"))
    lines += ["", tr("Reduction steps (Mason's gain formula)"), ""]
    for k, ((path, gain), dk) in enumerate(zip(result.steps.paths, result.steps.path_deltas), 1):
        lines.append(f"P{k} = {_rat(gain)}")
        lines.append("     " + " -> ".join(node(s) for s in path))
        lines.append(f"Delta{k} = {_rat(dk)}")
    if result.steps.loops:
        for i, (cycle, gain) in enumerate(result.steps.loops, 1):
            lines.append(f"L{i} = {_rat(gain)}")
            lines.append("     " + " -> ".join(node(s) for s in list(cycle) + [cycle[0]]))
    else:
        lines.append(tr("No feedback loops."))
    lines.append(f"Delta = 1 - sum(L) + sum(non-touching pairs) - ... = {_rat(result.steps.delta)}")
    lines.append("G = sum(Pk * Deltak) / Delta")
    return "\n".join(lines)


class TransferFunctionWindow(QWidget):
    """Top-level window listing one transfer function per output signal."""

    def __init__(self, sections, parent=None):
        """``sections``: list of ``(title, text)``."""
        super().__init__(parent)
        self.setWindowTitle(tr("Transfer Function"))
        layout = QVBoxLayout(self)
        self.text_views = []
        self.headings = [title for title, _ in sections]
        for title, text in sections:
            heading = QLabel(title)
            heading.setStyleSheet("font-weight: 600;")
            layout.addWidget(heading)
            view = QPlainTextEdit()
            view.setReadOnly(True)
            view.setFont(get_mono_font())
            view.setPlainText(text)
            layout.addWidget(view)
            self.text_views.append(view)
        row = QHBoxLayout()
        row.addStretch(1)
        close = QPushButton(tr("Close"))
        close.clicked.connect(self.close)
        row.addWidget(close)
        layout.addLayout(row)
        self.resize(640, 360 + 220 * max(0, len(sections) - 1))
