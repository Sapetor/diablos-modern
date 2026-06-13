"""Results window for the "Linearize & Analyze" feature.

``LinearizationResultWindow`` is a pure *view*: it CONSUMES the shared
linearization result-dict contract (produced by the AnalysisController) and
only renders the arrays already present in it. It performs no linearization or
control-systems computation of its own -- every number it shows or plots comes
straight from the supplied ``result`` dict.

Result-dict contract (see the feature spec)::

    result = {
      "ok": bool, "error": str,
      "n_states": int,
      "state_names": [str], "input_names": [str], "output_names": [str],
      "A": [[float]], "B": [[float]]|[], "C": [[float]]|[], "D": [[float]]|[],
      "poles": [[re,im], ...], "zeros": [[re,im], ...],
      "is_stable": bool, "time_constants": [float],
      "oscillatory_modes": [{"omega_n","zeta","period"}],
      "gain_margin_db": float|None, "phase_margin_deg": float|None,
      "gain_crossover": float|None, "phase_crossover": float|None,
      "tf_num": [float]|None, "tf_den": [float]|None,
      "bode": {"w":[float], "mag_db":[float], "phase_deg":[float]}|None,
      "step_response": {"t":[float], "y":[float]}|None,
      "impulse_response": {"t":[float], "y":[float]}|None,
      "controllable": bool|None, "observable": bool|None,
      "operating_point": {block_name: value},
      "summary": str,
    }
"""

import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QTabWidget,
    QLabel,
    QPlainTextEdit,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont


class LinearizationResultWindow(QWidget):
    """Tabbed window presenting a linearization/analysis result.

    Layout:
      * "Pole-Zero" -- scatter of poles (x) and zeros (o) on the complex plane.
      * "Bode"      -- stacked magnitude (dB) and phase (deg) vs log-frequency,
                       or a friendly hint when no Bode data is available.
      * "Step"      -- step response vs time (when a SISO TF is available).
      * "Impulse"   -- impulse response vs time (when a SISO TF is available).
      * "Summary"   -- read-only text: the human summary plus formatted
                       A/B/C/D matrices, stability margins, and the
                       controllable/observable flags.

    When ``result["ok"]`` is False the tabs are replaced by the error message.
    """

    def __init__(self, result: dict, parent=None):
        super().__init__(parent)
        self.result = result or {}

        self.setWindowTitle("Linearized System Analysis")
        self.resize(820, 620)

        layout = QVBoxLayout()
        self.setLayout(layout)

        if not self.result.get("ok", False):
            self._build_error_view(layout)
            return

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self.tabs.addTab(self._build_pole_zero_tab(), "Pole-Zero")
        self.tabs.addTab(self._build_bode_tab(), "Bode")
        self.tabs.addTab(
            self._build_response_tab("step_response", "Step Response"),
            "Step",
        )
        self.tabs.addTab(
            self._build_response_tab("impulse_response", "Impulse Response"),
            "Impulse",
        )
        self.tabs.addTab(self._build_summary_tab(), "Summary")

    # ------------------------------------------------------------------ error
    def _build_error_view(self, layout):
        msg = self.result.get("error") or "Linearization failed."
        label = QLabel(str(msg))
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("color: #b00020; font-size: 13px; padding: 24px;")
        layout.addWidget(label)

    # -------------------------------------------------------------- pole-zero
    def _build_pole_zero_tab(self):
        plot = pg.PlotWidget()
        plot.setBackground("w")
        plot.showGrid(x=True, y=True, alpha=0.3)
        plot.setLabel("bottom", "Real")
        plot.setLabel("left", "Imaginary")
        plot.setTitle("Pole-Zero Map")
        for axis in ("bottom", "left"):
            plot.getAxis(axis).setPen("k")
            plot.getAxis(axis).setTextPen("k")
        # Equal aspect so distances on the complex plane are not distorted.
        plot.getViewBox().setAspectLocked(True, ratio=1.0)
        legend = plot.addLegend(offset=(10, 10))

        poles = self._as_xy(self.result.get("poles"))
        zeros = self._as_xy(self.result.get("zeros"))

        # Axes through the origin.
        plot.addItem(pg.InfiniteLine(angle=0, pen=pg.mkPen("k", width=1)))
        plot.addItem(pg.InfiniteLine(angle=90, pen=pg.mkPen("k", width=1)))

        if zeros.size:
            zero_item = pg.ScatterPlotItem(
                x=zeros[:, 0],
                y=zeros[:, 1],
                symbol="o",
                size=12,
                pen=pg.mkPen("b", width=2),
                brush=None,
                name="Zeros",
            )
            plot.addItem(zero_item)
            legend.addItem(zero_item, "Zeros")

        if poles.size:
            pole_item = pg.ScatterPlotItem(
                x=poles[:, 0],
                y=poles[:, 1],
                symbol="x",
                size=14,
                pen=pg.mkPen("r", width=2),
                brush=pg.mkBrush("r"),
                name="Poles",
            )
            plot.addItem(pole_item)
            legend.addItem(pole_item, "Poles")

        if not poles.size and not zeros.size:
            container = QWidget()
            box = QVBoxLayout()
            container.setLayout(box)
            box.addWidget(plot)
            box.addWidget(QLabel("No poles or zeros to display."))
            return container

        return plot

    # ------------------------------------------------------------------- bode
    def _build_bode_tab(self):
        container = QWidget()
        box = QVBoxLayout()
        container.setLayout(box)

        bode = self.result.get("bode")
        if not bode or not bode.get("w"):
            label = QLabel(
                "Designate input & output blocks to compute a Bode plot."
            )
            label.setAlignment(Qt.AlignCenter)
            label.setWordWrap(True)
            label.setStyleSheet("color: #555; font-size: 13px; padding: 24px;")
            box.addWidget(label)
            return container

        w = np.asarray(bode.get("w", []), dtype=float)
        mag_db = np.asarray(bode.get("mag_db", []), dtype=float)
        phase_deg = np.asarray(bode.get("phase_deg", []), dtype=float)

        mag_plot = pg.PlotWidget()
        self._style_plot(mag_plot, "Magnitude (dB)", "Frequency (rad/s)", "Magnitude (dB)")
        mag_plot.setLogMode(x=True, y=False)
        if w.size and mag_db.size:
            mag_plot.plot(w, mag_db, pen=pg.mkPen("b", width=2))

        phase_plot = pg.PlotWidget()
        self._style_plot(phase_plot, "Phase (deg)", "Frequency (rad/s)", "Phase (deg)")
        phase_plot.setLogMode(x=True, y=False)
        if w.size and phase_deg.size:
            phase_plot.plot(w, phase_deg, pen=pg.mkPen("r", width=2))

        # Share the x-axis so panning/zooming stays aligned across the pair.
        phase_plot.setXLink(mag_plot)

        box.addWidget(mag_plot)
        box.addWidget(phase_plot)
        return container

    # --------------------------------------------------------- time responses
    def _build_response_tab(self, key, title):
        """Build a time-response tab from result[key] = {"t":[...], "y":[...]}."""
        container = QWidget()
        box = QVBoxLayout()
        container.setLayout(box)

        data = self.result.get(key)
        if not data or not data.get("t"):
            label = QLabel(
                "Designate input & output blocks to compute the time response."
            )
            label.setAlignment(Qt.AlignCenter)
            label.setWordWrap(True)
            label.setStyleSheet("color: #555; font-size: 13px; padding: 24px;")
            box.addWidget(label)
            return container

        t = np.asarray(data.get("t", []), dtype=float)
        y = np.asarray(data.get("y", []), dtype=float)

        plot = pg.PlotWidget()
        self._style_plot(plot, title, "Time (s)", "Output")
        if t.size and y.size:
            plot.plot(t, y, pen=pg.mkPen("b", width=2))
        box.addWidget(plot)
        return container

    # ---------------------------------------------------------------- summary
    def _build_summary_tab(self):
        text = QPlainTextEdit()
        text.setReadOnly(True)
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.Monospace)
        text.setFont(mono)
        text.setPlainText(self._compose_summary_text())
        return text

    def _compose_summary_text(self):
        r = self.result
        lines = []

        summary = r.get("summary")
        if summary:
            lines.append(str(summary))
            lines.append("")

        n_states = r.get("n_states")
        if n_states is not None:
            lines.append(f"States: {n_states}")
        state_names = r.get("state_names") or []
        if state_names:
            lines.append("  " + ", ".join(map(str, state_names)))
        input_names = r.get("input_names") or []
        if input_names:
            lines.append("Inputs:  " + ", ".join(map(str, input_names)))
        output_names = r.get("output_names") or []
        if output_names:
            lines.append("Outputs: " + ", ".join(map(str, output_names)))
        lines.append("")

        # State-space matrices.
        for key in ("A", "B", "C", "D"):
            mat = r.get(key)
            block = self._format_matrix(key, mat)
            if block:
                lines.append(block)
                lines.append("")

        # Stability / poles.
        is_stable = r.get("is_stable")
        if is_stable is not None:
            lines.append(f"Stable: {'yes' if is_stable else 'no'}")

        tcs = r.get("time_constants") or []
        if tcs:
            lines.append(
                "Time constants: "
                + ", ".join(f"{float(t):.4g}" for t in tcs)
            )

        modes = r.get("oscillatory_modes") or []
        if modes:
            lines.append("Oscillatory modes:")
            for m in modes:
                wn = m.get("omega_n")
                zeta = m.get("zeta")
                period = m.get("period")
                lines.append(
                    "  "
                    + ", ".join(
                        part
                        for part in (
                            f"wn={float(wn):.4g}" if wn is not None else None,
                            f"zeta={float(zeta):.4g}" if zeta is not None else None,
                            f"T={float(period):.4g}" if period is not None else None,
                        )
                        if part is not None
                    )
                )

        # Stability margins.
        margin_lines = []
        gm = r.get("gain_margin_db")
        if gm is not None:
            gco = r.get("gain_crossover")
            suffix = f" @ {float(gco):.4g} rad/s" if gco is not None else ""
            margin_lines.append(f"  Gain margin:  {float(gm):.4g} dB{suffix}")
        pm = r.get("phase_margin_deg")
        if pm is not None:
            pco = r.get("phase_crossover")
            suffix = f" @ {float(pco):.4g} rad/s" if pco is not None else ""
            margin_lines.append(f"  Phase margin: {float(pm):.4g} deg{suffix}")
        if margin_lines:
            lines.append("")
            lines.append("Stability margins:")
            lines.extend(margin_lines)

        # Transfer function.
        num = r.get("tf_num")
        den = r.get("tf_den")
        if num is not None and den is not None:
            lines.append("")
            lines.append("Transfer function:")
            lines.append("  num: " + self._format_coeffs(num))
            lines.append("  den: " + self._format_coeffs(den))

        # Controllability / observability.
        co_lines = []
        ctrl = r.get("controllable")
        if ctrl is not None:
            co_lines.append(f"  Controllable: {'yes' if ctrl else 'no'}")
        obs = r.get("observable")
        if obs is not None:
            co_lines.append(f"  Observable:   {'yes' if obs else 'no'}")
        if co_lines:
            lines.append("")
            lines.extend(co_lines)

        # Operating point.
        op = r.get("operating_point") or {}
        if op:
            lines.append("")
            lines.append("Operating point:")
            for name, val in op.items():
                lines.append(f"  {name} = {self._format_scalar(val)}")

        return "\n".join(lines).rstrip()

    # ----------------------------------------------------------- helpers/util
    @staticmethod
    def _as_xy(points):
        """Coerce a [[re, im], ...] list into an (N, 2) float array."""
        if points is None:
            return np.empty((0, 2), dtype=float)
        arr = np.asarray(points, dtype=float)
        if arr.size == 0:
            return np.empty((0, 2), dtype=float)
        arr = np.atleast_2d(arr)
        if arr.shape[1] < 2:
            # Real-only list -> pad imaginary parts with zeros.
            arr = np.column_stack([arr[:, 0], np.zeros(arr.shape[0])])
        return arr[:, :2]

    @staticmethod
    def _style_plot(plot, title, xlabel, ylabel):
        plot.setBackground("w")
        plot.showGrid(x=True, y=True, alpha=0.3)
        plot.setTitle(title)
        plot.setLabel("bottom", xlabel)
        plot.setLabel("left", ylabel)
        for axis in ("bottom", "left"):
            plot.getAxis(axis).setPen("k")
            plot.getAxis(axis).setTextPen("k")

    @staticmethod
    def _format_scalar(val):
        try:
            f = float(np.asarray(val).flatten()[0]) if np.ndim(val) else float(val)
            return f"{f:.4g}"
        except (TypeError, ValueError):
            return str(val)

    @staticmethod
    def _format_coeffs(coeffs):
        try:
            flat = np.atleast_1d(np.asarray(coeffs, dtype=float)).flatten()
            return "[" + ", ".join(f"{c:.4g}" for c in flat) + "]"
        except (TypeError, ValueError):
            return str(coeffs)

    @classmethod
    def _format_matrix(cls, name, mat):
        """Render a 2-D matrix as ``name =`` followed by aligned rows.

        Returns an empty string for empty/absent matrices.
        """
        if mat is None:
            return ""
        arr = np.asarray(mat, dtype=float)
        if arr.size == 0:
            return ""
        arr = np.atleast_2d(arr)
        # Format every cell, then pad to a common column width for alignment.
        cells = [[f"{v:.4g}" for v in row] for row in arr]
        width = max((len(c) for row in cells for c in row), default=1)
        body = []
        for row in cells:
            body.append("  [" + "  ".join(c.rjust(width) for c in row) + "]")
        return f"{name} =\n" + "\n".join(body)
