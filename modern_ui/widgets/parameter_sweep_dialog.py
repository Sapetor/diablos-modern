"""Parameter sweep - run configuration dialog.

Collects what :class:`lib.analysis.parameter_sweep.ParameterSweepRunner` needs:

  * mode      -- 1-D (one parameter) or 2-D (two parameters).
  * each axis -- a block, one of its numeric scalar parameters, and a
                 min/max/points range (expanded to ``np.linspace(min, max, points)``).
  * sim_time / sim_dt -- the simulation horizon and step (default: the diagram's).

The dialog performs no simulation; ``get_selection()`` returns::

    {"axes": [{"block": str, "param": str, "values": ndarray}, ...],
     "sim_time": float, "sim_dt": float}

with one axis entry for 1-D and two for 2-D.
"""

import logging

import numpy as np
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QGroupBox, QLabel,
    QComboBox, QSpinBox, QDoubleSpinBox, QDialogButtonBox,
)

logger = logging.getLogger(__name__)


def numeric_scalar_params(block):
    """Sorted names of a block's numeric scalar params (sweepable knobs).

    Excludes booleans, strings, lists/arrays, and private ``_``-prefixed keys.
    """
    params = getattr(block, "params", None)
    if not isinstance(params, dict):
        return []
    out = []
    for k, v in params.items():
        if k.startswith("_") or isinstance(v, bool):
            continue
        if isinstance(v, (int, float, np.integer, np.floating)):
            out.append(k)
    return sorted(out)


def sweepable_blocks(dsim):
    """Blocks of ``dsim`` that expose at least one numeric scalar parameter."""
    return [b for b in getattr(dsim, "blocks_list", []) if numeric_scalar_params(b)]


class ParameterSweepDialog(QDialog):
    """Configure a 1-D/2-D parameter sweep (axes, ranges, time, dt)."""

    def __init__(self, dsim, parent=None):
        """
        Args:
            dsim: DSim instance; its blocks seed the pickers and its
                ``sim_time`` / ``sim_dt`` seed the time defaults.
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self.dsim = dsim
        self._blocks = {b.name: b for b in sweepable_blocks(dsim)}

        self.setWindowTitle("Parameter Sweep")
        self.setMinimumWidth(460)
        self.setModal(True)

        self.default_sim_time = float(getattr(dsim, "sim_time", 10.0))
        self.default_sim_dt = float(getattr(dsim, "sim_dt", 0.01))

        self._setup_ui()

    # ------------------------------------------------------------------ UI ---
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["1-D (one parameter)", "2-D (two parameters)"])
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        top = QFormLayout()
        top.addRow("Sweep type:", self.mode_combo)
        layout.addLayout(top)

        self._x = self._make_axis_group("Parameter X")
        self._y = self._make_axis_group("Parameter Y")
        layout.addWidget(self._x["group"])
        layout.addWidget(self._y["group"])

        # Simulation horizon / step.
        self.sim_time_spin = QDoubleSpinBox()
        self.sim_time_spin.setRange(0.0, 1_000_000.0)
        self.sim_time_spin.setDecimals(4)
        self.sim_time_spin.setSingleStep(0.1)
        self.sim_time_spin.setValue(self.default_sim_time)

        self.sim_dt_spin = QDoubleSpinBox()
        self.sim_dt_spin.setRange(1e-9, 1_000_000.0)
        self.sim_dt_spin.setDecimals(6)
        self.sim_dt_spin.setSingleStep(0.001)
        self.sim_dt_spin.setValue(self.default_sim_dt)

        sim_form = QFormLayout()
        sim_form.addRow("Simulation time:", self.sim_time_spin)
        sim_form.addRow("Step size (dt):", self.sim_dt_spin)
        layout.addLayout(sim_form)

        if not self._blocks:
            warn = QLabel("No block exposes a numeric scalar parameter to sweep.")
            warn.setStyleSheet("color: #a33; padding: 4px;")
            warn.setWordWrap(True)
            layout.addWidget(warn)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.button_box.button(QDialogButtonBox.Ok).setEnabled(bool(self._blocks))
        layout.addWidget(self.button_box)

        self._on_mode_changed(0)

    def _make_axis_group(self, title):
        """Build one axis's widgets (block, param, min/max/points) in a group box."""
        group = QGroupBox(title)
        form = QFormLayout(group)

        block_combo = QComboBox()
        block_combo.addItems(list(self._blocks.keys()))
        param_combo = QComboBox()

        min_spin = QDoubleSpinBox()
        min_spin.setRange(-1e12, 1e12)
        min_spin.setDecimals(6)
        max_spin = QDoubleSpinBox()
        max_spin.setRange(-1e12, 1e12)
        max_spin.setDecimals(6)
        points_spin = QSpinBox()
        points_spin.setRange(2, 200)
        points_spin.setValue(11)

        form.addRow("Block:", block_combo)
        form.addRow("Parameter:", param_combo)
        form.addRow("Min:", min_spin)
        form.addRow("Max:", max_spin)
        form.addRow("Points:", points_spin)

        ax = {
            "group": group, "block": block_combo, "param": param_combo,
            "min": min_spin, "max": max_spin, "points": points_spin,
        }
        block_combo.currentTextChanged.connect(lambda _t, a=ax: self._on_block_changed(a))
        param_combo.currentTextChanged.connect(lambda _t, a=ax: self._on_param_changed(a))
        if self._blocks:
            self._on_block_changed(ax)
        return ax

    # --------------------------------------------------------------- events ---
    def _on_mode_changed(self, index):
        """Enable the Y axis only in 2-D mode."""
        self._y["group"].setVisible(index == 1)

    def _on_block_changed(self, ax):
        """Repopulate the param combo for the axis's selected block."""
        block = self._blocks.get(ax["block"].currentText())
        ax["param"].blockSignals(True)
        ax["param"].clear()
        if block is not None:
            ax["param"].addItems(numeric_scalar_params(block))
        ax["param"].blockSignals(False)
        self._on_param_changed(ax)

    def _on_param_changed(self, ax):
        """Seed min/max from the parameter's current value (value +/- |value|)."""
        block = self._blocks.get(ax["block"].currentText())
        pname = ax["param"].currentText()
        if block is None or not pname:
            return
        try:
            v = float(block.params.get(pname, 0.0))
        except (TypeError, ValueError):
            v = 0.0
        span = abs(v) if v != 0.0 else 1.0
        ax["min"].setValue(v - span)
        ax["max"].setValue(v + span)

    # ---------------------------------------------------------------- query ---
    def _axis_selection(self, ax):
        return {
            "block": ax["block"].currentText(),
            "param": ax["param"].currentText(),
            "values": np.linspace(
                float(ax["min"].value()),
                float(ax["max"].value()),
                int(ax["points"].value()),
            ),
        }

    def get_selection(self) -> dict:
        """Return the configured sweep (see module docstring)."""
        axes = [self._axis_selection(self._x)]
        if self.mode_combo.currentIndex() == 1:
            axes.append(self._axis_selection(self._y))
        return {
            "axes": axes,
            "sim_time": float(self.sim_time_spin.value()),
            "sim_dt": float(self.sim_dt_spin.value()),
        }
