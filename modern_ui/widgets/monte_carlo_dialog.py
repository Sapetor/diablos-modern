"""
Monte-Carlo ensemble - run configuration dialog.

Collects the parameters needed to launch a Monte-Carlo ensemble of the current
diagram (see :class:`lib.analysis.monte_carlo.MonteCarloRunner`):

  * n_runs      -- how many simulations to run.
  * master_seed -- a single experiment seed. Each run derives its own per-run
                   sub-seed from (master_seed, run_index, block_name), so every
                   run differs yet the whole ensemble is bit-reproducible from
                   this one number. master_seed 0 still produces a reproducible
                   ensemble (derive_seed maps it to a concrete sub-seed).
  * sim_time    -- ensemble simulation horizon (defaults to the diagram's).
  * sim_dt      -- fixed step size (defaults to the diagram's).

The dialog performs no simulation; it only gathers a selection.
``get_selection()`` returns:

    {"n_runs": int, "master_seed": int, "sim_time": float, "sim_dt": float}
"""

import logging

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLabel,
    QSpinBox, QDoubleSpinBox, QDialogButtonBox,
)

logger = logging.getLogger(__name__)


class MonteCarloDialog(QDialog):
    """Configure a Monte-Carlo ensemble run (n_runs, master seed, time, dt)."""

    def __init__(self, dsim, parent=None):
        """
        Args:
            dsim: DSim instance; its ``sim_time`` / ``sim_dt`` seed the defaults.
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self.dsim = dsim

        self.setWindowTitle("Monte-Carlo Ensemble")
        self.setMinimumWidth(420)
        self.setModal(True)

        default_sim_time = float(getattr(dsim, "sim_time", 10.0))
        default_sim_dt = float(getattr(dsim, "sim_dt", 0.01))

        # Number of runs in the ensemble.
        self.n_runs_spin = QSpinBox()
        self.n_runs_spin.setRange(2, 100000)
        self.n_runs_spin.setValue(100)

        # Single experiment seed; the whole ensemble is reproducible from it.
        self.master_seed_spin = QSpinBox()
        self.master_seed_spin.setRange(0, 2_000_000_000)
        self.master_seed_spin.setValue(12345)

        # Simulation horizon.
        self.sim_time_spin = QDoubleSpinBox()
        self.sim_time_spin.setRange(0.0, 1_000_000.0)
        self.sim_time_spin.setDecimals(4)
        self.sim_time_spin.setSingleStep(0.1)
        self.sim_time_spin.setValue(default_sim_time)

        # Fixed step size.
        self.sim_dt_spin = QDoubleSpinBox()
        self.sim_dt_spin.setRange(1e-9, 1_000_000.0)
        self.sim_dt_spin.setDecimals(6)
        self.sim_dt_spin.setSingleStep(0.001)
        self.sim_dt_spin.setValue(default_sim_dt)

        self._setup_ui()

    # ------------------------------------------------------------------ UI ---
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        form = QFormLayout()
        form.addRow("Number of runs:", self.n_runs_spin)
        form.addRow("Master seed:", self.master_seed_spin)
        form.addRow("Simulation time:", self.sim_time_spin)
        form.addRow("Step size (dt):", self.sim_dt_spin)
        layout.addLayout(form)

        helper = QLabel(
            "Each run derives its own per-run seed from the master seed, the run "
            "index, and the block name. Every run differs, yet the whole ensemble "
            "is reproducible from the master seed alone -- rerun with the same "
            "master seed to reproduce the exact ensemble."
        )
        helper.setWordWrap(True)
        layout.addWidget(helper)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    # ---------------------------------------------------------------- query ---
    def get_selection(self) -> dict:
        """
        Return the configured ensemble parameters.

        Returns:
            dict with keys:
                "n_runs":      int, number of Monte-Carlo runs
                "master_seed": int, experiment seed (ensemble reproducible from it)
                "sim_time":    float, simulation horizon
                "sim_dt":      float, fixed step size
        """
        return {
            "n_runs": int(self.n_runs_spin.value()),
            "master_seed": int(self.master_seed_spin.value()),
            "sim_time": float(self.sim_time_spin.value()),
            "sim_dt": float(self.sim_dt_spin.value()),
        }
