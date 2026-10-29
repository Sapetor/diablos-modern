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
  * workers     -- parallel worker processes (1 = run in the app's process).
  * keep_traces -- keep every run's trace (exact percentile bands). When off,
                   only a small sample of traces is kept for display and the
                   band is mean +/- std; mean/std/metrics stay exact.

The dialog performs no simulation; it only gathers a selection.
``get_selection()`` returns:

    {"n_runs": int, "master_seed": int, "sim_time": float, "sim_dt": float,
     "workers": int, "keep_traces": True | int}
"""

import logging
import os

from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QVBoxLayout,
    QFormLayout,
    QLabel,
    QSpinBox,
    QDoubleSpinBox,
    QDialogButtonBox,
)

from lib.analysis.parallel import default_workers
from lib.i18n import tr

logger = logging.getLogger(__name__)

# Traces kept for display when "keep all run traces" is off -- the ensemble
# window draws at most this many faint member runs anyway.
DISPLAY_SAMPLE_RUNS = 30


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

        self.setWindowTitle(tr("Monte-Carlo Ensemble"))
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
        self.sim_time_spin.setSuffix(" s")
        self.sim_time_spin.setValue(default_sim_time)

        # Fixed step size.
        self.sim_dt_spin = QDoubleSpinBox()
        self.sim_dt_spin.setRange(1e-9, 1_000_000.0)
        self.sim_dt_spin.setDecimals(6)
        self.sim_dt_spin.setSingleStep(0.001)
        self.sim_dt_spin.setSuffix(" s")
        self.sim_dt_spin.setValue(default_sim_dt)

        # Parallel worker processes (1 = in-process, the historical behaviour).
        self.workers_spin = make_workers_spin()

        # Keeping every trace gives exact percentile bands but costs
        # n_runs x samples memory; off keeps a display sample only.
        self.keep_traces_check = QCheckBox(tr("Keep all run traces (exact percentile bands)"))
        self.keep_traces_check.setChecked(True)
        self.keep_traces_check.setToolTip(
            tr(
                "Turn off for very large ensembles: mean, std, min/max and the "
                "histograms stay exact, the band becomes mean ± std and only a "
                "sample of runs is drawn."
            )
        )

        self._setup_ui()

    # ------------------------------------------------------------------ UI ---
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        form = QFormLayout()
        form.addRow(tr("Number of runs:"), self.n_runs_spin)
        form.addRow(tr("Master seed:"), self.master_seed_spin)
        form.addRow(tr("Simulation time:"), self.sim_time_spin)
        form.addRow(tr("Step size (dt):"), self.sim_dt_spin)
        form.addRow(tr("Parallel workers:"), self.workers_spin)
        form.addRow("", self.keep_traces_check)
        layout.addLayout(form)

        helper = QLabel(
            tr(
                "Each run derives its own per-run seed from the master seed, the run "
                "index, and the block name. Every run differs, yet the whole ensemble "
                "is reproducible from the master seed alone -- rerun with the same "
                "master seed to reproduce the exact ensemble."
            )
        )
        helper.setWordWrap(True)
        layout.addWidget(helper)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
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
                "workers":     int, worker processes (1 = in-process)
                "keep_traces": True (all traces) or an int (display-sample size)
        """
        return {
            "n_runs": int(self.n_runs_spin.value()),
            "master_seed": int(self.master_seed_spin.value()),
            "sim_time": float(self.sim_time_spin.value()),
            "sim_dt": float(self.sim_dt_spin.value()),
            "workers": int(self.workers_spin.value()),
            "keep_traces": True if self.keep_traces_check.isChecked() else DISPLAY_SAMPLE_RUNS,
        }


def make_workers_spin():
    """Spin box for the experiment worker-process count (shared with the sweep dialog)."""
    spin = QSpinBox()
    spin.setRange(1, max(1, os.cpu_count() or 1))
    spin.setValue(default_workers())
    spin.setToolTip(
        tr(
            "Simulations run in this many separate processes. Results are identical "
            "for any value; each extra process takes about a second to start, so 1 "
            "is fastest for small experiments."
        )
    )
    return spin
