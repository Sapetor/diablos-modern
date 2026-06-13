"""
Linearize & Analyze - input/output selection dialog.

Presents the user with two multi-select lists built from the current diagram:

  * INPUTS  -- source-like blocks (no input ports, or b_type == source) that can
               act as system inputs (Step, Constant, Sine, Ramp, Chirp, ...).
  * OUTPUTS -- any block that produces a signal (has output ports) that can act
               as a measured system output.

Plus a "Find operating point (trim) first" checkbox. The dialog itself performs
no analysis; it only collects a selection. ``get_selection()`` returns the
chosen block ``name`` identifiers (the internal names, matching what the
Linearizer / AnalysisController expect):

    {"input_blocks": [str], "output_blocks": [str], "find_trim": bool}

Both lists may be left empty -- an empty input/output selection yields an
A-only linearization downstream.
"""

import logging

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QCheckBox, QDialogButtonBox, QGroupBox,
)
from PyQt5.QtCore import Qt

logger = logging.getLogger(__name__)

# b_type values used across DiaBloS: 0 = source, 1 = state, 2 = algebraic,
# 3 = sink. Sources are the natural "input" candidates for linearization.
_SOURCE_B_TYPE = 0


class LinearizeDialog(QDialog):
    """Select system inputs/outputs (and trim option) for linearization."""

    def __init__(self, dsim, parent=None):
        """
        Args:
            dsim: DSim instance whose ``blocks_list`` is used to populate the
                  candidate lists.
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self.dsim = dsim

        self.setWindowTitle("Linearize & Analyze")
        self.setMinimumWidth(460)
        self.setModal(True)

        self.input_list = QListWidget()
        self.output_list = QListWidget()
        self.trim_checkbox = QCheckBox("Find operating point (trim) first")

        self._setup_ui()
        self._populate()

    # ------------------------------------------------------------------ UI ---
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        intro = QLabel(
            "Choose which source blocks act as system inputs and which blocks "
            "are measured outputs. Leave both empty for an A-only "
            "(eigenvalue/stability) analysis."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        lists_layout = QHBoxLayout()

        # Inputs
        in_group = QGroupBox("Inputs (sources)")
        in_v = QVBoxLayout(in_group)
        self.input_list.setSelectionMode(QListWidget.ExtendedSelection)
        in_v.addWidget(self.input_list)
        lists_layout.addWidget(in_group)

        # Outputs
        out_group = QGroupBox("Outputs (signals)")
        out_v = QVBoxLayout(out_group)
        self.output_list.setSelectionMode(QListWidget.ExtendedSelection)
        out_v.addWidget(self.output_list)
        lists_layout.addWidget(out_group)

        layout.addLayout(lists_layout)

        layout.addWidget(self.trim_checkbox)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    # ------------------------------------------------------------- populate ---
    def _populate(self):
        """Fill the input/output lists from ``dsim.blocks_list``."""
        blocks = list(getattr(self.dsim, "blocks_list", None) or [])

        for block in blocks:
            if self._is_input_candidate(block):
                self.input_list.addItem(self._make_item(block))
            if self._is_output_candidate(block):
                self.output_list.addItem(self._make_item(block))

    @staticmethod
    def _make_item(block):
        """Build a list item: friendly label as text, block.name as data."""
        name = getattr(block, "name", "")
        item = QListWidgetItem(LinearizeDialog._friendly_label(block))
        item.setData(Qt.UserRole, name)
        return item

    @staticmethod
    def _friendly_label(block):
        """Human-readable label: 'username (block_fn)' falling back to name."""
        block_fn = getattr(block, "block_fn", "") or ""
        username = getattr(block, "username", "") or ""
        name = getattr(block, "name", "") or ""
        display = username or name or block_fn
        if block_fn and block_fn != display:
            return f"{display} ({block_fn})"
        return display

    @staticmethod
    def _is_input_candidate(block):
        """Source-like: declared b_type source, or no input ports."""
        b_type = getattr(block, "b_type", None)
        if b_type == _SOURCE_B_TYPE:
            return True
        in_ports = getattr(block, "in_ports", None)
        if in_ports is not None:
            try:
                return int(in_ports) == 0
            except (TypeError, ValueError):
                return False
        return False

    @staticmethod
    def _is_output_candidate(block):
        """Anything that emits a signal (has at least one output port)."""
        out_ports = getattr(block, "out_ports", None)
        if out_ports is None:
            return False
        try:
            return int(out_ports) > 0
        except (TypeError, ValueError):
            return False

    # ---------------------------------------------------------------- query ---
    def get_selection(self) -> dict:
        """
        Return the user's selection.

        Returns:
            dict with keys:
                "input_blocks":  [block.name, ...] selected as inputs
                "output_blocks": [block.name, ...] selected as outputs
                "find_trim":     bool, whether to solve for a trim point first
        """
        return {
            "input_blocks": self._selected_names(self.input_list),
            "output_blocks": self._selected_names(self.output_list),
            "find_trim": bool(self.trim_checkbox.isChecked()),
        }

    @staticmethod
    def _selected_names(list_widget) -> list:
        names = []
        for item in list_widget.selectedItems():
            name = item.data(Qt.UserRole)
            if name:
                names.append(name)
        return names
