"""
Tuning Panel - Interactive parameter tuning for live re-simulation.

Provides a Mathematica Manipulate-style experience: drag sliders to change
block parameters and watch scope plots update in real-time.
"""

import logging
import math
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QToolButton, QFrame, QSizePolicy, QScrollArea,
    QMenu, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout
)
from PyQt5.QtCore import pyqtSignal, Qt
from modern_ui.widgets.property_editor import SliderSpinBox
from modern_ui.themes.theme_manager import theme_manager

logger = logging.getLogger(__name__)


class TuningParameterRow(QWidget):
    """A single parameter row: label + slider/spinbox + remove button."""

    value_changed = pyqtSignal(str, str, float)  # block_name, param_name, value
    removed = pyqtSignal(str, str)  # block_name, param_name

    def __init__(self, block_name, param_name, value, parent=None):
        super().__init__(parent)
        self.block_name = block_name
        self.param_name = param_name
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)

        # Label: "BlockName / param"
        self.label = QLabel(f"{block_name} / {param_name}")
        self.label.setFixedWidth(140)
        self.label.setToolTip(f"{block_name}.{param_name} (right-click to set range)")
        layout.addWidget(self.label)

        # Min label
        sr = self._get_slider_range(value)
        self._min_label = QLabel(self._fmt(sr[0]))
        self._min_label.setFixedWidth(36)
        self._min_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._min_label.setStyleSheet("font-size: 10px; color: gray;")
        layout.addWidget(self._min_label)

        # Slider + spinbox
        self.slider_spinbox = SliderSpinBox(value, sr[0], sr[1])
        self.slider_spinbox.slider.valueChanged.connect(self._on_slider_moved)
        self.slider_spinbox.spinbox.editingFinished.connect(self._on_spinbox_changed)
        layout.addWidget(self.slider_spinbox, stretch=1)

        # Max label
        self._max_label = QLabel(self._fmt(sr[1]))
        self._max_label.setFixedWidth(36)
        self._max_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._max_label.setStyleSheet("font-size: 10px; color: gray;")
        layout.addWidget(self._max_label)

        # Remove button
        remove_btn = QPushButton("\u2716")
        remove_btn.setFixedSize(22, 22)
        remove_btn.setToolTip("Remove from tuning")
        remove_btn.setCursor(Qt.PointingHandCursor)
        remove_btn.clicked.connect(self._on_remove)
        layout.addWidget(remove_btn)

        self._remove_btn = remove_btn

    @staticmethod
    def _fmt(val):
        """Format a range value compactly."""
        if val == int(val):
            return str(int(val))
        return f"{val:.2g}"

    def _get_slider_range(self, value):
        """Determine slider range from current value."""
        if not math.isfinite(value):
            return [0.0, 10.0]
        if value > 0:
            return [0.0, value * 10.0]
        elif value < 0:
            return [value * 10.0, 0.0]
        return [0.0, 10.0]

    def set_range(self, min_val, max_val):
        """Update the slider range and labels."""
        if min_val >= max_val:
            return
        current = self.slider_spinbox.value()
        self.slider_spinbox._min = min_val
        self.slider_spinbox._max = max_val
        self.slider_spinbox.slider.blockSignals(True)
        self.slider_spinbox.slider.setValue(self.slider_spinbox._float_to_slider(current))
        self.slider_spinbox.slider.blockSignals(False)
        self._min_label.setText(self._fmt(min_val))
        self._max_label.setText(self._fmt(max_val))

    def _show_context_menu(self, pos):
        """Right-click menu with Set Range option."""
        menu = QMenu(self)
        range_action = menu.addAction("Set Range...")
        remove_action = menu.addAction("Remove")
        action = menu.exec_(self.mapToGlobal(pos))
        if action == range_action:
            self._open_range_dialog()
        elif action == remove_action:
            self._on_remove()

    def _open_range_dialog(self):
        """Open dialog to set min/max range for this slider."""
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Range: {self.block_name} / {self.param_name}")
        form = QFormLayout(dlg)

        min_spin = QDoubleSpinBox()
        min_spin.setRange(-1e15, 1e15)
        min_spin.setDecimals(4)
        min_spin.setValue(self.slider_spinbox._min)
        form.addRow("Min:", min_spin)

        max_spin = QDoubleSpinBox()
        max_spin.setRange(-1e15, 1e15)
        max_spin.setDecimals(4)
        max_spin.setValue(self.slider_spinbox._max)
        form.addRow("Max:", max_spin)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)

        if dlg.exec_() == QDialog.Accepted:
            self.set_range(min_spin.value(), max_spin.value())

    def _on_slider_moved(self, pos):
        """Emit on every slider tick for continuous Manipulate feel."""
        val = self.slider_spinbox._slider_to_float(pos)
        self.slider_spinbox.spinbox.blockSignals(True)
        self.slider_spinbox.spinbox.setValue(val)
        self.slider_spinbox.spinbox.blockSignals(False)
        self.value_changed.emit(self.block_name, self.param_name, val)

    def _on_spinbox_changed(self):
        """Emit when spinbox value is committed."""
        val = self.slider_spinbox.value()
        self.value_changed.emit(self.block_name, self.param_name, val)

    def _on_remove(self):
        self.removed.emit(self.block_name, self.param_name)

    def set_value_silent(self, value):
        """Set value without emitting signals."""
        self.slider_spinbox.setValue(value)

    def update_theme(self, colors):
        self.label.setStyleSheet(f"color: {colors['text']}; font-size: 11px;")
        self.slider_spinbox.update_theme(colors)
        self._remove_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {colors['text']};
                border: 1px solid {colors['border']};
                border-radius: 4px; font-size: 12px;
            }}
            QPushButton:hover {{
                background-color: {colors['input_bg']};
                border-color: {colors['accent']};
            }}
        """)


class TuningPanel(QFrame):
    """
    Collapsible panel for interactive parameter tuning.

    Shows slider rows for user-selected parameters. Each slider change
    emits param_changed which triggers a headless re-simulation.
    """

    param_changed = pyqtSignal(str, str, float)  # block_name, param_name, value
    param_removed = pyqtSignal(str, str)  # block_name, param_name
    panel_cleared = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("TuningPanel")
        self.setFrameStyle(QFrame.StyledPanel)
        self._rows = {}  # key: (block_name, param_name) -> TuningParameterRow

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(6, 4, 6, 4)
        main_layout.setSpacing(4)

        # Header row
        header = QHBoxLayout()
        header.setSpacing(6)

        self._toggle_btn = QToolButton()
        self._toggle_btn.setArrowType(Qt.DownArrow)
        self._toggle_btn.setCheckable(True)
        self._toggle_btn.setChecked(True)
        self._toggle_btn.clicked.connect(self._toggle_content)
        header.addWidget(self._toggle_btn)

        title = QLabel("Parameter Tuning")
        title.setStyleSheet("font-weight: bold; font-size: 12px;")
        self._title_label = title
        header.addWidget(title)

        header.addStretch(1)

        clear_btn = QPushButton("Clear All")
        clear_btn.setFixedHeight(22)
        clear_btn.setCursor(Qt.PointingHandCursor)
        clear_btn.clicked.connect(self.clear_all)
        self._clear_btn = clear_btn
        header.addWidget(clear_btn)

        main_layout.addLayout(header)

        # Scrollable content area for parameter rows
        self._content = QScrollArea()
        self._content.setWidgetResizable(True)
        self._content.setFrameStyle(QFrame.NoFrame)
        self._content.setMaximumHeight(180)

        self._rows_widget = QWidget()
        self._rows_layout = QVBoxLayout(self._rows_widget)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(2)
        self._rows_layout.addStretch(1)

        self._content.setWidget(self._rows_widget)
        main_layout.addWidget(self._content)

        # Placeholder hint
        self._hint = QLabel("Right-click a block > \"Add to Tuning\" to pin parameters here")
        self._hint.setAlignment(Qt.AlignCenter)
        self._hint.setStyleSheet("color: gray; font-style: italic; padding: 8px;")
        main_layout.addWidget(self._hint)

        theme_manager.theme_changed.connect(self._update_theme)
        self._update_theme()

    def add_parameter(self, block, param_name):
        """Add a parameter slider for the given block and param.

        Supports indexed list params like 'denominator[1]' which tune
        individual elements of list-valued block parameters.
        """
        block_name = getattr(block, 'name', str(block))
        key = (block_name, param_name)
        if key in self._rows:
            logger.debug(f"Parameter {key} already in tuning panel")
            return

        # Parse indexed list params: "denominator[1]" -> ("denominator", 1)
        import re
        match = re.match(r'^(.+)\[(\d+)\]$', param_name)
        if match:
            base_name, idx = match.group(1), int(match.group(2))
            base_val = block.params.get(base_name)
            if isinstance(base_val, (list, tuple)) and idx < len(base_val):
                value = float(base_val[idx])
            else:
                logger.warning(f"Cannot tune {param_name}: invalid list index")
                return
        else:
            value = block.params.get(param_name, 0.0)
            if not isinstance(value, (int, float)):
                logger.warning(f"Cannot tune non-numeric param {param_name}={value}")
                return
            value = float(value)

        row = TuningParameterRow(block_name, param_name, value)
        row.value_changed.connect(self._on_row_value_changed)
        row.removed.connect(self.remove_parameter)
        self._rows[key] = row

        # Insert before the stretch at the end
        count = self._rows_layout.count()
        self._rows_layout.insertWidget(count - 1, row)

        self._hint.hide()
        self._update_theme()
        self.show()
        logger.info(f"Added tuning parameter: {block_name}/{param_name} = {value}")

    def remove_parameter(self, block_name, param_name):
        """Remove a parameter slider."""
        key = (block_name, param_name)
        row = self._rows.pop(key, None)
        if row:
            self._rows_layout.removeWidget(row)
            row.deleteLater()
            self.param_removed.emit(block_name, param_name)
            logger.info(f"Removed tuning parameter: {block_name}/{param_name}")

        if not self._rows:
            self._hint.show()
            self.hide()

    def remove_block_parameters(self, block_name):
        """Remove all parameters for a given block."""
        keys_to_remove = [k for k in self._rows if k[0] == block_name]
        for key in keys_to_remove:
            self.remove_parameter(*key)

    def clear_all(self):
        """Remove all parameter sliders."""
        for key in list(self._rows.keys()):
            row = self._rows.pop(key)
            self._rows_layout.removeWidget(row)
            row.deleteLater()
        self._hint.show()
        self.panel_cleared.emit()
        self.hide()

    def has_parameters(self):
        return len(self._rows) > 0

    def _on_row_value_changed(self, block_name, param_name, value):
        self.param_changed.emit(block_name, param_name, value)

    def _toggle_content(self, checked):
        self._toggle_btn.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self._content.setVisible(checked)
        self._hint.setVisible(checked and not self._rows)

    def _update_theme(self):
        bg = theme_manager.get_color('surface').name()
        border = theme_manager.get_color('border_primary').name()
        text = theme_manager.get_color('text_primary').name()
        accent = theme_manager.get_color('accent_primary').name()
        input_bg = theme_manager.get_color('surface_variant').name()

        self.setStyleSheet(f"""
            #TuningPanel {{
                background-color: {bg};
                border: 1px solid {border};
                border-radius: 4px;
            }}
        """)

        self._title_label.setStyleSheet(f"color: {text}; font-weight: bold; font-size: 12px;")
        self._hint.setStyleSheet(f"color: {theme_manager.get_color('text_secondary').name()}; "
                                  "font-style: italic; padding: 8px;")
        self._clear_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent; color: {text};
                border: 1px solid {border}; border-radius: 4px;
                padding: 2px 8px; font-size: 11px;
            }}
            QPushButton:hover {{
                background-color: {input_bg}; border-color: {accent};
            }}
        """)
        self._toggle_btn.setStyleSheet(f"""
            QToolButton {{
                background-color: transparent; color: {text}; border: none;
            }}
        """)

        colors = {
            'text': text, 'border': border, 'accent': accent,
            'input_bg': input_bg, 'surface_variant': input_bg,
            'text_primary': text,
        }
        for row in self._rows.values():
            row.update_theme(colors)
