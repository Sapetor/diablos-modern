"""
Tuning Panel - Interactive parameter tuning for live re-simulation.

Provides a Mathematica Manipulate-style experience: drag sliders to change
block parameters and watch scope plots update in real-time.
"""

import logging
import math
import re
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QToolButton, QFrame, QScrollArea, QSlider,
    QMenu, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
    QLineEdit
)
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QFont
from modern_ui.themes.theme_manager import theme_manager

logger = logging.getLogger(__name__)


class TuningParameterRow(QFrame):
    """Compact single-line parameter row: name | slider | value | buttons."""

    value_changed = pyqtSignal(str, str, float)  # block_name, param_name, value
    removed = pyqtSignal(str, str)  # block_name, param_name

    def __init__(self, block_name, param_name, value, parent=None):
        super().__init__(parent)
        self.block_name = block_name
        self.param_name = param_name
        self._initial_value = value
        self._min = 0.0
        self._max = 10.0
        self._steps = 2000
        self._suppress_signals = False

        self.setFrameStyle(QFrame.NoFrame)
        self.setFixedHeight(28)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

        # Compute initial range
        self._min, self._max = self._get_slider_range(value)

        row = QHBoxLayout(self)
        row.setContentsMargins(4, 0, 4, 0)
        row.setSpacing(4)

        # Param label: "BlockName.param"
        label_text = f"{block_name}.{param_name}"
        if len(label_text) > 20:
            label_text = f"..{param_name}" if len(param_name) <= 16 else f"..{param_name[-14:]}"
        name_label = QLabel(label_text)
        name_label.setFixedWidth(120)
        name_label.setToolTip(f"{block_name}.{param_name}  |  Right-click for options")
        self._name_label = name_label
        row.addWidget(name_label)

        # Slider
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(0, self._steps)
        self._slider.setValue(self._val_to_slider(value))
        self._slider.setFixedHeight(18)
        self._slider.valueChanged.connect(self._on_slider_moved)
        row.addWidget(self._slider, stretch=1)

        # Value field
        self._value_edit = QLineEdit(self._fmt_value(value))
        self._value_edit.setFixedWidth(58)
        self._value_edit.setFixedHeight(20)
        self._value_edit.setAlignment(Qt.AlignRight)
        self._value_edit.setFrame(False)
        font = QFont()
        font.setPointSize(11)
        self._value_edit.setFont(font)
        self._value_edit.editingFinished.connect(self._on_value_typed)
        row.addWidget(self._value_edit)

        # Reset button
        reset_btn = QToolButton()
        reset_btn.setText("\u21ba")
        reset_btn.setFixedSize(18, 18)
        reset_btn.setToolTip(f"Reset to {self._fmt_value(value)}")
        reset_btn.setCursor(Qt.PointingHandCursor)
        reset_btn.clicked.connect(self._reset_value)
        self._reset_btn = reset_btn
        row.addWidget(reset_btn)

        # Remove button
        remove_btn = QToolButton()
        remove_btn.setText("\u2716")
        remove_btn.setFixedSize(18, 18)
        remove_btn.setToolTip("Remove from tuning")
        remove_btn.setCursor(Qt.PointingHandCursor)
        remove_btn.clicked.connect(self._on_remove)
        self._remove_btn = remove_btn
        row.addWidget(remove_btn)

    # ── Value formatting ──

    @staticmethod
    def _fmt_value(val):
        """Format current value for display."""
        if abs(val) >= 1000 or (abs(val) < 0.01 and val != 0):
            return f"{val:.4g}"
        if val == int(val):
            return str(int(val))
        return f"{val:.4f}".rstrip('0').rstrip('.')

    @staticmethod
    def _fmt_range(val):
        """Format range endpoint compactly."""
        if val == int(val):
            return str(int(val))
        return f"{val:.3g}"

    # ── Slider <-> value conversion ──

    def _val_to_slider(self, val):
        if self._max == self._min:
            return 0
        ratio = max(0.0, min(1.0, (val - self._min) / (self._max - self._min)))
        return int(ratio * self._steps)

    def _slider_to_val(self, pos):
        return self._min + (pos / self._steps) * (self._max - self._min)

    # ── Range logic ──

    def _get_slider_range(self, value):
        if not math.isfinite(value):
            return [0.0, 10.0]
        if value > 0:
            return [0.0, value * 5.0]
        elif value < 0:
            return [value * 5.0, 0.0]
        return [-10.0, 10.0]

    def set_range(self, min_val, max_val):
        """Update slider range."""
        if min_val >= max_val:
            return
        self._min = min_val
        self._max = max_val
        current = self._slider_to_val(self._slider.value())
        self._slider.blockSignals(True)
        self._slider.setValue(self._val_to_slider(current))
        self._slider.blockSignals(False)

    # ── Event handlers ──

    def _on_slider_moved(self, pos):
        if self._suppress_signals:
            return
        val = self._slider_to_val(pos)
        self._value_edit.setText(self._fmt_value(val))
        self.value_changed.emit(self.block_name, self.param_name, val)

    def _on_value_typed(self):
        """User typed a value in the edit field."""
        try:
            val = float(self._value_edit.text())
        except ValueError:
            return
        # Auto-expand range if typed value is outside
        if val < self._min:
            self.set_range(val * 1.5 if val < 0 else val * 0.5, self._max)
        elif val > self._max:
            self.set_range(self._min, val * 1.5 if val > 0 else val * 0.5)
        self._suppress_signals = True
        self._slider.setValue(self._val_to_slider(val))
        self._suppress_signals = False
        self.value_changed.emit(self.block_name, self.param_name, val)

    def _reset_value(self):
        val = self._initial_value
        self._value_edit.setText(self._fmt_value(val))
        self._suppress_signals = True
        self._slider.setValue(self._val_to_slider(val))
        self._suppress_signals = False
        self.value_changed.emit(self.block_name, self.param_name, val)

    def _on_remove(self):
        self.removed.emit(self.block_name, self.param_name)

    def value(self):
        return self._slider_to_val(self._slider.value())

    # ── Context menu ──

    def _show_context_menu(self, pos):
        menu = QMenu(self)
        range_action = menu.addAction("Set Range...")
        reset_action = menu.addAction(f"Reset to {self._fmt_value(self._initial_value)}")
        menu.addSeparator()
        remove_action = menu.addAction("Remove")
        action = menu.exec_(self.mapToGlobal(pos))
        if action == range_action:
            self._open_range_dialog()
        elif action == reset_action:
            self._reset_value()
        elif action == remove_action:
            self._on_remove()

    def _open_range_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Range: {self.param_name}")
        form = QFormLayout(dlg)

        min_spin = QDoubleSpinBox()
        min_spin.setRange(-1e15, 1e15)
        min_spin.setDecimals(4)
        min_spin.setValue(self._min)
        form.addRow("Min:", min_spin)

        max_spin = QDoubleSpinBox()
        max_spin.setRange(-1e15, 1e15)
        max_spin.setDecimals(4)
        max_spin.setValue(self._max)
        form.addRow("Max:", max_spin)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)

        if dlg.exec_() == QDialog.Accepted:
            self.set_range(min_spin.value(), max_spin.value())

    # ── Theming ──

    def update_theme(self, colors):
        accent = colors['accent']
        text = colors['text']
        border = colors['border']
        input_bg = colors['input_bg']

        self._name_label.setStyleSheet(f"color: {text}; font-size: 11px;")
        self._value_edit.setStyleSheet(f"""
            QLineEdit {{
                color: {accent};
                background: transparent;
                border: none;
                border-bottom: 1px solid {border};
                padding: 0px 2px;
            }}
            QLineEdit:focus {{
                border-bottom: 1px solid {accent};
            }}
        """)

        btn_style = f"""
            QToolButton {{
                background: transparent;
                color: {text};
                border: none;
                font-size: 11px;
            }}
            QToolButton:hover {{
                background: {input_bg};
                border-radius: 3px;
            }}
        """
        self._reset_btn.setStyleSheet(btn_style)
        self._remove_btn.setStyleSheet(btn_style)

        self._slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                background: {border};
                height: 4px;
                border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: {accent};
                width: 12px;
                height: 12px;
                margin: -4px 0;
                border-radius: 6px;
            }}
            QSlider::sub-page:horizontal {{
                background: {accent};
                border-radius: 2px;
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
        main_layout.setContentsMargins(4, 2, 4, 2)
        main_layout.setSpacing(2)

        # Header row
        header = QHBoxLayout()
        header.setSpacing(4)

        self._toggle_btn = QToolButton()
        self._toggle_btn.setArrowType(Qt.DownArrow)
        self._toggle_btn.setFixedSize(16, 16)
        self._toggle_btn.setCheckable(True)
        self._toggle_btn.setChecked(True)
        self._toggle_btn.clicked.connect(self._toggle_content)
        header.addWidget(self._toggle_btn)

        title = QLabel("Tuning")
        title.setStyleSheet("font-weight: bold; font-size: 11px;")
        self._title_label = title
        header.addWidget(title)

        header.addStretch(1)

        clear_btn = QPushButton("Clear")
        clear_btn.setFixedHeight(18)
        clear_btn.setCursor(Qt.PointingHandCursor)
        clear_btn.clicked.connect(self.clear_all)
        self._clear_btn = clear_btn
        header.addWidget(clear_btn)

        main_layout.addLayout(header)

        # Scrollable content area for parameter rows
        self._content = QScrollArea()
        self._content.setWidgetResizable(True)
        self._content.setFrameStyle(QFrame.NoFrame)
        self._content.setMaximumHeight(160)

        self._rows_widget = QWidget()
        self._rows_layout = QVBoxLayout(self._rows_widget)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(0)
        self._rows_layout.addStretch(1)

        self._content.setWidget(self._rows_widget)
        main_layout.addWidget(self._content)

        # Placeholder hint
        self._hint = QLabel("Right-click block > Add to Tuning")
        self._hint.setAlignment(Qt.AlignCenter)
        self._hint.setStyleSheet("color: gray; font-style: italic; padding: 4px; font-size: 11px;")
        main_layout.addWidget(self._hint)

        theme_manager.theme_changed.connect(self._update_theme)
        self._update_theme()

    def add_parameter(self, block, param_name):
        """Add a parameter slider for the given block and param."""
        block_name = getattr(block, 'name', str(block))
        key = (block_name, param_name)
        if key in self._rows:
            logger.debug(f"Parameter {key} already in tuning panel")
            return

        # Parse indexed list params: "denominator[1]" -> ("denominator", 1)
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

        self._title_label.setStyleSheet(f"color: {text}; font-weight: bold; font-size: 11px;")
        self._hint.setStyleSheet(f"color: {theme_manager.get_color('text_secondary').name()}; "
                                  "font-style: italic; padding: 4px; font-size: 11px;")
        self._clear_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent; color: {text};
                border: 1px solid {border}; border-radius: 3px;
                padding: 1px 6px; font-size: 10px;
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
