"""
Property Editor Widget for DiaBloS

Upgraded with 7 UI improvements:
1. Generic choices support (any param with 'choices' metadata)
2. Per-parameter doc tooltips
3. Block identity header (type, category, ports)
4. Collapsible parameter groups
5. Slider + spinbox for bounded floats
6. Reset-to-default buttons per parameter
7. Inline validation messages
"""

import logging
import ast
import math
from collections import OrderedDict
from PyQt5.QtWidgets import (
    QLabel, QLineEdit, QComboBox, QCheckBox,
    QFrame, QFormLayout, QSpinBox, QDoubleSpinBox,
    QVBoxLayout, QHBoxLayout, QToolButton, QWidget,
    QSlider, QPushButton, QSizePolicy, QApplication
)
from PyQt5.QtCore import pyqtSignal, Qt, QSize
from PyQt5.QtGui import QColor
from modern_ui.themes.theme_manager import theme_manager
from lib.workspace import WorkspaceManager

logger = logging.getLogger(__name__)

# Param grouping constants
ADVANCED_PARAM_NAMES = frozenset({
    'sampling_time', 'method', 'verify_mode'
})
LIMIT_PARAM_NAMES = frozenset({
    'min', 'max', 'u_min', 'u_max', 'limit_output',
    'rising_slew', 'falling_slew'
})
NO_SLIDER_PARAMS = frozenset({
    'init_conds', 'init_temp', 'init_amplitude', 'N', 'Nx', 'Ny'
})


class CollapsibleSection(QWidget):
    """A collapsible section with toggle arrow and content area."""

    def __init__(self, title, parent=None, expanded=True):
        super().__init__(parent)
        main = QVBoxLayout(self)
        main.setContentsMargins(0, 2, 0, 2)
        main.setSpacing(0)

        self.toggle_btn = QToolButton()
        self.toggle_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_btn.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.toggle_btn.setText(f"  {title}")
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.setChecked(expanded)
        self.toggle_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.toggle_btn.setFixedHeight(28)
        self.toggle_btn.clicked.connect(self._toggle)
        main.addWidget(self.toggle_btn)

        self.content = QWidget()
        self.content_layout = QFormLayout(self.content)
        self.content_layout.setContentsMargins(4, 2, 2, 2)
        self.content_layout.setSpacing(4)
        self.content_layout.setVerticalSpacing(6)
        self.content_layout.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.content_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        self.content_layout.setLabelAlignment(Qt.AlignLeft)
        self.content.setVisible(expanded)
        main.addWidget(self.content)

    def _toggle(self, checked):
        self.toggle_btn.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self.content.setVisible(checked)

    def addRow(self, label, widget):
        self.content_layout.addRow(label, widget)

    def update_theme(self, colors):
        self.toggle_btn.setStyleSheet(f"""
            QToolButton {{
                background-color: {colors['surface_variant']};
                color: {colors['text_primary']};
                border: none;
                border-radius: 4px;
                font-weight: bold;
                font-size: 12px;
                text-align: left;
                padding-left: 4px;
            }}
            QToolButton:hover {{
                background-color: {colors['border']};
            }}
        """)


class SliderSpinBox(QWidget):
    """Composite widget: horizontal slider paired with a QDoubleSpinBox."""

    editingFinished = pyqtSignal()

    def __init__(self, value=0.0, min_val=0.0, max_val=10.0, decimals=4, parent=None):
        super().__init__(parent)
        self._min = min_val
        self._max = max_val
        self._steps = 1000

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, self._steps)
        self.slider.setValue(self._float_to_slider(value))
        self.slider.setMinimumWidth(40)
        layout.addWidget(self.slider, stretch=3)

        self.spinbox = QDoubleSpinBox()
        self.spinbox.setRange(-1e15, 1e15)
        self.spinbox.setDecimals(decimals)
        self.spinbox.setValue(value)
        self.spinbox.setFixedWidth(65)
        layout.addWidget(self.spinbox, stretch=0)

        self.slider.valueChanged.connect(self._on_slider_moved)
        self.slider.sliderReleased.connect(self._on_slider_released)
        self.spinbox.editingFinished.connect(self._on_spinbox_finished)

    def _float_to_slider(self, val):
        if self._max == self._min:
            return 0
        ratio = max(0.0, min(1.0, (val - self._min) / (self._max - self._min)))
        return int(ratio * self._steps)

    def _slider_to_float(self, pos):
        return self._min + (pos / self._steps) * (self._max - self._min)

    def _on_slider_moved(self, pos):
        val = self._slider_to_float(pos)
        self.spinbox.blockSignals(True)
        self.spinbox.setValue(val)
        self.spinbox.blockSignals(False)

    def _on_slider_released(self):
        self.editingFinished.emit()

    def _on_spinbox_finished(self):
        val = self.spinbox.value()
        clamped = max(self._min, min(self._max, val))
        self.slider.blockSignals(True)
        self.slider.setValue(self._float_to_slider(clamped))
        self.slider.blockSignals(False)
        self.editingFinished.emit()

    def value(self):
        return self.spinbox.value()

    def setValue(self, val):
        self.spinbox.blockSignals(True)
        self.spinbox.setValue(val)
        self.spinbox.blockSignals(False)
        self.slider.blockSignals(True)
        self.slider.setValue(self._float_to_slider(val))
        self.slider.blockSignals(False)

    def setToolTip(self, tip):
        super().setToolTip(tip)
        self.spinbox.setToolTip(tip)

    def toolTip(self):
        return self.spinbox.toolTip()

    def update_theme(self, colors):
        self.spinbox.setStyleSheet(f"""
            QDoubleSpinBox {{
                background-color: {colors['input_bg']};
                color: {colors['text']};
                border: 1px solid {colors['border']};
                border-radius: 4px;
                padding: 2px 4px;
                selection-background-color: {colors['accent']};
                selection-color: {colors['input_bg']};
            }}
            QDoubleSpinBox:focus {{
                border-color: {colors['accent']};
            }}
        """)
        self.slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                background: {colors['border']};
                height: 4px;
                border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: {colors['accent']};
                width: 14px;
                height: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }}
            QSlider::sub-page:horizontal {{
                background: {colors['accent']};
                border-radius: 2px;
            }}
        """)


class PropertyEditor(QFrame):
    """A widget that dynamically creates a form to edit block properties."""

    property_changed = pyqtSignal(str, str, object)
    pin_to_tuning = pyqtSignal(object, str)  # (block, param_name)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("PropertyEditor")
        self.setFrameStyle(QFrame.StyledPanel)
        self.setAutoFillBackground(True)

        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(4, 4, 4, 4)
        self._main_layout.setSpacing(6)
        self._main_layout.setAlignment(Qt.AlignTop)

        self.logger = logging.getLogger(__name__)
        self.block = None
        self._widgets = {}   # key -> (editor, reset_btn, val_label)
        self._defaults = {}  # key -> default value
        self._sections = []  # CollapsibleSection references

        theme_manager.theme_changed.connect(self._update_theme)
        self._update_theme()

    # ── Public API ──────────────────────────────────────────────

    def set_block(self, block):
        """Set the block to be edited and create the property form."""
        self.block = block
        self._clear_form()
        if self.block is None:
            self._show_placeholder()
        else:
            self._create_form()
        self.updateGeometry()

    # ── Form lifecycle ──────────────────────────────────────────

    def _clear_form(self):
        while self._main_layout.count():
            child = self._main_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
            elif child.layout():
                self._clear_sub_layout(child.layout())
        self._widgets.clear()
        self._defaults.clear()
        self._sections.clear()

    def _clear_sub_layout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
            elif child.layout():
                self._clear_sub_layout(child.layout())

    def _show_placeholder(self):
        placeholder = QLabel("Select a block to view its properties.")
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setWordWrap(True)
        text_color = theme_manager.get_color('text_secondary').name()
        placeholder.setStyleSheet(f"color: {text_color}; padding: 20px;")
        self._main_layout.addWidget(placeholder)

    # ── Form creation ───────────────────────────────────────────

    def _create_form(self):
        if not hasattr(self.block, 'params'):
            return

        self._create_block_header()
        self._create_name_field()
        self._create_port_count_field()

        keys = [k for k in self.block.params.keys() if not k.startswith('_')]
        groups = self._categorize_params(keys)

        for group_name, group_keys in groups.items():
            if not group_keys:
                continue
            expanded = (group_name == 'Parameters')
            section = CollapsibleSection(group_name, expanded=expanded)
            self._sections.append(section)
            for key in group_keys:
                self._add_param_row(section, key)
            self._main_layout.addWidget(section)

        self._create_doc_section()
        self._main_layout.addStretch(1)
        self._update_theme()

    def _create_block_header(self):
        """Block identity: type name, category badge, port count."""
        header = QWidget()
        h_layout = QVBoxLayout(header)
        h_layout.setContentsMargins(0, 0, 0, 4)
        h_layout.setSpacing(4)

        type_name = getattr(self.block, 'block_fn', None) or getattr(self.block, 'name', 'Block')
        text_color = theme_manager.get_color('text_primary').name()
        type_label = QLabel(type_name)
        type_label.setStyleSheet(f"color: {text_color}; font-weight: bold; font-size: 14px;")
        h_layout.addWidget(type_label)

        info_row = QHBoxLayout()
        info_row.setSpacing(8)

        category = getattr(self.block, 'category', None)
        if category:
            badge = QLabel(category)
            block_color = getattr(self.block, 'color', 'gray')
            bg_hex = self._color_name_to_hex(block_color, alpha=0.25)
            fg_hex = self._color_name_to_hex(block_color)
            badge.setStyleSheet(f"""
                background-color: {bg_hex};
                color: {fg_hex};
                border-radius: 4px;
                padding: 2px 8px;
                font-size: 11px;
                font-weight: bold;
            """)
            info_row.addWidget(badge)

        block_inst = getattr(self.block, 'block_instance', None)
        if block_inst:
            n_in = len(getattr(block_inst, 'inputs', []))
            n_out = len(getattr(block_inst, 'outputs', []))
            sec_color = theme_manager.get_color('text_secondary').name()
            port_label = QLabel(f"{n_in} in \u2192 {n_out} out")
            port_label.setStyleSheet(f"color: {sec_color}; font-size: 11px;")
            info_row.addWidget(port_label)

        info_row.addStretch(1)
        h_layout.addLayout(info_row)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet(f"background-color: {theme_manager.get_color('border_primary').name()};")
        h_layout.addWidget(line)

        self._main_layout.addWidget(header)

    def _create_name_field(self):
        text_color = theme_manager.get_color('text_primary').name()
        row = QHBoxLayout()
        row.setSpacing(6)

        name_label = QLabel("Name:")
        name_label.setStyleSheet(f"color: {text_color}; font-weight: bold;")
        row.addWidget(name_label)

        name_edit = QLineEdit(self.block.username)
        self._apply_widget_sizing(name_edit)
        name_edit.setPlaceholderText("Custom display name")
        name_edit.setToolTip("Set a custom display name (leave empty or '--' to reset)")
        name_edit.editingFinished.connect(lambda: self._on_name_changed(name_edit))
        row.addWidget(name_edit, stretch=1)

        self._main_layout.addLayout(row)

    def _create_port_count_field(self):
        """Add an Inputs / Outputs spinner for blocks that support variable port counts."""
        io_edit = getattr(self.block, 'io_edit', 'none')
        if io_edit in ('none', False, None):
            return

        text_color = theme_manager.get_color('text_primary').name()

        if io_edit in ('input', 'both'):
            row = QHBoxLayout()
            row.setSpacing(6)
            label = QLabel("Inputs:")
            label.setStyleSheet(f"color: {text_color}; font-weight: bold;")
            row.addWidget(label)

            sb = QSpinBox()
            sb.setRange(1, 20)
            sb.setValue(self.block.in_ports)
            self._apply_widget_sizing(sb)
            sb.editingFinished.connect(
                lambda: self._on_port_count_changed('_inputs_', sb.value())
            )
            row.addWidget(sb, stretch=1)
            self._main_layout.addLayout(row)

        if io_edit in ('output', 'both'):
            row = QHBoxLayout()
            row.setSpacing(6)
            label = QLabel("Outputs:")
            label.setStyleSheet(f"color: {text_color}; font-weight: bold;")
            row.addWidget(label)

            sb = QSpinBox()
            sb.setRange(1, 20)
            sb.setValue(self.block.out_ports)
            self._apply_widget_sizing(sb)
            sb.editingFinished.connect(
                lambda: self._on_port_count_changed('_outputs_', sb.value())
            )
            row.addWidget(sb, stretch=1)
            self._main_layout.addLayout(row)

    def _on_port_count_changed(self, prop_name, new_value):
        """Handle port count changes from the property editor."""
        if self.block is None:
            return
        block_name = getattr(self.block, 'name', 'Unknown')
        self.property_changed.emit(block_name, prop_name, new_value)

    # ── Parameter categorization (#4) ───────────────────────────

    def _categorize_params(self, keys):
        groups = OrderedDict()
        groups['Parameters'] = []
        for key in keys:
            meta = self._get_param_metadata(key)
            group = self._get_param_group(key, meta)
            groups.setdefault(group, []).append(key)
        return OrderedDict((k, v) for k, v in groups.items() if v)

    def _get_param_group(self, key, meta):
        if 'group' in meta:
            return meta['group']
        if key in ADVANCED_PARAM_NAMES or meta.get('advanced', False):
            return 'Advanced'
        if key in LIMIT_PARAM_NAMES or key.endswith('_min') or key.endswith('_max'):
            return 'Limits'
        return 'Parameters'

    # ── Parameter row ───────────────────────────────────────────

    def _add_param_row(self, section, key):
        value = self.block.params[key]
        meta = self._get_param_metadata(key)
        group = self._get_param_group(key, meta)

        text_color = theme_manager.get_color('text_primary').name()
        label = QLabel(f"{key.replace('_', ' ').title()}:")
        label.setStyleSheet(f"color: {text_color}; font-weight: bold;")

        # Per-param tooltip (#2)
        doc = meta.get('doc', '')
        if doc:
            label.setToolTip(doc)

        editor = self._create_editor_for_value(key, value, meta, group)
        if doc and not editor.toolTip():
            editor.setToolTip(doc)

        # Default for reset (#6)
        default = meta.get('default', value)
        self._defaults[key] = default

        reset_btn = QPushButton("\u21ba")
        reset_btn.setFixedSize(24, 24)
        reset_btn.setToolTip(f"Reset to default: {default}")
        reset_btn.setCursor(Qt.PointingHandCursor)
        reset_btn.clicked.connect(lambda checked, k=key: self._reset_param(k))
        reset_btn.setVisible(self._value_differs(value, default))

        # Validation label (#7)
        error_color = theme_manager.get_color('error').name()
        val_label = QLabel("")
        val_label.setWordWrap(True)
        val_label.setStyleSheet(f"color: {error_color}; font-size: 11px; padding-left: 2px;")
        val_label.hide()

        # Pin-to-tuning button for float params with sliders
        pin_btn = None
        if isinstance(editor, SliderSpinBox):
            pin_btn = QPushButton("\u25C9")  # ◉ pin icon
            pin_btn.setFixedSize(24, 24)
            pin_btn.setToolTip("Pin to Tuning Panel")
            pin_btn.setCursor(Qt.PointingHandCursor)
            pin_btn.clicked.connect(lambda checked, k=key: self._on_pin_to_tuning(k))

        self._widgets[key] = (editor, reset_btn, val_label)

        container = QWidget()
        c_layout = QVBoxLayout(container)
        c_layout.setContentsMargins(0, 0, 0, 0)
        c_layout.setSpacing(2)
        row = QHBoxLayout()
        row.setSpacing(4)
        row.addWidget(editor, stretch=1)
        if pin_btn:
            row.addWidget(pin_btn, stretch=0)
        row.addWidget(reset_btn, stretch=0)
        c_layout.addLayout(row)
        c_layout.addWidget(val_label)

        section.addRow(label, container)

    # ── Widget factory ──────────────────────────────────────────

    def _create_editor_for_value(self, key, value, meta, group):
        accepts_array = meta.get('accepts_array', False)
        choices = meta.get('choices', None)

        # Boolean → QCheckBox
        if isinstance(value, bool):
            cx = QCheckBox()
            cx.setChecked(value)
            cx.toggled.connect(lambda state, k=key: self._on_property_changed(k, state))
            return cx

        # Any param with choices → QComboBox (#1)
        if choices:
            cb = QComboBox()
            cb.addItems([str(c) for c in choices])
            cb.setCurrentText(str(value))
            cb.currentTextChanged.connect(
                lambda text, k=key: self._on_property_changed(k, text)
            )
            self._apply_widget_sizing(cb)
            return cb

        # Integer → QSpinBox
        if isinstance(value, int):
            sb = QSpinBox()
            sb.setRange(-999999, 999999)
            sb.setValue(value)
            sb.editingFinished.connect(lambda: self._on_property_changed(key, sb.value()))
            self._apply_widget_sizing(sb)
            return sb

        # Float
        if isinstance(value, float):
            if accepts_array:
                le = QLineEdit(str(value))
                self._apply_widget_sizing(le)
                le.setPlaceholderText("e.g. 1.0 or [1, 2, 3]")
                le.editingFinished.connect(
                    lambda: self._validate_and_submit_numeric(key, le)
                )
                return le

            # Slider + spinbox (#5)
            if self._should_show_slider(key, value, meta, group):
                sr = self._get_slider_range(value, meta)
                ssb = SliderSpinBox(value, sr[0], sr[1])
                ssb.editingFinished.connect(
                    lambda: self._on_property_changed(key, ssb.value())
                )
                return ssb

            # Standard spinbox
            dsb = QDoubleSpinBox()
            dsb.setRange(-float('inf'), float('inf'))
            dsb.setDecimals(4)
            dsb.setValue(value)
            dsb.editingFinished.connect(lambda: self._on_property_changed(key, dsb.value()))
            self._apply_widget_sizing(dsb)
            return dsb

        # Lists, strings, other → QLineEdit
        le = QLineEdit(str(value))
        self._apply_widget_sizing(le)
        le.editingFinished.connect(
            lambda: self._validate_and_submit_string(key, le, type(value))
        )
        if isinstance(value, list):
            le.setPlaceholderText("e.g. [1, 2, 3]")
        return le

    # ── Slider logic (#5) ──────────────────────────────────────

    def _should_show_slider(self, key, value, meta, group):
        if 'range' in meta:
            return True
        if meta.get('no_slider', False) or key in NO_SLIDER_PARAMS:
            return False
        if group == 'Advanced':
            return False
        if not math.isfinite(value) or value <= 0:
            return False
        return True

    def _get_slider_range(self, value, meta):
        if 'range' in meta:
            return list(meta['range'])
        return [0.0, value * 10] if value > 0 else [0.0, 10.0]

    # ── Documentation section ──────────────────────────────────

    def _create_doc_section(self):
        if not (hasattr(self.block, 'doc') and self.block.doc):
            return
        accent_color = theme_manager.get_color('accent_primary').name()
        sec_color = theme_manager.get_color('text_secondary').name()

        section = CollapsibleSection("Documentation", expanded=False)
        self._sections.append(section)

        doc_label = QLabel(str(self.block.doc).strip())
        doc_label.setWordWrap(True)
        doc_label.setStyleSheet(
            f"color: {sec_color}; font-style: italic; margin-bottom: 4px;"
        )
        section.content_layout.addRow("", doc_label)

        base_url = "https://github.com/Sapetor/diablos-modern/blob/main/docs/wiki"
        cat_file = f"{getattr(self.block, 'category', 'Home')}.md".replace(' ', '-')
        anchor = getattr(self.block, 'block_fn', 'Home').lower().replace(' ', '-')
        full_url = f"{base_url}/{cat_file}#{anchor}"
        link_label = QLabel(f'<a href="{full_url}">View Full Reference</a>')
        link_label.setOpenExternalLinks(True)
        link_label.setStyleSheet(f"color: {accent_color};")
        section.content_layout.addRow("", link_label)

        self._main_layout.addWidget(section)

    # ── Metadata helpers ───────────────────────────────────────

    def _get_param_metadata(self, key):
        if self.block is None:
            return {}
        block_instance = getattr(self.block, 'block_instance', None)
        if block_instance is None:
            return {}
        if hasattr(block_instance, 'params'):
            block_params = block_instance.params
            if key in block_params and isinstance(block_params[key], dict):
                return block_params[key]
        return {}

    def _apply_widget_sizing(self, widget):
        """Set flexible sizing: widgets expand to fill available space but can shrink."""
        widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        widget.setMinimumWidth(50)

    def _color_name_to_hex(self, name, alpha=None):
        c = QColor(name)
        if not c.isValid():
            c = QColor('#888888')
        if alpha is not None:
            surface = theme_manager.get_color('surface')
            r = int(c.red() * alpha + surface.red() * (1 - alpha))
            g = int(c.green() * alpha + surface.green() * (1 - alpha))
            b = int(c.blue() * alpha + surface.blue() * (1 - alpha))
            return QColor(r, g, b).name()
        return c.name()

    # ── Validation (#7) ────────────────────────────────────────

    def _validate_and_submit_string(self, key, widget, target_type):
        text = widget.text()
        is_valid = True
        converted_value = None
        error_msg = ""

        try:
            if target_type == list:
                try:
                    val = ast.literal_eval(text)
                    if not isinstance(val, list):
                        raise ValueError("Not a list")
                    converted_value = val
                except (ValueError, SyntaxError):
                    ws = WorkspaceManager()
                    if text in ws.variables and isinstance(ws.variables[text], list):
                        converted_value = text
                    else:
                        raise ValueError("Expected a list, e.g. [1, 2, 3]")
            elif target_type == int:
                try:
                    converted_value = int(text)
                except ValueError:
                    ws = WorkspaceManager()
                    if text in ws.variables:
                        converted_value = text
                    else:
                        raise ValueError("Expected an integer or workspace variable")
            elif target_type == float:
                try:
                    converted_value = float(text)
                except ValueError:
                    ws = WorkspaceManager()
                    if text in ws.variables:
                        converted_value = text
                    else:
                        raise ValueError("Expected a number or workspace variable")
            else:
                converted_value = text
        except (ValueError, SyntaxError) as e:
            is_valid = False
            error_msg = str(e)

        if is_valid:
            self._set_validation_error(key, None)
            self._on_property_changed(key, converted_value)
        else:
            self._set_validation_error(key, error_msg)
            self.logger.warning(f"Invalid input for {key}: {text}")

    def _validate_and_submit_numeric(self, key, widget):
        text = widget.text().strip()
        is_valid = True
        converted_value = None
        error_msg = ""

        try:
            val = ast.literal_eval(text)
            if isinstance(val, (int, float)):
                converted_value = float(val)
            elif isinstance(val, (list, tuple)):
                converted_value = [float(x) for x in val]
            else:
                raise ValueError("Expected a number or list of numbers")
        except (ValueError, SyntaxError):
            ws = WorkspaceManager()
            if text in ws.variables:
                converted_value = text
            else:
                is_valid = False
                error_msg = "Expected a number, list, or workspace variable"

        if is_valid:
            self._set_validation_error(key, None)
            self._on_property_changed(key, converted_value)
        else:
            self._set_validation_error(key, error_msg)
            self.logger.warning(f"Invalid numeric input for {key}: {text}")

    def _set_validation_error(self, key, message):
        if key not in self._widgets:
            return
        editor, _, val_label = self._widgets[key]
        if message:
            val_label.setText(message)
            val_label.show()
            self._set_widget_error_border(editor, True)
        else:
            val_label.setText("")
            val_label.hide()
            self._set_widget_error_border(editor, False)

    def _set_widget_error_border(self, widget, is_error):
        border_color = (
            theme_manager.get_color('error').name() if is_error
            else theme_manager.get_color('border_primary').name()
        )
        bg = theme_manager.get_color('surface_variant').name()
        txt = theme_manager.get_color('text_primary').name()
        target = widget.spinbox if isinstance(widget, SliderSpinBox) else widget
        if isinstance(target, QLineEdit):
            target.setStyleSheet(f"""
                QLineEdit {{
                    background-color: {bg}; color: {txt};
                    border: 2px solid {border_color};
                    border-radius: 4px; padding: 2px 4px;
                }}
            """)

    # ── Pin to tuning ─────────────────────────────────────────

    def _on_pin_to_tuning(self, key):
        """Emit signal to pin this parameter to the tuning panel."""
        if self.block is not None:
            self.pin_to_tuning.emit(self.block, key)

    # ── Property changes ───────────────────────────────────────

    def _on_property_changed(self, prop_name, new_value):
        if self.block is None:
            return
        # Update reset button visibility (#6)
        if prop_name in self._widgets:
            _, reset_btn, _ = self._widgets[prop_name]
            default = self._defaults.get(prop_name)
            reset_btn.setVisible(self._value_differs(new_value, default))

        block_name = getattr(self.block, 'name', 'Unknown')
        self.logger.info(f"Property changed: {block_name}.{prop_name} = {new_value}")
        self.property_changed.emit(block_name, prop_name, new_value)

    def _on_name_changed(self, widget):
        if self.block is None:
            return
        new_name = widget.text().strip()
        if new_name == '--' or new_name == '':
            self.block.username = self.block.name
            widget.setText(self.block.name)
        else:
            self.block.username = new_name
        self.property_changed.emit(self.block.name, '_username_', self.block.username)

    # ── Reset to default (#6) ──────────────────────────────────

    def _reset_param(self, key):
        default = self._defaults.get(key)
        if default is None:
            return
        editor, _, _ = self._widgets[key]
        if isinstance(editor, QCheckBox):
            editor.setChecked(bool(default))
        elif isinstance(editor, QComboBox):
            editor.setCurrentText(str(default))
        elif isinstance(editor, QSpinBox):
            editor.setValue(int(default))
        elif isinstance(editor, SliderSpinBox):
            editor.setValue(float(default))
        elif isinstance(editor, QDoubleSpinBox):
            editor.setValue(float(default))
        elif isinstance(editor, QLineEdit):
            editor.setText(str(default))
        self._set_validation_error(key, None)
        self._on_property_changed(key, default)

    def _value_differs(self, value, default):
        if value is None or default is None:
            return value != default
        try:
            if isinstance(value, float) and isinstance(default, float):
                if math.isnan(value) and math.isnan(default):
                    return False
                if math.isinf(value) and math.isinf(default):
                    return (value > 0) != (default > 0)
                return abs(value - default) > 1e-10
            result = value != default
            # numpy arrays return array from !=; coerce to scalar bool
            if hasattr(result, '__len__'):
                return bool(any(result))
            return bool(result)
        except (TypeError, ValueError):
            return str(value) != str(default)

    # ── Theming ────────────────────────────────────────────────

    def _update_theme(self):
        bg = theme_manager.get_color('surface').name()
        txt = theme_manager.get_color('text_primary').name()
        border = theme_manager.get_color('border_primary').name()
        input_bg = theme_manager.get_color('surface_variant').name()
        accent = theme_manager.get_color('accent_primary').name()
        error = theme_manager.get_color('error').name()

        self.setStyleSheet(f"""
            PropertyEditor {{
                background-color: {bg};
                border: 1px solid {border};
                border-radius: 4px;
            }}
            QCheckBox {{ color: {txt}; spacing: 8px; }}
            QLabel {{ color: {txt}; }}
        """)

        colors = {
            'surface_variant': input_bg, 'text_primary': txt, 'text': txt,
            'border_primary': border, 'border': border,
            'accent': accent, 'input_bg': input_bg, 'error': error,
        }

        for section in self._sections:
            section.update_theme(colors)

        for key, (editor, reset_btn, val_label) in self._widgets.items():
            reset_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: transparent;
                    color: {txt}; border: 1px solid {border};
                    border-radius: 4px; font-size: 14px;
                }}
                QPushButton:hover {{
                    background-color: {input_bg};
                    border-color: {accent};
                }}
            """)
            val_label.setStyleSheet(
                f"color: {error}; font-size: 11px; padding-left: 2px;"
            )
            if isinstance(editor, SliderSpinBox):
                editor.update_theme(colors)
            elif isinstance(editor, (QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox)):
                widget_name = type(editor).__name__
                editor.setStyleSheet(
                    f"{widget_name} {{ "
                    f"background-color: {input_bg}; color: {txt}; "
                    f"border: 1px solid {border}; border-radius: 4px; "
                    f"padding: 2px 4px; "
                    f"selection-background-color: {accent}; "
                    f"selection-color: {input_bg}; }}"
                    f"{widget_name}:focus {{ "
                    f"border-color: {accent}; }}"
                )

    def sizeHint(self):
        if self.block and hasattr(self.block, 'params'):
            num_props = len([p for p in self.block.params if not p.startswith('_')])
            if num_props > 0:
                row_height = 50
                height = (
                    num_props * row_height
                    + self._main_layout.contentsMargins().top()
                    + self._main_layout.contentsMargins().bottom()
                    + 120
                )
                return QSize(super().sizeHint().width(), height)
        return QSize(super().sizeHint().width(), 100)
