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
    QAbstractSpinBox, QLabel, QLineEdit, QComboBox, QCheckBox,
    QFrame, QFormLayout, QSpinBox, QDoubleSpinBox,
    QVBoxLayout, QHBoxLayout, QToolButton, QWidget,
    QSlider, QPushButton, QSizePolicy, QApplication
)
from PyQt5.QtCore import pyqtSignal, Qt, QSize, QEvent
from PyQt5.QtGui import QColor, QPalette
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
        main.setContentsMargins(0, 2, 2, 2)
        main.setSpacing(0)

        self.toggle_btn = QToolButton()
        self.toggle_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_btn.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.toggle_btn.setText(f"  {title}")
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.setChecked(expanded)
        self.toggle_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.toggle_btn.setMinimumHeight(34)
        self.toggle_btn.clicked.connect(self._toggle)
        main.addWidget(self.toggle_btn)

        # Parent to self so content is never a top-level window; calling
        # setVisible() on a parentless QWidget flashes default 640x480 native
        # chrome on Windows 11.
        self.content = QWidget(self)
        self.content_layout = QFormLayout(self.content)
        self.content_layout.setContentsMargins(2, 2, 2, 2)
        self.content_layout.setSpacing(4)
        self.content_layout.setVerticalSpacing(6)
        self.content_layout.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.content_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        self.content_layout.setLabelAlignment(Qt.AlignLeft)
        main.addWidget(self.content)
        self.content.setVisible(expanded)

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
                padding: 6px 4px 6px 4px;
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

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, self._steps)
        self.slider.setValue(self._float_to_slider(value))
        self.slider.setMinimumWidth(40)
        self.slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(self.slider, stretch=1)

        self.spinbox = QDoubleSpinBox()
        self.spinbox.setRange(-1e15, 1e15)
        self.spinbox.setDecimals(decimals)
        self.spinbox.setValue(value)
        self.spinbox.setButtonSymbols(QAbstractSpinBox.PlusMinus)
        self.spinbox.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.spinbox.setFixedWidth(78)
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
                selection-color: white;
            }}
            QDoubleSpinBox:focus {{
                border: 2px solid {colors['accent']};
                padding: 1px 3px;
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
        self._main_layout.setContentsMargins(2, 4, 2, 4)
        self._main_layout.setSpacing(6)
        self._main_layout.setAlignment(Qt.AlignTop)

        self.logger = logging.getLogger(__name__)
        self.block = None
        self._widgets = {}   # key -> (editor, reset_btn, val_label)
        self._pin_btns = {}  # key -> QPushButton (only for params with sliders)
        self._defaults = {}  # key -> default value
        self._sections = []  # CollapsibleSection references
        self._focus_out_submit = {}  # widget -> callable (PyQt5 5.15 workaround)
        # Diagram-inspector context (V1 empty-state): set by main_window.
        self._dsim = None
        self._main_window = None
        # TODO(multi-select V3): drop a richer multi-block view here. For now the
        # selection model already routes only the first block via set_block(), so
        # the empty-state view shows the diagram and the block-state view shows
        # the single-block form — no regression vs. previous behavior.

        theme_manager.theme_changed.connect(self._update_theme)
        self._update_theme()

    def set_diagram_context(self, dsim, main_window=None):
        """Wire the inspector to the application so the V1 empty-state view
        can read solver settings, workspace vars, recent runs, validation."""
        self._dsim = dsim
        self._main_window = main_window
        # Re-render the empty state if no block is selected
        if self.block is None:
            self._clear_form()
            self._show_diagram_inspector()

    def eventFilter(self, obj, event):
        """PyQt5 5.15 workaround: QLineEdit inside QScrollArea doesn't get
        Qt focus on click, so cursor is invisible and editingFinished never
        fires. Force focus on mouse press; catch FocusOut to trigger submit."""
        if event.type() == QEvent.MouseButtonPress and obj in self._focus_out_submit:
            obj.setFocus(Qt.MouseFocusReason)
        elif event.type() == QEvent.FocusOut and obj in self._focus_out_submit:
            self._focus_out_submit[obj]()
        return super().eventFilter(obj, event)

    def _flush_pending_edits(self):
        """Submit any in-progress QLineEdit edits before the form is rebuilt.

        PyQt5 5.15 doesn't always fire QLineEdit.editingFinished on focus
        loss, so we trigger its submit callback when switching blocks.
        Only QLineEdit needs this — QSpinBox/QDoubleSpinBox/QComboBox/
        QCheckBox all commit reliably via their own signals. Flushing
        those here would emit spurious property_changed events for the
        previously-selected block every time the user clicks another
        block, because submit_fn would fire for every visible widget
        regardless of whether the user edited it."""
        for widget, submit_fn in list(self._focus_out_submit.items()):
            try:
                if isinstance(widget, QLineEdit) and widget.isModified():
                    submit_fn()
            except RuntimeError:
                pass  # Widget already deleted

    # ── Public API ──────────────────────────────────────────────

    def set_block(self, block):
        """Set the block to be edited and create the property form."""
        self._flush_pending_edits()
        self.block = block
        self._clear_form()
        if self.block is None:
            if self._dsim is not None:
                self._show_diagram_inspector()
            else:
                self._show_placeholder()
        else:
            self._create_form()
            self._create_connections_section()
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
        self._pin_btns.clear()
        self._defaults.clear()
        self._sections.clear()
        self._focus_out_submit.clear()

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

    # ── Diagram inspector (V1 empty-state) ────────────────────────

    def _show_diagram_inspector(self):
        """Render the V1 'diagram defaults' view when no block is selected.

        Sections: header, Solver, Workspace, Recent runs, Validation.
        All values are best-effort read from dsim — missing data is hidden,
        not stubbed, so the view is honest about what's available.
        """
        import os as _os
        text_primary = theme_manager.get_color('text_primary').name()
        text_secondary = theme_manager.get_color('text_secondary').name()
        text_disabled = theme_manager.get_color('text_disabled').name()
        success = theme_manager.get_color('success').name()
        warning = theme_manager.get_color('warning').name()
        error = theme_manager.get_color('error').name()
        border = theme_manager.get_color('border_primary').name()
        accent = theme_manager.get_color('accent_primary').name()

        # ── Header
        header = QWidget()
        h_lay = QVBoxLayout(header)
        h_lay.setContentsMargins(8, 6, 8, 8)
        h_lay.setSpacing(2)

        eyebrow = QLabel("DIAGRAM")
        ef = eyebrow.font(); ef.setPointSize(8); ef.setBold(True); eyebrow.setFont(ef)
        eyebrow.setStyleSheet(f"color: {text_disabled};")
        h_lay.addWidget(eyebrow)

        filepath = (getattr(self._dsim, 'current_filepath', None)
                    or getattr(self._dsim, 'filepath', None))
        name = (_os.path.splitext(_os.path.basename(filepath))[0]
                if filepath else "untitled")
        title = QLabel(name)
        tf = title.font(); tf.setPointSize(13); tf.setBold(True); title.setFont(tf)
        title.setStyleSheet(f"color: {text_primary};")
        h_lay.addWidget(title)

        blocks = list(getattr(self._dsim, 'blocks_list', []) or [])
        wires = list(getattr(self._dsim, 'line_list', []) or [])
        sub = QLabel(f"{len(blocks)} blocks · {len(wires)} wires")
        sub.setStyleSheet(f"color: {text_secondary}; font-size: 10pt;")
        h_lay.addWidget(sub)

        self._main_layout.addWidget(header)

        # ── Solver section
        sec = CollapsibleSection("Solver", expanded=True)
        self._sections.append(sec)

        # Use SimulationConfig and DSim attributes
        sim_cfg = getattr(self._main_window, 'sim_config', None) if self._main_window else None
        solver_fields = []
        # Method
        solver_fields.append(('solver', getattr(sim_cfg, 'solver', 'RK45') if sim_cfg else 'RK45'))
        # Step / duration
        sim_dt = getattr(self._dsim, 'sim_dt', None)
        sim_time = getattr(self._dsim, 'sim_time', None)
        if sim_dt is not None:
            solver_fields.append(('step_size', f"{sim_dt} s"))
        if sim_time is not None:
            solver_fields.append(('duration', f"{sim_time} s"))
        # Tolerances
        if sim_cfg is not None:
            for attr, label in [('rtol', 'rtol'), ('atol', 'atol')]:
                if hasattr(sim_cfg, attr):
                    solver_fields.append((label, str(getattr(sim_cfg, attr))))
        # Fast solver
        fast = getattr(self._main_window, 'use_fast_solver', False) if self._main_window else False
        solver_fields.append(('fast_solver', "✓ on" if fast else "off"))

        for k, v in solver_fields:
            sec.addRow(self._mk_kv_label(k, text_secondary),
                       self._mk_kv_value(v, text_primary, mono=True))
        self._main_layout.addWidget(sec)

        # ── Workspace
        wsec = CollapsibleSection("Workspace", expanded=True)
        self._sections.append(wsec)
        try:
            ws = WorkspaceManager().variables or {}
        except Exception:
            ws = {}
        if not ws:
            empty = QLabel("(no workspace variables)")
            empty.setStyleSheet(f"color: {text_disabled}; padding: 4px;")
            wsec.addRow("", empty)
        else:
            # Show up to 12 — past that, link to the workspace editor
            for k, v in list(ws.items())[:12]:
                wsec.addRow(self._mk_kv_label(k, text_secondary),
                            self._mk_kv_value(str(v), text_primary, mono=True))
            if len(ws) > 12:
                more = QLabel(f"… +{len(ws) - 12} more")
                more.setStyleSheet(f"color: {text_disabled}; padding: 2px;")
                wsec.addRow("", more)
        self._main_layout.addWidget(wsec)

        # ── Recent runs
        history = []
        plotter = getattr(self._dsim, 'scope_plotter', None)
        if plotter is not None:
            history = list(getattr(plotter, 'run_history', None)
                           or getattr(self._dsim, 'run_history', None)
                           or [])
        if history:
            rsec = CollapsibleSection("Recent runs", expanded=False)
            self._sections.append(rsec)
            for i, run in enumerate(reversed(history[-4:])):
                label = getattr(run, 'label', None) or f"Run {len(history) - i}"
                rsec.addRow(self._mk_kv_label(str(label), text_secondary),
                            self._mk_kv_value("", text_disabled, mono=True))
            self._main_layout.addWidget(rsec)

        # ── Validation
        try:
            errors = []
            validator = getattr(self._dsim, 'diagram_validator', None)
            if validator is not None and hasattr(validator, 'validate'):
                errors = list(validator.validate() or [])
        except Exception:
            errors = []
        if errors:
            vsec = CollapsibleSection("Validation", expanded=True)
            self._sections.append(vsec)
            for err in errors[:10]:
                sev_raw = getattr(err, 'severity', None)
                sev = (sev_raw.value if hasattr(sev_raw, 'value')
                       else str(sev_raw or '')).lower()
                color = (error if 'error' in sev else
                         warning if 'warn' in sev else
                         success)
                dot = QLabel("●")
                dot.setStyleSheet(f"color: {color};")
                msg = QLabel(getattr(err, 'message', str(err)))
                msg.setWordWrap(True)
                msg.setStyleSheet(f"color: {text_primary}; font-size: 10pt;")
                vsec.addRow(dot, msg)
            self._main_layout.addWidget(vsec)
        elif blocks:
            # No issues — show the "all good" confirmation
            vsec = CollapsibleSection("Validation", expanded=False)
            self._sections.append(vsec)
            ok = QLabel("✓ no issues detected")
            ok.setStyleSheet(f"color: {success}; padding: 2px;")
            vsec.addRow("", ok)
            self._main_layout.addWidget(vsec)

        self._main_layout.addStretch(1)
        self._update_theme()

    def _mk_kv_label(self, text, color):
        lbl = QLabel(text)
        f = lbl.font(); f.setPointSize(10); lbl.setFont(f)
        lbl.setStyleSheet(f"color: {color};")
        return lbl

    def _mk_kv_value(self, text, color, mono=False):
        lbl = QLabel(text)
        if mono:
            from PyQt5.QtGui import QFont as _QF
            f = _QF("Menlo"); f.setStyleHint(_QF.Monospace); f.setPointSize(9)
            if hasattr(f, 'setFamilies'):
                f.setFamilies(["Menlo", "Consolas", "JetBrains Mono", "DejaVu Sans Mono", "monospace"])
            lbl.setFont(f)
        lbl.setStyleSheet(f"color: {color};")
        return lbl

    # ── Connections section (V2 block view) ───────────────────────

    def _create_connections_section(self):
        """Show `in[i] ← src` and `out[i] → dst` rows for the selected block.

        Best-effort: reads from `dsim.line_list`. Skips silently if line schema
        doesn't match what we expect (different DSim build).
        """
        if self.block is None or self._dsim is None:
            return
        try:
            lines = list(getattr(self._dsim, 'line_list', []) or [])
            name = getattr(self.block, 'name', None)
            if not name:
                return
            inbound = []
            outbound = []
            for line in lines:
                src_name = getattr(line, 'srcblock', None) or getattr(getattr(line, 'src_block', None), 'name', None)
                dst_name = getattr(line, 'dstblock', None) or getattr(getattr(line, 'dst_block', None), 'name', None)
                src_port = getattr(line, 'srcport', getattr(line, 'src_port', 0))
                dst_port = getattr(line, 'dstport', getattr(line, 'dst_port', 0))
                if dst_name == name:
                    inbound.append((dst_port, src_name, src_port))
                if src_name == name:
                    outbound.append((src_port, dst_name, dst_port))
            if not inbound and not outbound:
                return

            sec = CollapsibleSection("Connections", expanded=False)
            self._sections.append(sec)
            text_secondary = theme_manager.get_color('text_secondary').name()
            text_primary = theme_manager.get_color('text_primary').name()

            for port, peer, peer_port in sorted(inbound):
                sec.addRow(
                    self._mk_kv_label(f"in[{port}]", text_secondary),
                    self._mk_kv_value(f"← {peer}.out[{peer_port}]", text_primary, mono=True),
                )
            for port, peer, peer_port in sorted(outbound):
                sec.addRow(
                    self._mk_kv_label(f"out[{port}]", text_secondary),
                    self._mk_kv_value(f"→ {peer}.in[{peer_port}]", text_primary, mono=True),
                )
            self._main_layout.addWidget(sec)
            self._update_theme()
        except Exception as e:  # do not break the inspector for unknown DSim line shapes
            self.logger.debug("Could not build connections section: %s", e)

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
        reset_btn.setFixedSize(20, 22)
        reset_btn.setToolTip(f"Reset to default: {default}")
        reset_btn.setCursor(Qt.PointingHandCursor)
        reset_btn.clicked.connect(lambda checked, k=key: self._reset_param(k))
        # setVisible() deferred until after addWidget below \u2014 calling it on a
        # parentless QPushButton flashes default native chrome on Windows 11.
        reset_btn_visible = self._value_differs(value, default)

        # Validation label (#7)
        error_color = theme_manager.get_color('error').name()
        val_label = QLabel("")
        val_label.setWordWrap(True)
        val_label.setStyleSheet(f"color: {error_color}; font-size: 11px; padding-left: 2px;")
        val_label.hide()

        pin_btn = None
        if isinstance(editor, SliderSpinBox):
            accent = theme_manager.get_color('accent_primary').name()
            pin_btn = QPushButton("\u25C9  Pin to tuning")
            pin_btn.setFlat(True)
            pin_btn.setCursor(Qt.PointingHandCursor)
            pin_btn.setToolTip("Pin this parameter to the Tuning Panel")
            pin_btn.setStyleSheet(self._pin_button_stylesheet(accent))
            pin_btn.clicked.connect(lambda checked, k=key: self._on_pin_to_tuning(k))
            self._pin_btns[key] = pin_btn

        self._widgets[key] = (editor, reset_btn, val_label)

        container = QWidget()
        c_layout = QVBoxLayout(container)
        c_layout.setContentsMargins(0, 0, 0, 0)
        c_layout.setSpacing(2)
        row = QHBoxLayout()
        row.setSpacing(2)
        row.addWidget(editor, stretch=1)
        row.addWidget(reset_btn, stretch=0)
        c_layout.addLayout(row)
        if pin_btn:
            c_layout.addWidget(pin_btn, alignment=Qt.AlignRight)
        c_layout.addWidget(val_label)

        section.addRow(label, container)
        # Apply reset-button visibility only after the widget has been
        # parented through section.addRow — calling setVisible() on a
        # parentless QPushButton flashes a top-level window on Windows 11.
        reset_btn.setVisible(reset_btn_visible)

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
            submit = lambda: self._on_property_changed(key, sb.value())
            sb.editingFinished.connect(submit)
            sb.installEventFilter(self)
            self._focus_out_submit[sb] = submit
            self._apply_widget_sizing(sb)
            return sb

        # Float
        if isinstance(value, float):
            if accepts_array:
                le = QLineEdit(str(value))
                self._apply_widget_sizing(le)
                le.setPlaceholderText("e.g. 1.0 or [1, 2, 3]")
                submit = lambda: self._validate_and_submit_numeric(key, le)
                le.editingFinished.connect(submit)
                le.installEventFilter(self)
                self._focus_out_submit[le] = submit
                return le

            # Slider + spinbox (#5)
            if self._should_show_slider(key, value, meta, group):
                sr = self._get_slider_range(value, meta)
                ssb = SliderSpinBox(value, sr[0], sr[1])
                submit = lambda: self._on_property_changed(key, ssb.value())
                ssb.editingFinished.connect(submit)
                ssb.spinbox.installEventFilter(self)
                self._focus_out_submit[ssb.spinbox] = submit
                return ssb

            # Standard spinbox
            dsb = QDoubleSpinBox()
            dsb.setRange(-float('inf'), float('inf'))
            dsb.setDecimals(4)
            dsb.setValue(value)
            submit = lambda: self._on_property_changed(key, dsb.value())
            dsb.editingFinished.connect(submit)
            dsb.installEventFilter(self)
            self._focus_out_submit[dsb] = submit
            self._apply_widget_sizing(dsb)
            return dsb

        # Lists, strings, other → QLineEdit
        le = QLineEdit(str(value))
        self._apply_widget_sizing(le)
        submit = lambda: self._validate_and_submit_string(key, le, type(value))
        le.editingFinished.connect(submit)
        le.installEventFilter(self)
        self._focus_out_submit[le] = submit
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
        if not math.isfinite(value) or value == 0:
            return False
        return True

    def _get_slider_range(self, value, meta):
        if 'range' in meta:
            return list(meta['range'])
        span = abs(value) * 10
        return [-span, span]

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

    @staticmethod
    def _pin_button_stylesheet(accent):
        return (
            f"QPushButton {{ color: {accent}; font-size: 10px; "
            f"border: none; padding: 0px; text-align: right; }}"
            f"QPushButton:hover {{ text-decoration: underline; }}"
        )

    def _update_theme(self):
        bg = theme_manager.get_color('surface').name()
        txt = theme_manager.get_color('text_primary').name()
        border = theme_manager.get_color('border_primary').name()
        input_bg = theme_manager.get_color('surface_variant').name()
        accent = theme_manager.get_color('accent_primary').name()
        error = theme_manager.get_color('error').name()

        # Apply all input styling via the container stylesheet so it
        # cascades with (rather than replaces) the global QSS.
        # Per-widget setStyleSheet() kills the global :focus/:hover rules.
        self.setStyleSheet(f"""
            PropertyEditor {{
                background-color: {bg};
                border: 1px solid {border};
                border-radius: 4px;
            }}
            PropertyEditor QCheckBox {{ color: {txt}; spacing: 8px; }}
            PropertyEditor QLabel {{ color: {txt}; }}
            PropertyEditor QLineEdit,
            PropertyEditor QSpinBox,
            PropertyEditor QDoubleSpinBox,
            PropertyEditor QComboBox {{
                background-color: {input_bg};
                color: {txt};
                border: 1px solid {border};
                border-radius: 4px;
                padding: 2px 4px;
                selection-background-color: {accent};
                selection-color: white;
            }}
            PropertyEditor QLineEdit:focus,
            PropertyEditor QSpinBox:focus,
            PropertyEditor QDoubleSpinBox:focus,
            PropertyEditor QComboBox:focus {{
                color: {txt};
                border: 2px solid {accent};
                padding: 1px 3px;
            }}
            PropertyEditor QLineEdit:hover,
            PropertyEditor QSpinBox:hover,
            PropertyEditor QDoubleSpinBox:hover,
            PropertyEditor QComboBox:hover {{
                border-color: {accent};
            }}
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
                    min-width: 0px; min-height: 0px;
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

        pin_qss = self._pin_button_stylesheet(accent)
        for pin_btn in self._pin_btns.values():
            pin_btn.setStyleSheet(pin_qss)

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
