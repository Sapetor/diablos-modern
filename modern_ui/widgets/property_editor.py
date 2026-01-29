"""
Property Editor Widget for DiaBloS
Dynamically creates a form to edit the properties of a selected block.
Now features specialized widgets for different types and input validation.
"""

import logging
import ast
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QLineEdit, QComboBox, QCheckBox, 
    QFrame, QFormLayout, QSpinBox, QDoubleSpinBox, QToolTip
)
from PyQt5.QtCore import pyqtSignal, Qt, QSize
from PyQt5.QtGui import QIntValidator, QDoubleValidator, QPalette, QColor
from modern_ui.themes.theme_manager import theme_manager
from lib.workspace import WorkspaceManager

class PropertyEditor(QFrame):
    """A widget that dynamically creates a form to edit block properties."""

    property_changed = pyqtSignal(str, str, object)  # block_name, prop_name, new_value

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("PropertyEditor")
        self.setFrameStyle(QFrame.StyledPanel)
        self.setAutoFillBackground(True)

        self.layout = QFormLayout(self)
        self.layout.setContentsMargins(12, 12, 12, 12)
        self.layout.setSpacing(8)
        self.layout.setVerticalSpacing(12)
        self.layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        self.layout.setLabelAlignment(Qt.AlignLeft)

        self.logger = logging.getLogger(__name__)
        self.block = None

        # Connect to theme changes
        theme_manager.theme_changed.connect(self._update_theme)
        self._update_theme()
        
    def set_block(self, block):
        """Set the block to be edited and create the property form."""
        self.block = block
        self._clear_form()
        
        if self.block is None:
            self._show_placeholder()
        else:
            self._create_form()
        
        self.updateGeometry()
        
    def _clear_form(self):
        """Clear the existing form widgets."""
        while self.layout.count():
            child = self.layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
                
    def _show_placeholder(self):
        """Show a placeholder message when no block is selected."""
        placeholder = QLabel("Select a block to view its properties.")
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setWordWrap(True)
        text_color = theme_manager.get_color('text_secondary').name()
        placeholder.setStyleSheet(f"color: {text_color}; padding: 20px;")
        self.layout.addRow(placeholder)
        
    def _create_form(self):
        """Create the form widgets for the block's properties."""
        if not hasattr(self.block, 'params'):
            return
            
        # Sort keys to put 'Name' first (if it existed in params, usually it's separate)
        # For now, just generic loop but skip internal keys
        keys = [k for k in self.block.params.keys() if not k.startswith('_')]
        
        for key in keys:
            value = self.block.params[key]
            label = QLabel(f"{key.replace('_', ' ').title()}:")
            text_color = theme_manager.get_color('text_primary').name()
            label.setStyleSheet(f"color: {text_color}; font-weight: bold;")
            
            widget = self._create_editor_for_value(key, value)
            if widget:
                # Add tooltip if we had descriptions (future improvement)
                self.layout.addRow(label, widget)

        # Add Documentation Section
        if hasattr(self.block, 'doc') and self.block.doc:
            # Separator
            line = QFrame()
            line.setFrameShape(QFrame.HLine)
            line.setFrameShadow(QFrame.Sunken)
            line.setStyleSheet(f"background-color: {theme_manager.get_color('border_primary').name()};")
            self.layout.addRow(line)

            # Header
            doc_header = QLabel("Documentation")
            header_color = theme_manager.get_color('accent').name()
            doc_header.setStyleSheet(f"color: {header_color}; font-weight: bold; font-size: 13px; margin-top: 8px;")
            self.layout.addRow(doc_header)

            # Doc Text
            doc_label = QLabel(str(self.block.doc).strip())
            doc_label.setWordWrap(True)
            text_secondary = theme_manager.get_color('text_secondary').name()
            doc_label.setStyleSheet(f"color: {text_secondary}; font-style: italic; margin-bottom: 8px;")
            self.layout.addRow(doc_label)

            # Link (Dynamic GitHub Wiki Link)
            base_url = "https://github.com/Sapetor/diablos-modern/blob/main/docs/wiki"
            # Category file (e.g. "Control.md")
            cat_file = f"{getattr(self.block, 'category', 'Home')}.md".replace(' ', '-')
            # Anchor (e.g. "#integrator")
            # Use block_fn or base name if possible (strip ID suffix digits if needed, but block_fn is usually pure)
            anchor = getattr(self.block, 'block_fn', 'Home').lower().replace(' ', '-')
            
            full_url = f"{base_url}/{cat_file}#{anchor}"
            
            link_label = QLabel(f'<a href="{full_url}">View Full Reference</a>')
            link_label.setOpenExternalLinks(True)
            link_label.setStyleSheet(f"color: {theme_manager.get_color('accent').name()};")
            self.layout.addRow(link_label)

    def _create_editor_for_value(self, key, value):
        """Factory method to create the appropriate editor widget based on value type."""
        
        # 1. Boolean -> QCheckBox
        if isinstance(value, bool):
            cx = QCheckBox()
            cx.setChecked(value)
            cx.toggled.connect(lambda state, k=key: self._on_property_changed(k, state))
            return cx
            
        # 2. Known Enums -> QComboBox
        if key == 'method':
            cb = QComboBox()
            cb.addItems(["FWD_EULER", "BWD_EULER", "TUSTIN", "RK45", "SOLVE_IVP"])
            cb.setCurrentText(str(value))
            cb.currentTextChanged.connect(lambda text, k=key: self._on_property_changed(k, text))
            self._apply_widget_sizing(cb)
            return cb
            
        # 3. Integers -> QSpinBox
        # Note: Some 'ints' in params might be large, so check range or use QLineEdit if unsure.
        # But commonly these are small counts like ports.
        if isinstance(value, int):
            sb = QSpinBox()
            sb.setRange(-999999, 999999) # Generous range
            sb.setValue(value)
            # Use 'valueChanged' but maybe wait for editingFinished to avoid spamming updates during typing?
            # QSpinBox emits valueChanged on typing too. 'editingFinished' is safer for complex updates.
            sb.editingFinished.connect(lambda: self._on_property_changed(key, sb.value()))
            self._apply_widget_sizing(sb)
            return sb
            
        # 4. Floats -> QDoubleSpinBox
        if isinstance(value, float):
            dsb = QDoubleSpinBox()
            dsb.setRange(-float('inf'), float('inf'))
            dsb.setDecimals(4) # Reasonable default
            dsb.setValue(value)
            dsb.editingFinished.connect(lambda: self._on_property_changed(key, dsb.value()))
            self._apply_widget_sizing(dsb)
            return dsb
            
        # 5. Lists/Arrays or Complex Strings -> QLineEdit with Validation support
        # We'll use a generic LineEdit but try to validate
        le = QLineEdit(str(value))
        
        # Determine styling based on device pixel ratio
        self._apply_widget_sizing(le)
        
        # Connect signal
        le.editingFinished.connect(lambda: self._validate_and_submit_string(key, le, type(value)))
        
        # Tooltip hint for format
        if isinstance(value, list):
            le.setPlaceholderText("e.g. [1, 2, 3]")
            le.setToolTip("Enter a list of numbers, e.g., [1, 0, 0]")
            
        return le

    def _apply_widget_sizing(self, widget):
        """Apply responsive sizing logic for widgets."""
        from PyQt5.QtWidgets import QApplication
        screen = QApplication.primaryScreen()
        device_ratio = screen.devicePixelRatio()
        base_width = 180
        if device_ratio > 1.25:
            scaled_width = int(base_width * 1.3)
        else:
            scaled_width = base_width
        widget.setMinimumWidth(scaled_width)

    def _validate_and_submit_string(self, key, widget, target_type):
        """Validate string input before submission."""
        text = widget.text()
        is_valid = True
        converted_value = None
        
        try:
            if target_type == list:
                # Try to parse as list
                try:
                    val = ast.literal_eval(text)
                    if not isinstance(val, list):
                        raise ValueError("Not a list")
                    converted_value = val
                except (ValueError, SyntaxError) as e:
                    # Check if it's a variable in workspace that evaluates to a list
                    ws = WorkspaceManager()
                    if text in ws.variables and isinstance(ws.variables[text], list):
                         converted_value = text # Keep as string reference
                    else:
                         raise e

            elif target_type == int:
                try:
                    converted_value = int(text)
                except ValueError:
                    # Check workspace
                    ws = WorkspaceManager()
                    if text in ws.variables:
                        # We accept it as a reference. Use string.
                        # Backend (resolve_params) will handle it.
                        converted_value = text
                    else:
                        raise ValueError(f"'{text}' is not an integer or known variable")

            elif target_type == float:
                try:
                     converted_value = float(text)
                except ValueError:
                    # Check workspace
                    ws = WorkspaceManager()
                    if text in ws.variables:
                         converted_value = text
                    else:
                         raise ValueError(f"'{text}' is not a float or known variable")
            else:
                converted_value = text # String is always valid
                
        except (ValueError, SyntaxError):
            is_valid = False
            
        if is_valid:
            # clear error style
            self._set_widget_error_state(widget, False)
            self._on_property_changed(key, converted_value)
        else:
            # set error style
            self._set_widget_error_state(widget, True)
            self.logger.warning(f"Invalid input not submitted for {key}: {text}")

    def _set_widget_error_state(self, widget, is_error):
        """Apply error styling to widget."""
        border_color = theme_manager.get_color('error').name() if is_error else theme_manager.get_color('border_primary').name()
        
        # We need to maintain the base style but change border
        # This is a bit hacky because we're overriding the whole stylesheet again for this widget
        bg_color = theme_manager.get_color('surface_variant').name()
        text_color = theme_manager.get_color('text_primary').name()
        
        widget.setStyleSheet(f"""
            QLineEdit {{
                background-color: {bg_color};
                color: {text_color};
                border: 2px solid {border_color};
                border-radius: 4px;
                padding: 2px 4px;
            }}
        """)
        if is_error:
            widget.setToolTip("Invalid input format")
        else:
            widget.setToolTip("")

    def _on_property_changed(self, prop_name, new_value):
        """Emit a signal for a changed property."""
        if self.block is None:
            return

        block_name = getattr(self.block, 'name', 'Unknown')
        self.logger.info(f"Property changed: {block_name}.{prop_name} = {new_value}")
        self.property_changed.emit(block_name, prop_name, new_value)

    def _update_theme(self):
        """Update styling when theme changes."""
        bg_color = theme_manager.get_color('surface').name()
        text_color = theme_manager.get_color('text_primary').name()
        border_color = theme_manager.get_color('border_primary').name()
        input_bg = theme_manager.get_color('surface_variant').name()
        
        # Global stylesheet for the container properties
        self.setStyleSheet(f"""
            PropertyEditor {{
                background-color: {bg_color};
                border: 1px solid {border_color};
                border-radius: 4px;
            }}
            QCheckBox {{
                color: {text_color};
                spacing: 8px;
            }}
            QLabel {{
                color: {text_color};
            }}
        """)
        
        # Re-apply specific widget styles via helper to ensure consistency
        # We iterate over rows and update children styles
        for i in range(self.layout.rowCount()):
            item = self.layout.itemAt(i, QFormLayout.FieldRole)
            if item and item.widget():
                w = item.widget()
                self._apply_widget_theme(w, input_bg, text_color, border_color)

    def _apply_widget_theme(self, widget, bg, text, border):
        """Apply theme to individual editor widgets."""
        base_style = f"""
            background-color: {bg};
            color: {text};
            border: 1px solid {border};
            border-radius: 4px;
            padding: 2px 4px;
        """
        
        if isinstance(widget, (QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox)):
            widget.setStyleSheet(f"{type(widget).__name__} {{ {base_style} }}")

    def sizeHint(self):
        """Provide a dynamic size hint based on the number of properties."""
        if self.block and hasattr(self.block, 'params'):
            num_props = len([p for p in self.block.params if not p.startswith('_')])
            if num_props > 0:
                row_height = 40  # Slightly taller for spacing
                height = num_props * row_height + self.layout.contentsMargins().top() + self.layout.contentsMargins().bottom()
                return QSize(super().sizeHint().width(), height)
        
        return QSize(super().sizeHint().width(), 100)

