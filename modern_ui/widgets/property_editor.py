"""
Property Editor Widget for DiaBloS
Dynamically creates a form to edit the properties of a selected block.
"""

import logging
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QLineEdit, QComboBox, QCheckBox, QFrame, QFormLayout
from PyQt5.QtCore import pyqtSignal, Qt, QSize


class PropertyEditor(QFrame):
    """A widget that dynamically creates a form to edit block properties."""
    
    property_changed = pyqtSignal(str, str, object)  # block_name, prop_name, new_value
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("PropertyEditor")
        self.setFrameStyle(QFrame.StyledPanel)

        # Ensure proper background for property editor
        self.setAutoFillBackground(True)

        self.layout = QFormLayout(self)
        self.layout.setContentsMargins(12, 12, 12, 12)
        self.layout.setSpacing(8)
        self.layout.setVerticalSpacing(12)
        self.layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        self.layout.setLabelAlignment(Qt.AlignLeft)

        self.logger = logging.getLogger(__name__)
        self.block = None
        
    def set_block(self, block):
        """Set the block to be edited and create the property form."""
        self.block = block
        self._clear_form()
        
        if self.block is None:
            self._show_placeholder()
        else:
            self._create_form()
        
        self.updateGeometry() # Notify layout that size hint has changed
        
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
        # Ensure visible styling
        placeholder.setStyleSheet("color: palette(text); padding: 20px;")
        self.layout.addRow(placeholder)
        
    def _create_form(self):
        """Create the form widgets for the block's properties."""
        if not hasattr(self.block, 'params'):
            return
            
        for key, value in self.block.params.items():
            if key.startswith('_'):
                continue
                
            label = QLabel(f"{key.replace('_', ' ').title()}:")
            # Ensure labels are visible
            label.setStyleSheet("color: palette(text);")

            widget = None
            if isinstance(value, bool):
                widget = QCheckBox()
                widget.setChecked(value)
                widget.toggled.connect(
                    lambda state, k=key: self._on_property_changed(k, state)
                )
            elif key == 'method':
                widget = QComboBox()
                widget.addItems(["FWD_EULER", "BWD_EULER", "TUSTIN", "RK45", "SOLVE_IVP"])
                widget.setCurrentText(value)
                widget.currentTextChanged.connect(
                    lambda text, k=key: self._on_property_changed(k, text)
                )
            elif isinstance(value, (list, int, float, str)):
                widget = QLineEdit(str(value))
                # Use devicePixelRatio for proper scaling with Qt's high DPI support
                from PyQt5.QtWidgets import QApplication
                screen = QApplication.primaryScreen()
                device_ratio = screen.devicePixelRatio()

                # Base width that works well - more generous for visibility
                base_width = 150

                # Scale for high DPI
                if device_ratio > 1.25:
                    scaled_width = int(base_width * 1.3)
                else:
                    scaled_width = base_width

                widget.setMinimumWidth(scaled_width)
                # Ensure widget is visible and has proper sizing
                widget.setSizePolicy(widget.sizePolicy().horizontalPolicy(), widget.sizePolicy().verticalPolicy())
                widget.editingFinished.connect(
                    lambda w=widget, k=key: self._on_property_changed(k, w.text())
                )
            else:
                # For unhandled types, just display them as labels
                widget = QLabel(str(value))
                
            if widget:
                self.layout.addRow(label, widget)

    def _on_property_changed(self, prop_name, new_value):
        """Emit a signal for a changed property."""
        if self.block is None:
            return
            
        block_name = getattr(self.block, 'name', 'Unknown')
        self.logger.debug(f"Property changed: {block_name}.{prop_name} = {new_value}")
        self.property_changed.emit(block_name, prop_name, new_value)

    def sizeHint(self):
        """Provide a dynamic size hint based on the number of properties."""
        if self.block and hasattr(self.block, 'params'):
            num_props = len([p for p in self.block.params if not p.startswith('_')])
            if num_props > 0:
                # Calculate height based on number of rows
                row_height = 32  # Approximate height of a form row
                height = num_props * row_height + self.layout.contentsMargins().top() + self.layout.contentsMargins().bottom()
                return QSize(super().sizeHint().width(), height)
        
        # Default size for placeholder
        return QSize(super().sizeHint().width(), 80)
