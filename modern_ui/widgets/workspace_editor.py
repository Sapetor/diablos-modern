from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, 
                             QTableWidgetItem, QPushButton, QHeaderView, QInputDialog, 
                             QMessageBox, QToolBar, QAction, QMenu)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QIcon
import logging
from lib.workspace import WorkspaceManager
from modern_ui.themes.theme_manager import theme_manager

logger = logging.getLogger(__name__)

class WorkspaceEditor(QWidget):
    """
    A widget that provides a table view for managing workspace variables.
    Allows users to view, add, edit, and delete variables.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.workspace_manager = WorkspaceManager()
        self._init_ui()
        self.refresh_variables()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toolbar
        self.toolbar = QToolBar()
        self.toolbar.setIconSize(theme_manager.get_icon_size())
        layout.addWidget(self.toolbar)

        # Actions
        self.action_refresh = QAction("Refresh", self)
        self.action_refresh.setToolTip("Refresh variables from workspace")
        self.action_refresh.triggered.connect(self.refresh_variables)
        self.action_refresh.setText("🔄")
        self.toolbar.addAction(self.action_refresh)

        self.action_add = QAction("Add", self)
        self.action_add.setToolTip("Add new variable")
        self.action_add.triggered.connect(self.add_variable)
        self.action_add.setText("➕")
        self.toolbar.addAction(self.action_add)

        self.action_delete = QAction("Delete", self)
        self.action_delete.setToolTip("Delete selected variable")
        self.action_delete.triggered.connect(self.delete_variable)
        self.action_delete.setText("➖")
        self.toolbar.addAction(self.action_delete)
        
        self.action_save = QAction("Save", self)
        self.action_save.setToolTip("Save workspace to file")
        self.action_save.triggered.connect(self.save_workspace)
        self.action_save.setText("💾")
        self.toolbar.addAction(self.action_save)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Name", "Value", "Type"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents) # Name
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)          # Value
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents) # Type
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.cellChanged.connect(self.on_cell_changed)
        
        layout.addWidget(self.table)

        # Connect to theme changes
        theme_manager.theme_changed.connect(self._apply_theme)
        
        # Apply initial theme
        self._apply_theme()

    def _apply_theme(self):
        """Apply the current theme to the widget."""
        bg_color = theme_manager.get_color("background_secondary").name()
        border_color = theme_manager.get_color("border_primary").name()
        text_color = theme_manager.get_color("text_primary").name()
        table_bg = theme_manager.get_color("surface").name()
        alt_bg = theme_manager.get_color("surface_variant").name()
        
        # Toolbar styling
        self.toolbar.setStyleSheet(f"""
            QToolBar {{ 
                border-bottom: 1px solid {border_color}; 
                background: {bg_color}; 
                spacing: 5px; 
                color: {text_color};
            }}
            QToolButton {{
                color: {text_color};
                background: transparent;
                padding: 5px;
                border-radius: 4px;
            }}
            QToolButton:hover {{
                background: {theme_manager.get_color("surface_elevated").name()};
            }}
        """)
        
        # Table styling
        self.table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {table_bg};
                color: {text_color};
                gridline-color: {border_color};
                border: none;
                selection-background-color: {theme_manager.get_color("primary").name()};
                selection-color: white;
            }}
            QHeaderView::section {{
                background-color: {bg_color};
                color: {text_color};
                padding: 4px;
                border: 1px solid {border_color};
            }}
        """)
        self.table.setAlternatingRowColors(True)
        # Note: setAlternatingRowColors uses the widget's palette, which might need specific tuning if the stylesheet doesn't override it fully for rows. 
        # But usually QTableWidget stylesheet handles it if we set alternate-background-color property, or we can leave it to defaults.

    def refresh_variables(self):
        """Reload variables from the workspace manager."""
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        
        variables = self.workspace_manager.get_all_variables()
        row = 0
        for name, value in sorted(variables.items()):
            self.table.insertRow(row)
            
            # Name (Read-only for now, created via Add)
            name_item = QTableWidgetItem(name)
            name_item.setFlags(name_item.flags() ^ Qt.ItemIsEditable)
            self.table.setItem(row, 0, name_item)
            
            # Value (Editable)
            # We display repr(value) to show quotes for strings, list brackets, etc.
            # But for editing, maybe just the str? No, using repr is safer for parsing back.
            val_str = repr(value)
            val_item = QTableWidgetItem(val_str)
            self.table.setItem(row, 1, val_item)
            
            # Type (Read-only)
            type_item = QTableWidgetItem(type(value).__name__)
            type_item.setFlags(type_item.flags() ^ Qt.ItemIsEditable)
            self.table.setItem(row, 2, type_item)
            
            row += 1
            
        self.table.blockSignals(False)

    def add_variable(self):
        """Add a new variable dialog."""
        name, ok = QInputDialog.getText(self, "Add Variable", "Variable Name:")
        if ok and name:
            name = name.strip()
            if not name.isidentifier():
                QMessageBox.warning(self, "Invalid Name", "Variable name must be a valid Python identifier.")
                return
                
            if name in self.workspace_manager.variables:
                QMessageBox.warning(self, "Exists", "Variable already exists. Edit it directly in the table.")
                return
            
            val_str, ok_val = QInputDialog.getText(self, "Initial Value", f"Value for {name}:", text="0")
            if ok_val:
                try:
                    # Evaluate the string to get the actual python object
                    # We accept basic literals: numbers, strings, lists, bools
                    val = eval(val_str, {"__builtins__": {}}, {})
                    self.workspace_manager.set_variable(name, val)
                    self.refresh_variables()
                    logger.info(f"Added variable {name} = {val}")
                except Exception as e:
                    QMessageBox.warning(self, "Invalid Value", f"Could not parse value: {e}")

    def delete_variable(self):
        """Delete user selected variable."""
        row = self.table.currentRow()
        if row >= 0:
            name = self.table.item(row, 0).text()
            confirm = QMessageBox.question(self, "Confirm Delete", f"Are you sure you want to delete '{name}'?", 
                                         QMessageBox.Yes | QMessageBox.No)
            if confirm == QMessageBox.Yes:
                self.workspace_manager.delete_variable(name)
                self.refresh_variables()
                logger.info(f"Deleted variable {name}")

    def on_cell_changed(self, row, column):
        """Handle cell edits (Value column only)."""
        if column == 1:
            name = self.table.item(row, 0).text()
            new_val_str = self.table.item(row, 1).text()
            
            try:
                # Basic safety: allow math literals but not arbitrary code execution if possible.
                # using eval on user input is risky but this is a workspace editor for engineers.
                # Assuming trusted user.
                import math
                safe_env = {"__builtins__": {}, "math": math, "list": list, "int": int, "float": float, "True": True, "False": False}
                
                # We also want to support referencing other variables? 
                # WorkspaceManager.resolve_params does that. 
                # But here we want to define the *base* value.
                # If I type 'K + 1' here, and K is a variable, do I want to store the result or the expression?
                # Usually a workspace stores VALUES. So we evaluate it now.
                
                new_val = eval(new_val_str, safe_env, self.workspace_manager.variables)
                
                # Update backend
                self.workspace_manager.set_variable(name, new_val)
                
                # Update type column
                self.table.blockSignals(True) # Prevent recursion
                type_item = QTableWidgetItem(type(new_val).__name__)
                type_item.setFlags(type_item.flags() ^ Qt.ItemIsEditable)
                self.table.setItem(row, 2, type_item)
                self.table.blockSignals(False)
                
                logger.info(f"Updated variable {name} to {new_val}")
                
            except Exception as e:
                # If invalid, revert or show error?
                # For now, just log and maybe change color?
                logger.warning(f"Invalid value for {name}: {e}")
                self.table.item(row, 1).setForeground(QColor("red"))
                self.table.item(row, 1).setToolTip(str(e))

    def save_workspace(self):
        """Save workspace to file."""
        if self.workspace_manager.save_to_file():
            QMessageBox.information(self, "Saved", "Workspace saved successfully.")
        else:
            QMessageBox.warning(self, "Error", "Failed to save workspace. Check log for details.")
