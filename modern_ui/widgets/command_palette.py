"""
Command Palette - Quick search for blocks, commands, and files.
"""

import logging
from typing import List, Dict, Any, Callable
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QLineEdit, QListWidget, QListWidgetItem,
    QLabel, QWidget, QHBoxLayout
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QKeyEvent, QFont, QColor, QPalette
from modern_ui.themes.theme_manager import theme_manager

logger = logging.getLogger(__name__)

# UI Constants
PALETTE_WIDTH = 280
PALETTE_HEIGHT = 150
RESULTS_MIN_HEIGHT = 70
RESULTS_MAX_HEIGHT = 90
SEARCH_INPUT_HEIGHT = 32
SEARCH_FONT_SIZE = 10
ITEM_FONT_SIZE = 9
CURSOR_OFFSET_X = 10
CURSOR_OFFSET_Y = 10


class CommandPalette(QDialog):
    """
    Quick search/command palette for fast access to blocks, commands, and files.

    Opens with Ctrl+P, allows searching for:
    - Block types to add to canvas
    - Application commands (New, Open, Save, Run, etc.)
    - Recent files
    - Actions and settings
    """

    # Signals
    command_selected = pyqtSignal(str, dict)  # command_type, data

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Popup)
        self.setAttribute(Qt.WA_TranslucentBackground, True)  # Enable transparency

        self.commands: List[Dict[str, Any]] = []
        self.filtered_commands: List[Dict[str, Any]] = []

        self._setup_ui()
        self._apply_theme()

        # Connect to theme changes
        theme_manager.theme_changed.connect(self._apply_theme)

    def _setup_ui(self):
        """Setup the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        # Search input - minimal
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search...")
        self.search_input.setMinimumHeight(SEARCH_INPUT_HEIGHT)
        # Use system default font for cross-platform compatibility
        font = QFont()
        font.setPointSize(SEARCH_FONT_SIZE)
        self.search_input.setFont(font)
        self.search_input.textChanged.connect(self._on_search_changed)
        self.search_input.returnPressed.connect(self._on_item_selected)
        # Install event filter to capture navigation keys from the input field
        self.search_input.installEventFilter(self)
        layout.addWidget(self.search_input)

        # Results list - very compact, show only 2-3 items
        self.results_list = QListWidget()
        self.results_list.setMinimumHeight(RESULTS_MIN_HEIGHT)
        self.results_list.setMaximumHeight(RESULTS_MAX_HEIGHT)
        self.results_list.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self.results_list)

        # Set dialog size - very compact and minimal
        self.setFixedWidth(PALETTE_WIDTH)
        self.setFixedHeight(PALETTE_HEIGHT)

    def _apply_theme(self):
        """Apply current theme styling - minimalist with transparency."""
        bg_color = theme_manager.get_color('surface')
        text_color = theme_manager.get_color('text_primary').name()
        accent_color = theme_manager.get_color('accent').name()

        # Create semi-transparent background
        bg_rgba = f"rgba({bg_color.red()}, {bg_color.green()}, {bg_color.blue()}, 0.95)"

        self.setStyleSheet(f"""
            QDialog {{
                background-color: {bg_rgba};
                border: 1px solid rgba(128, 128, 128, 0.3);
                border-radius: 6px;
            }}
            QLineEdit {{
                background-color: rgba(255, 255, 255, 0.05);
                color: {text_color};
                border: 1px solid rgba(128, 128, 128, 0.2);
                border-radius: 3px;
                padding: 6px 10px;
                font-size: 10pt;
            }}
            QLineEdit:focus {{
                border: 1px solid {accent_color};
            }}
            QListWidget {{
                background-color: transparent;
                color: {text_color};
                border: none;
                outline: none;
                padding: 0px;
            }}
            QListWidget::item {{
                padding: 6px 10px;
                border-radius: 3px;
                margin: 1px 0px;
            }}
            QListWidget::item:hover {{
                background-color: rgba(255, 255, 255, 0.08);
            }}
            QListWidget::item:selected {{
                background-color: {accent_color};
                color: white;
            }}
        """)

    def set_commands(self, commands: List[Dict[str, Any]]):
        """
        Set available commands for the palette.

        Args:
            commands: List of command dictionaries with keys:
                - 'name': Display name
                - 'type': Command type ('block', 'action', 'file', 'setting')
                - 'description': Optional description
                - 'callback': Function to call when selected
                - 'data': Optional data to pass to callback
        """
        self.commands = commands
        self.filtered_commands = commands
        self._update_results()

    def _on_search_changed(self, text: str):
        """Filter commands based on search text."""
        text = text.lower().strip()

        if not text:
            self.filtered_commands = self.commands
        else:
            self.filtered_commands = [
                cmd for cmd in self.commands
                if text in cmd['name'].lower() or
                   (cmd.get('description', '').lower().find(text) != -1)
            ]

        self._update_results()

    def _update_results(self):
        """Update the results list with filtered commands."""
        self.results_list.clear()

        for cmd in self.filtered_commands:
            # Just show the name - plain and minimal
            name = cmd['name']

            item = QListWidgetItem(name)
            item.setData(Qt.UserRole, cmd)

            # Set item font - use system default for cross-platform compatibility
            font = QFont()
            font.setPointSize(ITEM_FONT_SIZE)
            item.setFont(font)

            self.results_list.addItem(item)

        # Select first item
        if self.results_list.count() > 0:
            self.results_list.setCurrentRow(0)

    def _get_type_icon(self, cmd_type: str) -> str:
        """Get emoji icon for command type."""
        icons = {
            'block': '🔷',
            'action': '⚡',
            'file': '📄',
            'setting': '⚙️',
            'recent': '🕐'
        }
        return icons.get(cmd_type, '•')

    def _on_item_selected(self):
        """Handle Enter key press to execute selected command."""
        current_item = self.results_list.currentItem()
        if current_item:
            cmd = current_item.data(Qt.UserRole)
            self._execute_command(cmd)

    def _on_item_double_clicked(self, item):
        """Handle double-click on item."""
        cmd = item.data(Qt.UserRole)
        self._execute_command(cmd)

    def _execute_command(self, cmd: Dict[str, Any]):
        """Execute the selected command."""
        logger.info(f"Executing command: {cmd['name']} ({cmd['type']})")

        # Call callback if provided
        if 'callback' in cmd and callable(cmd['callback']):
            cmd['callback']()

        # Emit signal with command type and data
        self.command_selected.emit(cmd['type'], cmd.get('data', {}))

        # Close palette
        self.close()

    def keyPressEvent(self, event: QKeyEvent):
        """Handle keyboard navigation."""
        if event.key() == Qt.Key_Escape:
            self.close()
        elif event.key() == Qt.Key_Down:
            current_row = self.results_list.currentRow()
            if current_row < self.results_list.count() - 1:
                self.results_list.setCurrentRow(current_row + 1)
        elif event.key() == Qt.Key_Up:
            current_row = self.results_list.currentRow()
            if current_row > 0:
                self.results_list.setCurrentRow(current_row - 1)
        else:
            super().keyPressEvent(event)

    def eventFilter(self, obj, event):
        """Handle key events from input field."""
        if obj == self.search_input and event.type() == event.KeyPress:
            if event.key() == Qt.Key_Down:
                current_row = self.results_list.currentRow()
                if current_row < self.results_list.count() - 1:
                    self.results_list.setCurrentRow(current_row + 1)
                return True
            elif event.key() == Qt.Key_Up:
                current_row = self.results_list.currentRow()
                if current_row > 0:
                    self.results_list.setCurrentRow(current_row - 1)
                return True
            elif event.key() in (Qt.Key_Return, Qt.Key_Enter):
                self._on_item_selected()
                return True
            elif event.key() == Qt.Key_Escape:
                self.close()
                return True
        return super().eventFilter(obj, event)

    def showEvent(self, event):
        """When shown, position near cursor and focus search."""
        super().showEvent(event)

        # Position near cursor with offset
        from PyQt5.QtGui import QCursor
        from PyQt5.QtWidgets import QApplication

        cursor_pos = QCursor.pos()
        screen_geometry = QApplication.desktop().availableGeometry()

        # Calculate desired position with offset
        desired_x = cursor_pos.x() + CURSOR_OFFSET_X
        desired_y = cursor_pos.y() + CURSOR_OFFSET_Y

        # Ensure palette stays within screen bounds
        palette_width = self.width()
        palette_height = self.height()

        # Clamp to screen bounds
        if desired_x + palette_width > screen_geometry.right():
            desired_x = screen_geometry.right() - palette_width
        if desired_y + palette_height > screen_geometry.bottom():
            desired_y = screen_geometry.bottom() - palette_height
        if desired_x < screen_geometry.left():
            desired_x = screen_geometry.left()
        if desired_y < screen_geometry.top():
            desired_y = screen_geometry.top()

        # Position the palette
        if self.parent():
            local_pos = self.parent().mapFromGlobal(QCursor.pos())
            # Apply the same offset but use global coordinates for bounds checking
            self.move(desired_x, desired_y)
        else:
            self.move(desired_x, desired_y)

        # Clear previous search and focus
        self.search_input.clear()
        self.search_input.setFocus()

    def show_palette(self):
        """Show the command palette."""
        self.show()
        self.raise_()
        self.activateWindow()
