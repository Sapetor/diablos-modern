"""Error Panel Widget
Displays validation errors and warnings in a collapsible panel.
"""

import logging
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QFrame, QMenu, QApplication
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QCursor
from modern_ui.themes.theme_manager import theme_manager
from lib.diagram_validator import ErrorSeverity

logger = logging.getLogger(__name__)


class ErrorItemWidget(QFrame):
    """Widget representing a single error/warning item."""

    clicked = pyqtSignal(object)  # Emits the error object when clicked

    def __init__(self, error, parent=None):
        super().__init__(parent)
        self.error = error
        self._setup_ui()
        self._apply_styling()

    def _setup_ui(self):
        """Setup the UI components."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(10)

        # Severity icon
        severity_label = QLabel()
        if self.error.severity == ErrorSeverity.ERROR:
            severity_label.setText("🔴")
            severity_label.setToolTip("Error")
        elif self.error.severity == ErrorSeverity.WARNING:
            severity_label.setText("🟡")
            severity_label.setToolTip("Warning")
        else:
            severity_label.setText("ℹ️")
            severity_label.setToolTip("Info")
        severity_label.setFixedWidth(24)
        layout.addWidget(severity_label)

        # Error message
        message_label = QLabel(self.error.message)
        message_label.setWordWrap(True)
        message_label.setFont(QFont("Segoe UI", 9))
        layout.addWidget(message_label, 1)

        # Make clickable
        self.setCursor(Qt.PointingHandCursor)

    def _apply_styling(self):
        """Apply theme-aware styling."""
        bg_color = theme_manager.get_color('surface_secondary')
        text_color = theme_manager.get_color('text_primary')
        border_color = theme_manager.get_color('border_primary')

        # Different background for different severities
        if self.error.severity == ErrorSeverity.ERROR:
            accent = "rgba(220, 53, 69, 0.1)"
        elif self.error.severity == ErrorSeverity.WARNING:
            accent = "rgba(255, 193, 7, 0.1)"
        else:
            accent = "rgba(23, 162, 184, 0.1)"

        self.setStyleSheet(f"""
            ErrorItemWidget {{
                background-color: {accent};
                border: 1px solid {border_color.name()};
                border-radius: 4px;
                margin: 2px 0px;
            }}
            ErrorItemWidget:hover {{
                background-color: {theme_manager.get_color('surface_primary').name()};
                border: 1px solid {theme_manager.get_color('accent_primary').name()};
            }}
            QLabel {{
                color: {text_color.name()};
                background: transparent;
                border: none;
            }}
        """)

    def mousePressEvent(self, event):
        """Handle mouse press - emit clicked signal."""
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.error)
        super().mousePressEvent(event)

    def contextMenuEvent(self, event):
        """Show context menu with copy option."""
        menu = QMenu(self)
        copy_action = menu.addAction("Copy Message")
        copy_action.triggered.connect(self._copy_to_clipboard)
        menu.exec_(QCursor.pos())

    def _copy_to_clipboard(self):
        """Copy the error message to clipboard."""
        clipboard = QApplication.clipboard()
        severity_text = self.error.severity.name if hasattr(self.error.severity, 'name') else str(self.error.severity)
        text = f"[{severity_text}] {self.error.message}"
        if hasattr(self.error, 'block_name') and self.error.block_name:
            text += f" (Block: {self.error.block_name})"
        clipboard.setText(text)


class ErrorPanel(QWidget):
    """Panel displaying validation errors and warnings."""

    error_clicked = pyqtSignal(object)  # Emitted when an error is clicked

    def __init__(self, parent=None):
        super().__init__(parent)
        self.errors = []
        self._is_collapsed = False
        self._setup_ui()
        self._apply_styling()

        # Connect to theme changes
        theme_manager.theme_changed.connect(self._on_theme_changed)

    def _setup_ui(self):
        """Setup the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QFrame()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(10, 8, 10, 8)

        # Title
        self.title_label = QLabel("Validation Results")
        self.title_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        header_layout.addWidget(self.title_label)

        # Error count badge
        self.count_label = QLabel("0 issues")
        self.count_label.setFont(QFont("Segoe UI", 9))
        header_layout.addWidget(self.count_label)

        header_layout.addStretch()

        # Copy all button
        self.copy_btn = QPushButton("📋")
        self.copy_btn.setFixedSize(24, 24)
        self.copy_btn.setToolTip("Copy all messages")
        self.copy_btn.clicked.connect(self._copy_all_to_clipboard)
        header_layout.addWidget(self.copy_btn)

        # Collapse/expand button
        self.collapse_btn = QPushButton("▼")
        self.collapse_btn.setFixedSize(24, 24)
        self.collapse_btn.clicked.connect(self.toggle_collapse)
        header_layout.addWidget(self.collapse_btn)

        # Clear button
        clear_btn = QPushButton("✕")
        clear_btn.setFixedSize(24, 24)
        clear_btn.setToolTip("Close panel")
        clear_btn.clicked.connect(self.hide)
        header_layout.addWidget(clear_btn)

        layout.addWidget(header)

        # Scroll area for errors
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Content widget
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(5, 5, 5, 5)
        self.content_layout.setSpacing(2)
        self.content_layout.addStretch()

        scroll.setWidget(self.content_widget)
        layout.addWidget(scroll)

        # Initially hidden
        self.hide()

    def _apply_styling(self):
        """Apply theme-aware styling."""
        bg_color = theme_manager.get_color('surface_primary')
        text_color = theme_manager.get_color('text_primary')
        border_color = theme_manager.get_color('border_primary')

        self.setStyleSheet(f"""
            ErrorPanel {{
                background-color: {bg_color.name()};
                border: 1px solid {border_color.name()};
                border-radius: 6px;
            }}
            QLabel {{
                color: {text_color.name()};
            }}
            QPushButton {{
                background-color: transparent;
                color: {text_color.name()};
                border: 1px solid {border_color.name()};
                border-radius: 4px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {theme_manager.get_color('surface_secondary').name()};
                border: 1px solid {theme_manager.get_color('accent_primary').name()};
            }}
            QScrollArea {{
                border: none;
                background-color: transparent;
            }}
        """)

    def set_errors(self, errors):
        """
        Set the errors to display.

        Args:
            errors: List of ValidationError objects
        """
        self.errors = errors

        # Clear existing error widgets
        while self.content_layout.count() > 1:  # Keep the stretch
            item = self.content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Add new error widgets
        for error in errors:
            error_widget = ErrorItemWidget(error)
            error_widget.clicked.connect(self.error_clicked.emit)
            self.content_layout.insertWidget(self.content_layout.count() - 1, error_widget)

        # Update counts
        error_count = sum(1 for e in errors if e.severity == ErrorSeverity.ERROR)
        warning_count = sum(1 for e in errors if e.severity == ErrorSeverity.WARNING)

        if error_count > 0 and warning_count > 0:
            self.count_label.setText(f"{error_count} errors, {warning_count} warnings")
        elif error_count > 0:
            self.count_label.setText(f"{error_count} error{'s' if error_count != 1 else ''}")
        elif warning_count > 0:
            self.count_label.setText(f"{warning_count} warning{'s' if warning_count != 1 else ''}")
        else:
            self.count_label.setText("No issues")

        # Show panel if there are errors
        if errors:
            self.show()
        else:
            self.hide()

    def toggle_collapse(self):
        """Toggle panel collapsed/expanded state."""
        self._is_collapsed = not self._is_collapsed

        if self._is_collapsed:
            self.content_widget.hide()
            self.collapse_btn.setText("▶")
        else:
            self.content_widget.show()
            self.collapse_btn.setText("▼")

    def clear(self):
        """Clear all errors and hide panel."""
        self.set_errors([])
        self.hide()

    def _on_theme_changed(self):
        """Handle theme changes."""
        self._apply_styling()
        # Re-create error widgets to apply new theme colors
        if self.errors:
            self.set_errors(self.errors)

    def _copy_all_to_clipboard(self):
        """Copy all error/warning messages to clipboard."""
        if not self.errors:
            return

        lines = []
        for error in self.errors:
            severity_text = error.severity.name if hasattr(error.severity, 'name') else str(error.severity)
            text = f"[{severity_text}] {error.message}"
            if hasattr(error, 'block_name') and error.block_name:
                text += f" (Block: {error.block_name})"
            lines.append(text)

        clipboard = QApplication.clipboard()
        clipboard.setText("\n".join(lines))
