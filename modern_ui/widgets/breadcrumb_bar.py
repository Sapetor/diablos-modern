
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QPushButton, QLabel, QStyle
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QIcon
import logging

logger = logging.getLogger(__name__)

class BreadcrumbBar(QWidget):
    """
    A widget that displays the current hierarchy path (e.g., Main > Subsystem1 > Subsystem2).
    Allows navigation by clicking on path components.
    """
    path_clicked = pyqtSignal(int)  # Emits the index of the clicked path component

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(30)
        
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(5, 0, 5, 0)
        self.layout.setSpacing(2)
        self.layout.setAlignment(Qt.AlignLeft)
        
        # Initial path
        self.set_path(["Main"])

    def set_path(self, path_list):
        """
        Update the breadcrumb trail.
        Args:
            path_list: List of names representing the path (e.g. ['Main', 'Subsystem1'])
        """
        # Clear current items
        while self.layout.count():
            item = self.layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for i, name in enumerate(path_list):
            # Separator (except for first item)
            if i > 0:
                sep = QLabel(">")
                sep.setStyleSheet("color: #888; font-weight: bold; margin: 0 4px;")
                self.layout.addWidget(sep)

            # Button
            btn = QPushButton(name)
            btn.setFlat(True)
            btn.setCursor(Qt.PointingHandCursor)
            
            # Styling
            is_last = (i == len(path_list) - 1)
            font_weight = "bold" if is_last else "normal"
            
            theme_colors = theme_manager.get_current_theme()
            color = theme_colors['text_primary'] if is_last else theme_colors['text_secondary']
            hover_color = theme_colors['accent_primary']
            
            btn.setStyleSheet(f"""
                QPushButton {{
                    border: none;
                    background: transparent;
                    color: {color};
                    font-weight: {font_weight};
                    text-align: left;
                    padding: 2px 5px;
                    border-radius: 4px;
                }}
                QPushButton:hover {{
                    background: rgba(128, 128, 128, 0.1);
                    color: {hover_color};
                }}
            """)
            
            # Store index for signal
            # Use default arg to capture 'i' value
            btn.clicked.connect(lambda checked, idx=i: self.path_clicked.emit(idx))
            
            self.layout.addWidget(btn)

        # Add stretch at end to push items left
        self.layout.addStretch()

# Import needs to be late to avoid circular dep availability or if theme_manager is globally available
from modern_ui.themes.theme_manager import theme_manager
