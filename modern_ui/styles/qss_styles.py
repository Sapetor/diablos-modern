"""
QSS Stylesheet definitions for Modern DiaBloS UI
Provides consistent styling across all Qt widgets.
"""

from typing import Dict
from modern_ui.themes.theme_manager import theme_manager


class ModernStyles:
    """Modern stylesheet generator for DiaBloS components."""
    
    @staticmethod
    def _replace_theme_variables(qss: str) -> str:
        """Replace theme variables in QSS with actual colors."""
        theme_vars = theme_manager.get_qss_variables()
        for var, color in theme_vars.items():
            qss = qss.replace(var, color)
        return qss
    
    @classmethod
    def get_main_window_style(cls) -> str:
        """Get stylesheet for main window with improved spacing and depth."""
        qss = """
        QMainWindow {
            background-color: @background_primary;
            color: @text_primary;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "SF Pro Display", "Roboto", sans-serif;
            font-size: 10pt;
        }

        QMainWindow::separator {
            background-color: @border_primary;
            width: 1px;
            height: 1px;
        }

        QMainWindow::separator:hover {
            background-color: @accent_primary;
        }

        /* Typography improvements */
        * {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "SF Pro Display", "Roboto", sans-serif;
        }
        """
        return cls._replace_theme_variables(qss)
    
    @classmethod
    def get_toolbar_style(cls) -> str:
        """Get stylesheet for modern toolbar with improved visual hierarchy."""
        qss = """
        QToolBar {
            background-color: @background_secondary;
            border: none;
            border-bottom: 1px solid @border_primary;
            spacing: 6px;
            padding: 8px 12px;
            font-weight: 500;
        }

        QToolBar::separator {
            background-color: @border_primary;
            width: 1px;
            margin: 6px 10px;
        }

        QToolButton {
            background-color: transparent;
            border: 1px solid transparent;
            border-radius: 8px;
            padding: 10px 14px;
            color: @text_primary;
            font-weight: 500;
            text-align: center;
            min-width: 60px;
            transition: all 0.2s ease;
        }

        QToolButton:hover {
            background-color: @background_tertiary;
            border-color: @border_hover;
        }

        QToolButton:pressed {
            background-color: @accent_pressed;
            color: white;
            transform: scale(0.98);
        }

        QToolButton:checked {
            background-color: @accent_primary;
            color: white;
            border-color: @accent_secondary;
        }

        QToolButton:disabled {
            color: @text_disabled;
            background-color: transparent;
            opacity: 0.5;
        }
        """
        return cls._replace_theme_variables(qss)
    
    @classmethod
    def get_splitter_style(cls) -> str:
        """Get stylesheet for splitter widgets."""
        qss = """
        QSplitter {
            background-color: @background_primary;
        }
        
        QSplitter::handle {
            background-color: @border_primary;
        }
        
        QSplitter::handle:horizontal {
            width: 2px;
            margin: 2px 0px;
        }
        
        QSplitter::handle:vertical {
            height: 2px;
            margin: 0px 2px;
        }
        
        QSplitter::handle:hover {
            background-color: @accent_primary;
        }
        """
        return cls._replace_theme_variables(qss)
    
    @classmethod
    def get_panel_style(cls) -> str:
        """Get stylesheet for side panels with elevation and improved spacing."""
        qss = """
        QFrame#ModernPanel {
            background-color: @surface;
            border: 1px solid @border_primary;
            border-radius: 10px;
            padding: 12px;
        }

        QLabel#PanelTitle {
            color: @text_primary;
            font-weight: 600;
            font-size: 13pt;
            padding: 8px 4px;
            letter-spacing: -0.3px;
        }

        QGroupBox {
            font-weight: 600;
            color: @text_primary;
            border: 1px solid @border_primary;
            border-radius: 8px;
            margin-top: 16px;
            padding-top: 8px;
            background-color: @surface_variant;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 12px;
            padding: 4px 12px;
            background-color: @surface;
            border-radius: 4px;
            color: @text_primary;
            font-size: 10pt;
        }

        /* Scrollbar styling for panels */
        QScrollArea {
            border: none;
            background-color: transparent;
        }

        QScrollBar:vertical {
            background-color: @surface_variant;
            width: 12px;
            border-radius: 6px;
            margin: 0px;
        }

        QScrollBar::handle:vertical {
            background-color: @border_secondary;
            border-radius: 6px;
            min-height: 30px;
        }

        QScrollBar::handle:vertical:hover {
            background-color: @border_hover;
        }

        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }

        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
            background: none;
        }
        """
        return cls._replace_theme_variables(qss)
    
    @classmethod
    def get_button_style(cls) -> str:
        """Get stylesheet for modern buttons with improved visual feedback."""
        qss = """
        QPushButton {
            background-color: @surface;
            border: 1px solid @border_primary;
            border-radius: 8px;
            padding: 10px 20px;
            color: @text_primary;
            font-weight: 500;
            min-width: 80px;
            min-height: 32px;
            transition: all 0.2s ease;
        }

        QPushButton:hover {
            background-color: @background_tertiary;
            border-color: @border_hover;
        }

        QPushButton:pressed {
            background-color: @accent_pressed;
            color: white;
        }

        QPushButton:default {
            background-color: @accent_primary;
            color: white;
            border-color: @accent_secondary;
            font-weight: 600;
        }

        QPushButton:default:hover {
            background-color: @accent_hover;
        }

        QPushButton:default:pressed {
            background-color: @accent_pressed;
        }

        QPushButton:disabled {
            background-color: @background_secondary;
            color: @text_disabled;
            border-color: @border_primary;
            opacity: 0.5;
        }

        /* Input fields */
        QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox {
            background-color: @surface;
            border: 1px solid @border_primary;
            border-radius: 6px;
            padding: 8px 12px;
            color: @text_primary;
            selection-background-color: @accent_primary;
            selection-color: white;
        }

        QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus,
        QSpinBox:focus, QDoubleSpinBox:focus {
            border-color: @border_focus;
            border-width: 2px;
            padding: 7px 11px;
        }

        QLineEdit:hover, QTextEdit:hover, QPlainTextEdit:hover,
        QSpinBox:hover, QDoubleSpinBox:hover {
            border-color: @border_hover;
        }

        QLineEdit:disabled, QTextEdit:disabled, QPlainTextEdit:disabled,
        QSpinBox:disabled, QDoubleSpinBox:disabled {
            background-color: @background_secondary;
            color: @text_disabled;
            border-color: @border_primary;
        }

        /* ComboBox */
        QComboBox {
            background-color: @surface;
            border: 1px solid @border_primary;
            border-radius: 6px;
            padding: 8px 12px;
            color: @text_primary;
            min-height: 24px;
        }

        QComboBox:hover {
            border-color: @border_hover;
        }

        QComboBox:focus {
            border-color: @border_focus;
            border-width: 2px;
        }

        QComboBox::drop-down {
            border: none;
            width: 24px;
        }

        QComboBox::down-arrow {
            image: none;
            border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-top: 6px solid @text_secondary;
            margin-right: 8px;
        }

        QComboBox QAbstractItemView {
            background-color: @surface;
            border: 1px solid @border_primary;
            border-radius: 6px;
            padding: 4px;
            selection-background-color: @accent_primary;
            selection-color: white;
        }
        """
        return cls._replace_theme_variables(qss)
    
    @classmethod
    def get_statusbar_style(cls) -> str:
        """Get stylesheet for status bar."""
        qss = """
        QStatusBar {
            background-color: @background_secondary;
            border-top: 1px solid @border_primary;
            color: @text_secondary;
            font-size: 9pt;
            padding: 4px 8px;
        }
        
        QStatusBar::item {
            border: none;
            padding: 2px 8px;
        }
        
        QStatusBar QLabel {
            color: @text_secondary;
            padding: 2px 8px;
        }
        """
        return cls._replace_theme_variables(qss)
    
    @classmethod
    def get_menubar_style(cls) -> str:
        """Get stylesheet for menu bar with refined styling."""
        qss = """
        QMenuBar {
            background-color: @background_secondary;
            border-bottom: 1px solid @border_primary;
            color: @text_primary;
            font-weight: 500;
            padding: 6px 8px;
        }

        QMenuBar::item {
            background-color: transparent;
            padding: 8px 14px;
            border-radius: 6px;
            margin: 0px 2px;
        }

        QMenuBar::item:selected {
            background-color: @background_tertiary;
        }

        QMenuBar::item:pressed {
            background-color: @accent_primary;
            color: white;
        }

        QMenu {
            background-color: @surface_elevated;
            border: 1px solid @border_primary;
            border-radius: 8px;
            padding: 6px;
            color: @text_primary;
        }

        QMenu::item {
            padding: 8px 32px 8px 24px;
            border-radius: 6px;
            margin: 2px 4px;
        }

        QMenu::item:selected {
            background-color: @accent_primary;
            color: white;
        }

        QMenu::item:disabled {
            color: @text_disabled;
        }

        QMenu::separator {
            height: 1px;
            background-color: @border_primary;
            margin: 6px 12px;
        }

        QMenu::icon {
            padding-left: 8px;
        }

        QMenu::indicator {
            width: 16px;
            height: 16px;
            left: 8px;
        }
        """
        return cls._replace_theme_variables(qss)
    
    @classmethod
    def get_complete_stylesheet(cls) -> str:
        """Get complete application stylesheet."""
        styles = [
            cls.get_main_window_style(),
            cls.get_toolbar_style(),
            cls.get_splitter_style(),
            cls.get_panel_style(),
            cls.get_button_style(),
            cls.get_statusbar_style(),
            cls.get_menubar_style(),
        ]
        return "\n\n".join(styles)


def apply_modern_theme(app):
    """Apply modern theme to the entire application."""
    stylesheet = ModernStyles.get_complete_stylesheet()
    app.setStyleSheet(stylesheet)
    
    # Update when theme changes
    def on_theme_changed():
        new_stylesheet = ModernStyles.get_complete_stylesheet()
        app.setStyleSheet(new_stylesheet)
    
    theme_manager.theme_changed.connect(on_theme_changed)