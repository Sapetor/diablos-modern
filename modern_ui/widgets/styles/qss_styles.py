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
        """Get stylesheet for main window."""
        qss = """
        QMainWindow {
            background-color: @background_primary;
            color: @text_primary;
            font-family: "Segoe UI", "San Francisco", "Roboto", sans-serif;
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
        """
        return cls._replace_theme_variables(qss)
    
    @classmethod
    def get_toolbar_style(cls) -> str:
        """Get stylesheet for modern toolbar."""
        qss = """
        QToolBar {
            background-color: @background_secondary;
            border: none;
            spacing: 4px;
            padding: 6px;
            font-weight: 500;
        }
        
        QToolBar::separator {
            background-color: @border_primary;
            width: 1px;
            margin: 4px 8px;
        }
        
        QToolButton {
            background-color: transparent;
            border: 1px solid transparent;
            border-radius: 6px;
            padding: 8px 12px;
            color: @text_primary;
            font-weight: 500;
            text-align: center;
            min-width: 60px;
        }
        
        QToolButton:hover {
            background-color: @background_tertiary;
            border-color: @border_secondary;
        }
        
        QToolButton:pressed {
            background-color: @accent_secondary;
            color: white;
        }
        
        QToolButton:checked {
            background-color: @accent_primary;
            color: white;
            border-color: @accent_secondary;
        }
        
        QToolButton:disabled {
            color: @text_disabled;
            background-color: transparent;
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
        """Get stylesheet for side panels."""
        qss = """
        .ModernPanel {
            background-color: @background_secondary;
            border: 1px solid @border_primary;
            border-radius: 8px;
            padding: 8px;
        }
        
        .ModernPanel QLabel {
            color: @text_primary;
            font-weight: 600;
            font-size: 11pt;
        }
        
        .ModernPanel QGroupBox {
            font-weight: 600;
            color: @text_primary;
            border: 1px solid @border_primary;
            border-radius: 6px;
            margin-top: 12px;
            padding-top: 4px;
        }
        
        .ModernPanel QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 8px 0 8px;
            background-color: @background_secondary;
        }
        """
        return cls._replace_theme_variables(qss)
    
    @classmethod
    def get_button_style(cls) -> str:
        """Get stylesheet for modern buttons."""
        qss = """
        QPushButton {
            background-color: @surface;
            border: 1px solid @border_primary;
            border-radius: 6px;
            padding: 8px 16px;
            color: @text_primary;
            font-weight: 500;
            min-width: 80px;
            min-height: 24px;
        }
        
        QPushButton:hover {
            background-color: @background_tertiary;
            border-color: @border_secondary;
        }
        
        QPushButton:pressed {
            background-color: @accent_secondary;
            color: white;
        }
        
        QPushButton:default {
            background-color: @accent_primary;
            color: white;
            border-color: @accent_secondary;
        }
        
        QPushButton:default:hover {
            background-color: @accent_hover;
        }
        
        QPushButton:disabled {
            background-color: @background_secondary;
            color: @text_disabled;
            border-color: @border_primary;
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
        """Get stylesheet for menu bar."""
        qss = """
        QMenuBar {
            background-color: @background_secondary;
            border-bottom: 1px solid @border_primary;
            color: @text_primary;
            font-weight: 500;
            padding: 4px;
        }
        
        QMenuBar::item {
            background-color: transparent;
            padding: 6px 12px;
            border-radius: 4px;
            margin: 2px;
        }
        
        QMenuBar::item:selected {
            background-color: @background_tertiary;
        }
        
        QMenuBar::item:pressed {
            background-color: @accent_primary;
            color: white;
        }
        
        QMenu {
            background-color: @surface;
            border: 1px solid @border_primary;
            border-radius: 6px;
            padding: 4px;
            color: @text_primary;
        }
        
        QMenu::item {
            padding: 6px 20px;
            border-radius: 4px;
            margin: 1px;
        }
        
        QMenu::item:selected {
            background-color: @accent_primary;
            color: white;
        }
        
        QMenu::separator {
            height: 1px;
            background-color: @border_primary;
            margin: 4px 8px;
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