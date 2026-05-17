"""
QSS Stylesheet definitions for Modern DiaBloS UI — V1 "Tightened" redesign.

Same public class (ModernStyles) and entry point (apply_modern_theme), but:
  * toolbar padding and button min-widths are tighter
  * GroupBox top-margin reduced
  * QLineEdit/QSpinBox padding tightened
  * New selectors for the new toolbar widgets:
      QToolButton#TransportPlay / TransportPause / TransportStop / TransportStep
      QLabel#TransportTimeLabel
      QLabel#StatusPill  with dynamic property `state` in {idle, running, paused, error}
      QToolButton#ZoomRockerBtn
      QLabel#ZoomRockerLabel
      QPushButton#CommandPaletteBtn
"""

from modern_ui.themes.theme_manager import theme_manager


class ModernStyles:
    """Modern stylesheet generator for DiaBloS components."""

    @staticmethod
    def _replace_theme_variables(qss: str) -> str:
        theme_vars = theme_manager.get_qss_variables()
        sorted_vars = sorted(theme_vars.items(), key=lambda x: len(x[0]), reverse=True)
        for var, color in sorted_vars:
            qss = qss.replace(var, color)
        return qss

    # -- main window --------------------------------------------------------

    @classmethod
    def get_main_window_style(cls) -> str:
        qss = """
        QMainWindow {
            background-color: @background_primary;
            color: @text_primary;
            font-family: -apple-system, "Segoe UI", "Inter", "Roboto", sans-serif;
            font-size: 10pt;
        }
        QMainWindow::separator {
            background-color: @border_primary;
            width: 1px;
            height: 1px;
        }
        QMainWindow::separator:hover { background-color: @accent_primary; }
        * { font-family: -apple-system, "Segoe UI", "Inter", "Roboto", sans-serif; }
        /* NOTE: Qt's QSS does NOT support `letter-spacing`. All small-caps /
           tracking is applied via QFont.setLetterSpacing(...) in Python. */
        """
        return cls._replace_theme_variables(qss)

    # -- toolbar ------------------------------------------------------------

    @classmethod
    def get_toolbar_style(cls) -> str:
        qss = """
        QToolBar#ModernToolBar {
            background-color: @background_secondary;
            border: none;
            border-bottom: 1px solid @border_primary;
            spacing: 2px;
            padding: 4px 8px;
        }

        QToolBar#ModernToolBar::separator {
            background-color: @border_primary;
            width: 1px;
            margin: 4px 8px;
        }

        /* Generic toolbar button — compact, icon-only */
        QToolBar#ModernToolBar QToolButton {
            background-color: transparent;
            border: 1px solid transparent;
            border-radius: 6px;
            padding: 4px 6px;
            color: @text_primary;
            min-width: 0px;
            min-height: 24px;
        }
        QToolBar#ModernToolBar QToolButton:hover {
            background-color: @background_tertiary;
            border-color: @border_hover;
        }
        QToolBar#ModernToolBar QToolButton:pressed {
            background-color: @accent_pressed;
            color: white;
        }
        QToolBar#ModernToolBar QToolButton:checked {
            background-color: @accent_primary;
            color: white;
            border-color: @accent_secondary;
        }
        QToolBar#ModernToolBar QToolButton:disabled {
            color: @text_disabled;
            background-color: transparent;
            border-color: transparent;
        }

        /* Transport cluster — Play/Pause/Stop/Step */
        QToolButton#TransportPlay,
        QToolButton#TransportPause,
        QToolButton#TransportStop,
        QToolButton#TransportStep {
            background-color: @surface_variant;
            border: 1px solid @border_primary;
            border-radius: 6px;
            padding: 0px;
            min-width: 28px;
            min-height: 26px;
            color: @text_primary;
        }
        QToolButton#TransportPlay:hover,
        QToolButton#TransportPause:hover,
        QToolButton#TransportStop:hover,
        QToolButton#TransportStep:hover {
            background-color: @background_tertiary;
            border-color: @border_hover;
        }
        QToolButton#TransportPlay:disabled,
        QToolButton#TransportPause:disabled,
        QToolButton#TransportStop:disabled,
        QToolButton#TransportStep:disabled {
            background-color: @surface_variant;
            color: @text_disabled;
            border-color: @border_primary;
        }
        QToolButton#TransportPlay:enabled  { color: @success; }
        QToolButton#TransportPause:enabled { color: @warning; }
        QToolButton#TransportStop:enabled  { color: @error; }
        QToolButton#TransportStep:enabled  { color: @text_secondary; }

        QLabel#TransportTimeLabel {
            color: @text_secondary;
            background-color: @surface_variant;
            border: 1px solid @border_primary;
            border-radius: 6px;
            padding: 3px 10px;
        }

        /* Status pill — right side. Pill is a QFrame containing a colored
           dot widget + a QLabel, so the dot's color is independent of text. */
        QFrame#StatusPill {
            border-radius: 11px;
            border: 1px solid @border_primary;
        }
        QFrame#StatusPill[state="idle"]    { background-color: @surface_variant; }
        QFrame#StatusPill[state="running"] { background-color: @success_bg; border-color: @success; }
        QFrame#StatusPill[state="paused"]  { background-color: @warning_bg; border-color: @warning; }
        QFrame#StatusPill[state="error"]   { background-color: @error_bg;   border-color: @error;   }

        QFrame#StatusPill QLabel#StatusPillLabel {
            background: transparent; border: none;
            font-weight: 500; padding: 0;
        }
        QFrame#StatusPill[state="idle"]    QLabel#StatusPillLabel { color: @text_secondary; }
        QFrame#StatusPill[state="running"] QLabel#StatusPillLabel { color: @success; }
        QFrame#StatusPill[state="paused"]  QLabel#StatusPillLabel { color: @warning; }
        QFrame#StatusPill[state="error"]   QLabel#StatusPillLabel { color: @error;   }

        /* Zoom rocker */
        QToolButton#ZoomRockerBtn {
            background-color: transparent;
            border: 1px solid transparent;
            border-radius: 4px;
            min-width: 22px;
            min-height: 22px;
        }
        QToolButton#ZoomRockerBtn:hover {
            background-color: @background_tertiary;
            border-color: @border_hover;
        }
        QLabel#ZoomRockerLabel {
            color: @text_secondary;
            padding: 0 4px;
            min-width: 40px;
        }

        /* Command-palette pill */
        QPushButton#CommandPaletteBtn {
            background-color: @surface_variant;
            border: 1px solid @border_primary;
            border-radius: 6px;
            padding: 3px 10px;
            color: @text_secondary;
            font-weight: 400;
            min-width: 100px;
            min-height: 24px;
            text-align: left;
        }
        QPushButton#CommandPaletteBtn:hover {
            background-color: @background_tertiary;
            border-color: @border_hover;
            color: @text_primary;
        }
        """
        return cls._replace_theme_variables(qss)

    # -- splitter -----------------------------------------------------------

    @classmethod
    def get_splitter_style(cls) -> str:
        qss = """
        QSplitter { background-color: @background_primary; }
        QSplitter::handle { background-color: @border_primary; }
        QSplitter::handle:horizontal { width: 4px; margin: 2px 0px; }
        QSplitter::handle:vertical   { height: 4px; margin: 0px 2px; }
        QSplitter::handle:hover { background-color: @accent_primary; }
        """
        return cls._replace_theme_variables(qss)

    # -- panel --------------------------------------------------------------

    @classmethod
    def get_panel_style(cls) -> str:
        qss = """
        QFrame#ModernPanel {
            background-color: @surface;
            border: 1px solid @border_primary;
            border-radius: 8px;
            padding: 8px;
        }
        QLabel#PanelTitle {
            color: @text_primary;
            font-weight: 600;
            font-size: 12pt;
            padding: 6px 4px;
        }
        QGroupBox {
            font-weight: 600;
            color: @text_primary;
            border: 1px solid @border_primary;
            border-radius: 6px;
            margin-top: 12px;
            padding-top: 4px;
            background-color: @surface_variant;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 10px;
            padding: 2px 8px;
            background-color: @surface;
            border-radius: 3px;
            color: @text_primary;
            font-size: 9pt;
        }
        QScrollArea { border: none; background-color: transparent; }
        QScrollBar:vertical {
            background-color: @surface_variant;
            width: 10px;
            border-radius: 5px;
            margin: 0px;
        }
        QScrollBar::handle:vertical {
            background-color: @border_secondary;
            border-radius: 5px;
            min-height: 24px;
        }
        QScrollBar::handle:vertical:hover { background-color: @border_hover; }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
        """
        return cls._replace_theme_variables(qss)

    # -- buttons / inputs ---------------------------------------------------

    @classmethod
    def get_button_style(cls) -> str:
        qss = """
        QPushButton {
            background-color: @surface;
            border: 1px solid @border_primary;
            border-radius: 6px;
            padding: 5px 12px;
            color: @text_primary;
            font-weight: 500;
            min-width: 64px;
            min-height: 28px;
        }
        QPushButton:hover { background-color: @background_tertiary; border-color: @border_hover; }
        QPushButton:pressed { background-color: @accent_pressed; color: white; }
        QPushButton:default {
            background-color: @accent_primary;
            color: white;
            border-color: @accent_secondary;
            font-weight: 600;
        }
        QPushButton:default:hover { background-color: @accent_hover; }
        QPushButton:default:pressed { background-color: @accent_pressed; }
        QPushButton:disabled {
            background-color: @background_secondary;
            color: @text_disabled;
            border-color: @border_primary;
        }

        QLineEdit, QTextEdit, QPlainTextEdit {
            background-color: @surface;
            border: 1px solid @border_primary;
            border-radius: 5px;
            padding: 5px 9px;
            color: @text_primary;
            selection-background-color: @accent_primary;
            selection-color: white;
        }
        QSpinBox, QDoubleSpinBox {
            background-color: @surface;
            border: 1px solid @border_primary;
            border-radius: 5px;
            padding: 4px 18px 4px 6px;
            color: @text_primary;
            selection-background-color: @accent_primary;
            selection-color: white;
            min-height: 22px;
        }
        QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus,
        QSpinBox:focus, QDoubleSpinBox:focus {
            color: @text_primary;
            border-color: @border_focus;
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

        QComboBox {
            background-color: @surface;
            border: 1px solid @border_primary;
            border-radius: 5px;
            padding: 5px 10px;
            color: @text_primary;
            min-height: 22px;
        }
        QComboBox:hover { border-color: @border_hover; }
        QComboBox:focus { border-color: @border_focus; }
        QComboBox::drop-down { border: none; width: 20px; }
        QComboBox::down-arrow {
            image: none;
            border-left: 3.5px solid transparent;
            border-right: 3.5px solid transparent;
            border-top: 5px solid @text_secondary;
            margin-right: 8px;
        }
        QComboBox QAbstractItemView {
            background-color: @surface;
            border: 1px solid @border_primary;
            border-radius: 5px;
            padding: 4px;
            selection-background-color: @accent_primary;
            selection-color: white;
        }
        """
        return cls._replace_theme_variables(qss)

    # -- status / menus -----------------------------------------------------

    @classmethod
    def get_statusbar_style(cls) -> str:
        qss = """
        QStatusBar {
            background-color: @background_secondary;
            border-top: 1px solid @border_primary;
            color: @text_secondary;
            font-size: 9pt;
            padding: 3px 8px;
        }
        QStatusBar::item { border: none; padding: 2px 8px; }
        QStatusBar QLabel { color: @text_secondary; padding: 2px 8px; }
        """
        return cls._replace_theme_variables(qss)

    @classmethod
    def get_menubar_style(cls) -> str:
        qss = """
        QMenuBar {
            background-color: @background_secondary;
            border-bottom: 1px solid @border_primary;
            color: @text_primary;
            font-weight: 500;
            padding: 4px 6px;
        }
        QMenuBar::item {
            background-color: transparent;
            padding: 6px 10px;
            border-radius: 5px;
            margin: 0px 1px;
        }
        QMenuBar::item:selected { background-color: @background_tertiary; }
        QMenuBar::item:pressed  { background-color: @accent_primary; color: white; }

        QMenu {
            background-color: @surface_elevated;
            border: 1px solid @border_primary;
            border-radius: 6px;
            padding: 4px;
            color: @text_primary;
        }
        QMenu::item {
            padding: 6px 28px 6px 22px;
            border-radius: 4px;
            margin: 1px 3px;
        }
        QMenu::item:selected { background-color: @accent_primary; color: white; }
        QMenu::item:disabled { color: @text_disabled; }
        QMenu::separator { height: 1px; background-color: @border_primary; margin: 4px 10px; }
        QMenu::icon { padding-left: 6px; }
        QMenu::indicator { width: 16px; height: 16px; left: 6px; }
        """
        return cls._replace_theme_variables(qss)

    @classmethod
    def get_complete_stylesheet(cls) -> str:
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
    stylesheet = ModernStyles.get_complete_stylesheet()
    app.setStyleSheet(stylesheet)

    def on_theme_changed():
        app.setStyleSheet(ModernStyles.get_complete_stylesheet())

    theme_manager.theme_changed.connect(on_theme_changed)
