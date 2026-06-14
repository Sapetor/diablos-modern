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

from PyQt5.QtGui import QPalette, QColor

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
            font-size: @font_body_strong;
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
            spacing: @space_xs;
            padding: @space_sm @space_md;
        }

        QToolBar#ModernToolBar::separator {
            background-color: @border_primary;
            width: 1px;
            margin: @space_sm @space_md;
        }

        /* Generic toolbar button — compact, icon-only */
        QToolBar#ModernToolBar QToolButton {
            background-color: transparent;
            border: 1px solid transparent;
            border-radius: @radius_md;
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
            border-radius: @radius_md;
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
        /* Keyboard-focus ring for the transport cluster — 2px accent border,
           distinct from the 1px @border_hover hover edge, for a11y. */
        QToolButton#TransportPlay:focus,
        QToolButton#TransportPause:focus,
        QToolButton#TransportStop:focus,
        QToolButton#TransportStep:focus {
            border: 2px solid @border_focus;
            border-radius: @radius_md;
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
            border-radius: @radius_md;
            padding: 3px 10px;
        }

        /* Status pill — right side. Pill is a QFrame containing a colored
           dot widget + a QLabel, so the dot's color is independent of text. */
        QFrame#StatusPill {
            border-radius: @radius_pill;
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
            border-radius: @radius_sm;
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
            border-radius: @radius_md;
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
        /* Keyboard-focus ring — 2px accent border, distinct from the 1px
           @border_hover hover edge, so the palette pill is visible on tab. */
        QPushButton#CommandPaletteBtn:focus {
            border: 2px solid @border_focus;
            border-radius: @radius_md;
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
        QDialog,
        QMessageBox,
        QInputDialog,
        QFileDialog {
            background-color: @background_secondary;
            color: @text_primary;
        }

        /* QLabel / QCheckBox color does not inherit from QDialog reliably in
           Qt's QSS — scope explicit rules to dialog/groupbox descendants so
           the text stays readable in dark mode. QMessageBox / QInputDialog /
           QFileDialog need explicit selectors on Windows: the QDialog QLabel
           descendant rule alone is not picked up for their internal labels. */
        QDialog QLabel,
        QGroupBox QLabel,
        QMessageBox QLabel,
        QInputDialog QLabel,
        QFileDialog QLabel {
            color: @text_primary;
            background: transparent;
        }
        QDialog QCheckBox,
        QGroupBox QCheckBox,
        QMessageBox QCheckBox,
        QInputDialog QCheckBox,
        QFileDialog QCheckBox {
            color: @text_primary;
            background: transparent;
        }
        QFileDialog QListView,
        QFileDialog QTreeView,
        QFileDialog QToolButton {
            color: @text_primary;
            background-color: @surface;
        }

        /* Hint / muted labels inside dialogs (use objectName="HintLabel" or
           the inline italic class to opt in — see SimulationDialog hint). */
        QLabel#HintLabel {
            color: @text_secondary;
            font-size: @font_body_strong;
            font-style: italic;
        }

        QFrame#ModernPanel {
            background-color: @surface;
            border: 1px solid @border_primary;
            border-radius: @radius_lg;
            padding: @space_md;
        }
        QLabel#PanelTitle {
            color: @text_primary;
            font-weight: 600;
            font-size: @font_title;
            padding: 6px 4px;
        }
        QGroupBox {
            font-weight: 600;
            color: @text_primary;
            border: 1px solid @border_primary;
            border-radius: @radius_md;
            margin-top: @space_lg;
            padding-top: @space_sm;
            background-color: @surface_variant;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 10px;
            padding: 2px 8px;
            background-color: @surface;
            border-radius: @radius_sm;
            color: @text_primary;
            font-size: @font_body;
        }
        QScrollArea { border: none; background-color: transparent; }
        QScrollBar:vertical {
            background-color: @surface_variant;
            width: 10px;
            border-radius: @radius_md;
            margin: 0px;
        }
        QScrollBar::handle:vertical {
            background-color: @border_secondary;
            border-radius: @radius_md;
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
            border-radius: @radius_md;
            padding: 5px 12px;
            color: @text_primary;
            font-weight: 500;
            min-width: 64px;
            min-height: 28px;
        }
        QPushButton:hover { background-color: @background_tertiary; border-color: @border_hover; }
        QPushButton:pressed { background-color: @accent_pressed; color: white; }
        /* Keyboard-focus ring — a 2px accent border (distinct from the 1px
           @border_hover hover edge) so tab-navigation is visible for a11y. */
        QPushButton:focus {
            border: 2px solid @border_focus;
            border-radius: @radius_md;
        }
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
            border-radius: @radius_md;
            padding: 5px 9px;
            color: @text_primary;
            selection-background-color: @accent_primary;
            selection-color: white;
        }
        QSpinBox, QDoubleSpinBox {
            background-color: @surface;
            border: 1px solid @border_primary;
            border-radius: @radius_md;
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
            border-radius: @radius_md;
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
            border-radius: @radius_md;
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
            font-size: @font_body;
            padding: 3px 8px;
        }
        QStatusBar::item { border: none; padding: 2px 8px; }
        QStatusBar QLabel { color: @text_secondary; padding: 2px 8px; }
        """
        return cls._replace_theme_variables(qss)

    @classmethod
    def get_menubar_style(cls) -> str:
        qss = """
        /* Top-level menubar — matches the toolbar's tightened look. */
        QMenuBar {
            background-color: @background_secondary;
            border-bottom: 1px solid @border_primary;
            color: @text_primary;
            font-weight: 500;
            padding: 3px 6px;
        }
        QMenuBar::item {
            background-color: transparent;
            padding: 6px 10px;
            border-radius: @radius_md;
            margin: 0px 1px;
            color: @text_secondary;
        }
        QMenuBar::item:selected { background-color: @surface_elevated; color: @text_primary; }
        QMenuBar::item:pressed  { background-color: @surface_elevated; color: @text_primary; }

        /* Dropdown menus + context menus */
        QMenu {
            background-color: @surface_elevated;
            border: 1px solid @border_primary;
            border-radius: @radius_lg;
            padding: 6px 0px;
            color: @text_primary;
        }
        QMenu::item {
            padding: 5px 22px 5px 16px;
            border-radius: @radius_sm;
            margin: 1px 4px;
        }
        QMenu::item:selected { background-color: @accent_primary; color: white; }
        QMenu::item:disabled { color: @text_disabled; }
        /* Items tagged role=danger (Delete, Close diagram, etc.) — red text. */
        QMenu::item[role="danger"]            { color: @error; }
        QMenu::item[role="danger"]:selected   { background-color: @error; color: white; }
        QMenu::separator { height: 1px; background-color: @border_primary; margin: 4px 10px; }
        QMenu::icon { padding-left: 8px; }
        QMenu::indicator { width: 14px; height: 14px; left: 6px; }
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


def _build_qpalette() -> QPalette:
    """Build a QPalette from the active theme.

    QSS alone is not enough on Windows: native-styled widgets and any widget
    without an explicit QSS color rule fall back to QApplication.palette(),
    which Qt only auto-syncs with the OS dark mode on macOS. Without this,
    dark mode shows dark text on dark backgrounds on Windows/Linux.
    """
    theme = theme_manager.get_current_theme()
    text       = QColor(theme['text_primary'])
    text_dim   = QColor(theme['text_secondary'])
    text_off   = QColor(theme['text_disabled'])
    window_bg  = QColor(theme['background_primary'])
    panel_bg   = QColor(theme['background_secondary'])
    surface    = QColor(theme['surface'])
    accent     = QColor(theme['accent_primary'])
    error      = QColor(theme['error'])
    white      = QColor('#FFFFFF')

    p = QPalette()
    p.setColor(QPalette.Window,          window_bg)
    p.setColor(QPalette.WindowText,      text)
    p.setColor(QPalette.Base,            surface)
    p.setColor(QPalette.AlternateBase,   panel_bg)
    p.setColor(QPalette.Text,            text)
    p.setColor(QPalette.PlaceholderText, text_dim)
    p.setColor(QPalette.Button,          panel_bg)
    p.setColor(QPalette.ButtonText,      text)
    p.setColor(QPalette.BrightText,      error)
    p.setColor(QPalette.ToolTipBase,     surface)
    p.setColor(QPalette.ToolTipText,     text)
    p.setColor(QPalette.Highlight,       accent)
    p.setColor(QPalette.HighlightedText, white)
    p.setColor(QPalette.Link,            accent)
    p.setColor(QPalette.LinkVisited,     accent)

    p.setColor(QPalette.Disabled, QPalette.WindowText, text_off)
    p.setColor(QPalette.Disabled, QPalette.Text,       text_off)
    p.setColor(QPalette.Disabled, QPalette.ButtonText, text_off)
    p.setColor(QPalette.Disabled, QPalette.Highlight,  panel_bg)
    p.setColor(QPalette.Disabled, QPalette.HighlightedText, text_off)
    return p


def _maybe_use_fusion_style(app):
    """Work around QTBUG-109450 on macOS.

    On macOS with Qt 5.15 the native ``macintosh`` style fails to draw the
    blinking text caret in any QLineEdit/QSpinBox that has a stylesheet
    setting ``background-color`` — which is every input field in this app.
    The Fusion style is fully stylesheet-aware and draws the caret itself,
    so switching to it restores cursor visibility everywhere.

    Scoped to macOS + Qt >= 5.10: the x86_64 release build ships PyQt5 5.9,
    whose native style renders the caret correctly, so it is left untouched.
    """
    import sys
    from PyQt5.QtCore import QT_VERSION
    if sys.platform == 'darwin' and QT_VERSION >= 0x050A00:  # 5.10.0
        from PyQt5.QtWidgets import QStyleFactory
        fusion = QStyleFactory.create("Fusion")
        if fusion is not None:
            app.setStyle(fusion)


def apply_modern_theme(app):
    _maybe_use_fusion_style(app)
    app.setPalette(_build_qpalette())
    app.setStyleSheet(ModernStyles.get_complete_stylesheet())

    def on_theme_changed():
        app.setPalette(_build_qpalette())
        qss = ModernStyles.get_complete_stylesheet()
        # Never call app.setStyleSheet() after startup — Qt's stylesheet
        # engine segfaults traversing pyqtgraph/OpenGL widgets.  Apply the
        # stylesheet per-window, restricted to QMainWindow instances only.
        #
        # Non-QMainWindow top-level windows (free-floating dialogs, detached
        # panels, editors) are intentionally skipped for two reasons:
        # (1) the app-wide palette re-applied above already retones
        # native/palette-driven widgets, and (2) the known self-styling
        # widgets (error_panel, command_palette, workspace_editor,
        # variable_editor, property_editor, …) subscribe to
        # theme_manager.theme_changed and re-style themselves. Blindly
        # restyling every top-level widget would also risk pushing this QSS
        # onto pyqtgraph/OpenGL windows, re-triggering the segfault noted
        # above. A top-level window that relies purely on inherited QSS and
        # does not self-subscribe will keep stale colors until reconstructed.
        from PyQt5.QtWidgets import QMainWindow
        for w in app.topLevelWidgets():
            try:
                if isinstance(w, QMainWindow):
                    w.setStyleSheet(qss)
            except RuntimeError:
                pass

    theme_manager.theme_changed.connect(on_theme_changed)
