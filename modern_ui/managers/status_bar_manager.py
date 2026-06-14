"""
StatusBarManager -- builds and refreshes the main window's compact pill-style
status bar.

Extracted verbatim (behavior-preserving) from ``ModernDiaBloSWindow`` so the
main window keeps only thin facades. Follows the same manager pattern as the
other ``modern_ui/managers`` (constructed with the main window, held as
``self.window``).

``setup()`` assigns the status-bar segments back onto the window as attributes
(``status_message``, ``status_pill``, ``file_status``, ``file_unsaved_status``,
``counts_status``, ``cursor_status``, ``zoom_status``, ``theme_status``,
``_counts_refresh_timer``) because those names are referenced widely across the
codebase (e.g. ``window.status_message.setText(...)`` appears in many call
sites, and ``AppearanceManager`` reads ``window.theme_status``).
"""

import os
import logging

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QLabel, QFrame

from modern_ui.themes.theme_manager import theme_manager, ThemeType

logger = logging.getLogger(__name__)


class StatusBarManager:
    """Owns construction and periodic refresh of the window status bar."""

    def __init__(self, main_window):
        self.window = main_window

    def setup(self):
        """Compact pill-style status bar (≤ 28px tall).

        Layout left-to-right:  state-pill · file-pill · counts · ⟶ · cursor · zoom · theme-pill
        Segments are separated by 1px vertical dividers (no pipes).
        """
        from modern_ui.widgets.modern_toolbar import _StatusPill  # reuse toolbar pill

        window = self.window
        statusbar = window.statusBar()
        statusbar.setSizeGripEnabled(False)
        statusbar.setFixedHeight(26)

        def _vsep():
            f = QFrame()
            f.setFrameShape(QFrame.VLine)
            f.setObjectName("StatusDivider")
            f.setStyleSheet(
                f"color: {theme_manager.get_color('border_primary').name()};"
                f" background: {theme_manager.get_color('border_primary').name()};"
                f" max-width: 1px; min-width: 1px;"
            )
            f.setFixedHeight(14)
            return f

        def _mono_label(text=""):
            from PyQt5.QtGui import QFont as _QF
            lbl = QLabel(text)
            f = _QF("Menlo")
            f.setStyleHint(_QF.Monospace)
            if hasattr(f, 'setFamilies'):
                f.setFamilies(["Menlo", "Consolas", "JetBrains Mono", "DejaVu Sans Mono", "monospace"])
            f.setPointSize(8)
            lbl.setFont(f)
            return lbl

        # Left: status pill (reused from toolbar)
        window.status_pill = _StatusPill(window)
        window.status_pill.setToolTip("Simulation state")
        statusbar.addWidget(window.status_pill)

        # Hidden compatibility shim — many call sites still call status_message.setText(...)
        window.status_message = QLabel()
        window.status_message.hide()
        # Forward text changes to the pill (idle/running/paused detection)
        def _on_status_text_changed(text):
            try:
                window.toolbar.set_status(text)
            except Exception:
                pass
            t = (text or "").lower()
            if 'run' in t and 'paus' not in t:
                window.status_pill.set_state('running')
            elif 'paus' in t:
                window.status_pill.set_state('paused')
            elif 'error' in t or 'fail' in t:
                window.status_pill.set_state('error', text)
            else:
                window.status_pill.set_state('idle', text if text else None)
        # Replace setText to propagate to the pill
        _orig_setText = window.status_message.setText
        def _propagating_setText(text):
            _orig_setText(text)
            _on_status_text_changed(text)
        window.status_message.setText = _propagating_setText  # type: ignore[attr-defined]

        statusbar.addWidget(_vsep())

        # File info: filename + unsaved indicator
        window.file_status = QLabel("untitled")
        window.file_status.setToolTip("Current diagram file")
        window.file_status.setStyleSheet(
            f"color: {theme_manager.get_color('text_primary').name()};"
        )
        window.file_unsaved_status = QLabel("")
        window.file_unsaved_status.setToolTip("Unsaved changes indicator")
        window.file_unsaved_status.setStyleSheet(
            f"color: {theme_manager.get_color('text_disabled').name()};"
            f" font-size: 9pt;"
        )
        statusbar.addWidget(window.file_status)
        statusbar.addWidget(window.file_unsaved_status)

        statusbar.addWidget(_vsep())

        # Counts pill: blocks N · wires M · scopes K
        window.counts_status = _mono_label("blocks 0 · wires 0 · scopes 0")
        window.counts_status.setToolTip("Blocks · wires · scopes")
        window.counts_status.setStyleSheet(
            f"color: {theme_manager.get_color('text_secondary').name()};"
        )
        statusbar.addWidget(window.counts_status)

        # ----- right-aligned permanent widgets -----
        window.cursor_status = _mono_label("cursor 0,0")
        window.cursor_status.setToolTip("Cursor position (x, y)")
        window.cursor_status.setStyleSheet(
            f"color: {theme_manager.get_color('text_secondary').name()};"
        )
        statusbar.addPermanentWidget(window.cursor_status)

        statusbar.addPermanentWidget(_vsep())

        window.zoom_status = _mono_label("zoom 100%")
        window.zoom_status.setToolTip("Canvas zoom level")
        window.zoom_status.setStyleSheet(
            f"color: {theme_manager.get_color('text_secondary').name()};"
        )
        statusbar.addPermanentWidget(window.zoom_status)

        statusbar.addPermanentWidget(_vsep())

        # Theme + palette tag
        theme_label = "Dark" if theme_manager.current_theme == ThemeType.DARK else "Light"
        from modern_ui.themes.theme_manager import PALETTE_DISPLAY_NAMES
        palette_label = PALETTE_DISPLAY_NAMES.get(theme_manager.current_palette, theme_manager.current_palette).split()[0]
        window.theme_status = QLabel(f"{theme_label} · {palette_label}")
        window.theme_status.setToolTip("Click to toggle theme (Ctrl+T)")
        window.theme_status.setStyleSheet(
            f"color: {theme_manager.get_color('text_secondary').name()};"
            f" background-color: {theme_manager.get_color('background_tertiary').name()};"
            f" padding: 1px 8px; border-radius: 4px; font-size: 9pt;"
        )
        statusbar.addPermanentWidget(window.theme_status)

        # Drive zoom from toolbar's zoom rocker so the two stay in sync
        try:
            window.toolbar.zoom_changed.connect(
                lambda f: window.zoom_status.setText(f"zoom {int(round(f*100))}%")
            )
        except Exception:
            pass

        # Cursor pos from canvas
        try:
            window.canvas.cursor_moved.connect(
                lambda x, y: window.cursor_status.setText(f"cursor {x},{y}")
            )
        except Exception:
            pass

        # Periodic counts refresh (cheap; runs on the same timer that paints)
        window._counts_refresh_timer = QTimer(window)
        window._counts_refresh_timer.timeout.connect(self.refresh_counts)
        window._counts_refresh_timer.start(500)

        # Initial state
        self.refresh_counts()
        self.refresh_file_status()

        # Apply theme palette to the statusbar host
        window.appearance_manager.update_statusbar_colors()

    def refresh_counts(self):
        """Update the counts pill from current dsim state."""
        window = self.window
        try:
            dsim = getattr(window, 'dsim', None)
            if dsim is None:
                return
            blocks = list(getattr(dsim, 'blocks_list', []) or [])
            wires = list(getattr(dsim, 'line_list', []) or [])
            scopes = sum(1 for b in blocks if getattr(b, 'block_fn', '') in ('Scope', 'FieldScope'))
            window.counts_status.setText(
                f"blocks {len(blocks)} · wires {len(wires)} · scopes {scopes}"
            )
        except Exception:
            pass

    def refresh_file_status(self):
        """Update filename + unsaved indicator in the status bar."""
        window = self.window
        try:
            path = getattr(window.dsim, 'current_filepath', None) or getattr(window.dsim, 'filepath', None)
            name = os.path.basename(path) if path else "untitled"
            window.file_status.setText(name)
            window.file_unsaved_status.setText("unsaved" if getattr(window.dsim, 'dirty', False) else "")
        except Exception:
            pass
