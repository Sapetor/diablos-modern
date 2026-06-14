"""
Keyboard Shortcuts reference dialog.

A read-only, themed listing of the application's keyboard shortcuts grouped by
category (File, Edit, Simulation, View, Help). The shortcut (label, key) data is
copied from ``CommandPaletteManager.setup`` (modern_ui/managers/
command_palette_manager.py, the sim/view/file action tables) plus the menu
accelerators wired up in ``MenuBuilder`` so the two stay visually in sync. The
dialog performs no actions; it only displays the bindings.

Styling follows the project convention: every color comes from
``theme_manager.get_color(...)`` and the typographic scale from
``get_ui_font``/``get_mono_font`` — no hardcoded hex, px, or font families.
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QGridLayout, QScrollArea, QWidget, QDialogButtonBox,
)
from PyQt5.QtCore import Qt

from modern_ui.themes.theme_manager import (
    theme_manager, get_ui_font, get_mono_font, TYPE, WEIGHT, SPACE,
)


# Shortcut catalogue grouped by category. Each entry is a (label, key) pair;
# the sim/view/file groups mirror the action tables in CommandPaletteManager.
# An empty key string means "no default binding" and renders the key column
# blank. Keep this in sync with MenuBuilder and CommandPaletteManager.
SHORTCUT_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
    ("File", [
        ("New diagram", "Ctrl+N"),
        ("Open diagram", "Ctrl+O"),
        ("Save diagram", "Ctrl+S"),
        ("Export as TikZ", ""),
        ("Show plots", ""),
        ("Exit", "Alt+F4"),
    ]),
    ("Edit", [
        ("Undo", "Ctrl+Z"),
        ("Redo", "Ctrl+Y"),
        ("Select all", "Ctrl+A"),
        ("Create subsystem", "Ctrl+G"),
        ("Command palette", "Ctrl+P"),
    ]),
    ("Simulation", [
        ("Run simulation", "F5"),
        ("Pause simulation", "F6"),
        ("Stop simulation", "F7"),
        ("Step simulation", "F8"),
        ("Toggle fast solver", ""),
    ]),
    ("View", [
        ("Zoom in", "Ctrl++"),
        ("Zoom out", "Ctrl+-"),
        ("Fit to window", "Ctrl+0"),
        ("Toggle theme", "Ctrl+T"),
        ("Toggle grid", "Ctrl+Shift+G"),
        ("Toggle minimap", "Ctrl+Shift+M"),
        ("Toggle variable editor", "Ctrl+Shift+V"),
        ("Toggle workspace variables", "Ctrl+Shift+W"),
        ("Toggle tuning panel", "Ctrl+Shift+T"),
    ]),
    ("Help", [
        ("Keyboard shortcuts", "F1"),
    ]),
]


class KeyboardShortcutsDialog(QDialog):
    """Read-only listing of keyboard shortcuts grouped by category."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Keyboard Shortcuts")
        self.setMinimumWidth(420)
        self.setMinimumHeight(480)
        self.setModal(True)

        self._setup_ui()

    # ------------------------------------------------------------------ UI ---
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(SPACE["lg"])

        heading = QLabel("Keyboard Shortcuts")
        heading.setFont(get_ui_font(TYPE["heading"], WEIGHT["semibold"]))
        heading.setStyleSheet(
            f"color: {theme_manager.get_color('text_primary').name()};"
        )
        layout.addWidget(heading)

        # Scrollable body so a long catalogue stays usable on small screens.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setSpacing(SPACE["xl"])
        body_layout.setContentsMargins(0, 0, 0, 0)

        for title, entries in SHORTCUT_GROUPS:
            body_layout.addWidget(self._make_group(title, entries))
        body_layout.addStretch(1)

        scroll.setWidget(body)
        layout.addWidget(scroll, 1)

        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        button_box.rejected.connect(self.reject)
        button_box.accepted.connect(self.accept)
        layout.addWidget(button_box)

    def _make_group(self, title: str, entries: list[tuple[str, str]]) -> QWidget:
        """Build one category block: a title plus a label/key grid."""
        container = QWidget()
        col = QVBoxLayout(container)
        col.setSpacing(SPACE["sm"])
        col.setContentsMargins(0, 0, 0, 0)

        title_label = QLabel(title)
        title_label.setFont(get_ui_font(TYPE["subtitle"], WEIGHT["semibold"]))
        title_label.setStyleSheet(
            f"color: {theme_manager.get_color('accent_primary').name()};"
        )
        col.addWidget(title_label)

        grid = QGridLayout()
        grid.setHorizontalSpacing(SPACE["xl"])
        grid.setVerticalSpacing(SPACE["xs"])
        grid.setColumnStretch(0, 1)

        for row, (label, key) in enumerate(entries):
            grid.addWidget(self._make_label(label), row, 0)
            grid.addWidget(self._make_key(key), row, 1)

        col.addLayout(grid)
        return container

    def _make_label(self, text: str) -> QLabel:
        """Action description in the left column."""
        label = QLabel(text)
        label.setFont(get_ui_font(TYPE["body"], WEIGHT["regular"]))
        label.setStyleSheet(
            f"color: {theme_manager.get_color('text_primary').name()};"
        )
        return label

    def _make_key(self, key: str) -> QLabel:
        """Key binding (kbd glyph) in the right column; mono, dimmed when empty."""
        kbd = QLabel(key or "—")
        kbd.setFont(get_mono_font(TYPE["caption"], WEIGHT["medium"]))
        kbd.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        token = 'text_secondary' if key else 'text_disabled'
        kbd.setStyleSheet(
            f"color: {theme_manager.get_color(token).name()};"
        )
        return kbd
