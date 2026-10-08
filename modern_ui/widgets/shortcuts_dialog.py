"""
Keyboard Shortcuts reference dialog.

A read-only, themed listing of the application's keyboard shortcuts grouped by
category. When opened from the main window the groups are generated from the
live menu bar's ``QAction`` shortcuts (``build_live_shortcut_groups``), so the
reference is the real binding table and cannot drift. Without a window (unit
tests, tools) it falls back to the static catalogue: the Simulation/View groups
and the bulk of File come from
``command_palette_manager.palette_command_groups``, the rest is an explicit
supplement kept in sync with ``MenuBuilder``. The dialog performs no actions.

Styling follows the project convention: every color comes from
``theme_manager.get_color(...)`` and the typographic scale from
``get_ui_font``/``get_mono_font`` — no hardcoded hex, px, or font families.
"""

from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel,
    QFrame,
    QGridLayout,
    QScrollArea,
    QWidget,
    QDialogButtonBox,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence

from lib.i18n import tr, tr_noop
from modern_ui.managers.command_palette_manager import palette_command_groups
from modern_ui.themes.theme_manager import (
    theme_manager,
    get_ui_font,
    get_mono_font,
    TYPE,
    WEIGHT,
    SPACE,
)


# Rows that are NOT palette commands: menu-only accelerators and editor actions
# wired up in ``MenuBuilder``. Each entry is a (label, key) pair; an empty key
# string means "no default binding". The File supplement is appended after the
# live File group; Edit and Help are standalone groups. Keep in sync with
# MenuBuilder.
_FILE_SUPPLEMENT: list[tuple[str, str]] = [
    (tr_noop("Exit"), "Ctrl+Q"),
]
_EDIT_GROUP: list[tuple[str, str]] = [
    (tr_noop("Undo"), "Ctrl+Z"),
    (tr_noop("Redo"), "Ctrl+Shift+Z"),
    (tr_noop("Cut"), "Ctrl+X"),
    (tr_noop("Copy"), "Ctrl+C"),
    (tr_noop("Paste"), "Ctrl+V"),
    (tr_noop("Select all"), "Ctrl+A"),
    (tr_noop("Create subsystem"), "Ctrl+G"),
    (tr_noop("Flip block"), "Ctrl+F"),
    (tr_noop("Command palette"), "Ctrl+K"),
]
# Canvas keys that are handled by key events, not actions (see MenuBuilder).
_CANVAS_GROUP: list[tuple[str, str]] = [
    (tr_noop("Delete selection"), "Delete"),
    (tr_noop("Cancel / clear selection"), "Esc"),
]
_HELP_GROUP: list[tuple[str, str]] = [
    (tr_noop("Keyboard shortcuts"), "F1"),
]


def build_shortcut_groups() -> list[tuple[str, list[tuple[str, str]]]]:
    """Assemble the dialog's display catalogue: group -> [(label, key)].

    The Simulation/View groups and the bulk of File come straight from the
    command registry (``palette_command_groups``); the menu-only File
    accelerators, the Edit group, and Help are appended as an explicit
    supplement. Ordered File, Edit, Simulation, View, Help.
    """
    registry = palette_command_groups()
    return [
        (tr_noop("File"), registry["File"] + _FILE_SUPPLEMENT),
        (tr_noop("Edit"), list(_EDIT_GROUP)),
        (tr_noop("Simulation"), registry["Simulation"]),
        (tr_noop("View"), registry["View"]),
        (tr_noop("Help"), list(_HELP_GROUP)),
        (tr_noop("Canvas"), list(_CANVAS_GROUP)),
    ]


def _strip_mnemonic(text: str) -> str:
    """Menu label -> plain label: ``&Open`` -> ``Open``, ``&&`` -> ``&``."""
    return text.replace("&&", "\0").replace("&", "").replace("\0", "&").strip()


def build_live_shortcut_groups(window) -> list[tuple[str, list[tuple[str, str]]]]:
    """Generate the catalogue from ``window``'s menu bar actions (the real bindings).

    One group per top-level menu that has bound actions, plus the key-event-only
    canvas keys. Labels are already translated; ``tr`` on them later is a no-op.
    """
    groups: list[tuple[str, list[tuple[str, str]]]] = []

    def collect(menu, out, seen):
        for action in menu.actions():
            if action.menu() is not None:
                collect(action.menu(), out, seen)
                continue
            keys = [k.toString(QKeySequence.SequenceFormat.NativeText) for k in action.shortcuts()]
            if not keys or id(action) in seen:
                continue
            seen.add(id(action))
            out.append((_strip_mnemonic(action.text()), " / ".join(keys)))

    for top in window.menuBar().actions():
        menu = top.menu()
        if menu is None:
            continue
        entries: list[tuple[str, str]] = []
        collect(menu, entries, set())
        if entries:
            groups.append((_strip_mnemonic(top.text()), entries))
    groups.append((tr("Canvas"), [(tr(label), key) for label, key in _CANVAS_GROUP]))
    return groups


# Display catalogue grouped by category. Built once at import time from the
# command registry plus the menu-only supplement above.
SHORTCUT_GROUPS: list[tuple[str, list[tuple[str, str]]]] = build_shortcut_groups()


class KeyboardShortcutsDialog(QDialog):
    """Read-only listing of keyboard shortcuts grouped by category."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._groups = SHORTCUT_GROUPS
        if parent is not None and hasattr(parent, "menuBar"):
            try:
                self._groups = build_live_shortcut_groups(parent)
            except Exception:  # fall back to the static catalogue
                self._groups = SHORTCUT_GROUPS

        self.setWindowTitle(tr("Keyboard Shortcuts"))
        self.setMinimumWidth(420)
        self.setMinimumHeight(480)
        self.setModal(True)

        self._setup_ui()

    # ------------------------------------------------------------------ UI ---
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(SPACE["lg"])

        heading = QLabel(tr("Keyboard Shortcuts"))
        heading.setFont(get_ui_font(TYPE["heading"], WEIGHT["semibold"]))
        heading.setStyleSheet(f"color: {theme_manager.get_color('text_primary').name()};")
        layout.addWidget(heading)

        # Scrollable body so a long catalogue stays usable on small screens.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setSpacing(SPACE["xl"])
        body_layout.setContentsMargins(0, 0, 0, 0)

        for title, entries in self._groups:
            body_layout.addWidget(self._make_group(title, entries))
        body_layout.addStretch(1)

        scroll.setWidget(body)
        layout.addWidget(scroll, 1)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.reject)
        button_box.accepted.connect(self.accept)
        layout.addWidget(button_box)

    def _make_group(self, title: str, entries: list[tuple[str, str]]) -> QWidget:
        """Build one category block: a title plus a label/key grid."""
        container = QWidget()
        col = QVBoxLayout(container)
        col.setSpacing(SPACE["sm"])
        col.setContentsMargins(0, 0, 0, 0)

        title_label = QLabel(tr(title))
        title_label.setFont(get_ui_font(TYPE["subtitle"], WEIGHT["semibold"]))
        title_label.setStyleSheet(f"color: {theme_manager.get_color('accent_primary').name()};")
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
        label = QLabel(tr(text))
        label.setFont(get_ui_font(TYPE["body"], WEIGHT["regular"]))
        label.setStyleSheet(f"color: {theme_manager.get_color('text_primary').name()};")
        return label

    def _make_key(self, key: str) -> QLabel:
        """Key binding (kbd glyph) in the right column; mono, dimmed when empty."""
        kbd = QLabel(key or "—")
        kbd.setFont(get_mono_font(TYPE["caption"], WEIGHT["medium"]))
        kbd.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        token = "text_secondary" if key else "text_disabled"
        kbd.setStyleSheet(f"color: {theme_manager.get_color(token).name()};")
        return kbd
