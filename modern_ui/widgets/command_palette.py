"""Command Palette (⌘K) — Linear-style V1 from the menu-handoff redesign.

A 520px-wide semi-transparent modal positioned at the top of the host window.
Rows show a left icon column, a label, an optional category badge (BLOCK / SIM /
VIEW / FILE / HELP), and an optional shortcut hint. A footer surfaces nav /
run / complete hints and a result count.

Recently-used commands float to the top via ``~/.diablos/commands.json``.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QSize
from PyQt5.QtGui import QFont, QKeyEvent
from PyQt5.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QLineEdit, QListWidget,
    QListWidgetItem, QLabel, QWidget, QFrame, QSizePolicy
)

from modern_ui.themes.theme_manager import theme_manager

logger = logging.getLogger(__name__)


PALETTE_WIDTH = 520
ROW_MAX_HEIGHT = 312
SEARCH_HEIGHT = 44
ICON_COL_W = 22
BADGE_W = 50

# Map command 'type' -> short category badge text
TYPE_BADGE = {
    'block':   'BLOCK',
    'action':  'ACTION',
    'sim':     'SIM',
    'view':    'VIEW',
    'file':    'FILE',
    'recent':  'FILE',
    'setting': 'SET',
    'help':    'HELP',
}


def _mono(size: int = 9) -> QFont:
    f = QFont("Menlo")
    f.setStyleHint(QFont.Monospace)
    if hasattr(f, 'setFamilies'):
        f.setFamilies(["Menlo", "Consolas", "JetBrains Mono", "DejaVu Sans Mono", "monospace"])
    f.setPointSize(size)
    return f


def _recent_path() -> str:
    """Per-user recents file."""
    base = os.path.expanduser("~/.diablos")
    try:
        os.makedirs(base, exist_ok=True)
    except OSError as e:
        logger.debug("Could not create recents dir %s: %s", base, e)
    return os.path.join(base, "commands.json")


def _load_recents() -> List[str]:
    try:
        with open(_recent_path(), "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [str(x) for x in data][:32]
    except (OSError, json.JSONDecodeError) as e:
        logger.debug("Could not load recents: %s", e)
    return []


def _save_recents(names: List[str]):
    try:
        with open(_recent_path(), "w") as f:
            json.dump(names[:32], f)
    except OSError as e:
        logger.debug("Could not save recents: %s", e)


class _PaletteRow(QWidget):
    """Custom list-row widget: icon | label | category | shortcut."""

    def __init__(self, cmd: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.setObjectName("PaletteRow")
        self._cmd = cmd

        self._text = theme_manager.get_color('text_primary').name()
        self._text_dim = theme_manager.get_color('text_secondary').name()
        self._border = theme_manager.get_color('border_primary').name()

        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 6, 12, 6)
        lay.setSpacing(12)

        # Icon column — simple glyph or category dot
        icon_lbl = QLabel(self._glyph_for_type(cmd.get('type', 'action')))
        icon_lbl.setFixedWidth(ICON_COL_W)
        icon_lbl.setAlignment(Qt.AlignCenter)
        icon_lbl.setStyleSheet(f"color: {self._text_dim}; font-size: 13pt;")
        lay.addWidget(icon_lbl)

        # Label
        label = QLabel(cmd.get('name', ''))
        label.setStyleSheet(f"color: {self._text};")
        lf = label.font()
        lf.setPointSize(11)
        label.setFont(lf)
        lay.addWidget(label, 1)

        # Category badge
        badge_text = TYPE_BADGE.get(cmd.get('type', ''), '')
        if badge_text:
            badge = QLabel(badge_text)
            badge.setFont(_mono(8))
            badge.setStyleSheet(
                f"color: {self._text_dim};"
                f" border: 1px solid {self._border};"
                f" border-radius: 3px;"
                f" padding: 1px 6px;"
            )
            badge.setAlignment(Qt.AlignCenter)
            lay.addWidget(badge)

        # Shortcut hint (optional)
        kbd = cmd.get('shortcut') or cmd.get('data', {}).get('shortcut')
        if kbd:
            kbd_lbl = QLabel(kbd)
            kbd_lbl.setFont(_mono(8))
            kbd_lbl.setStyleSheet(
                f"color: {self._text_dim};"
                f" background: rgba(255,255,255,0.04);"
                f" border: 1px solid {self._border};"
                f" border-radius: 3px;"
                f" padding: 1px 5px;"
            )
            lay.addWidget(kbd_lbl)

    def _glyph_for_type(self, t: str) -> str:
        # ASCII-only glyphs to stay font-independent
        return {
            'block':   '+',
            'action':  '>',
            'sim':     '▶',
            'view':    '☰',
            'file':    '◰',
            'recent':  '◷',
            'setting': '⚙',
            'help':    '?',
        }.get(t, '•')


class CommandPalette(QDialog):
    """⌘K command palette — Linear-style minimal V1.

    Surface a flat list of executable commands; filter by typed input;
    Enter to run; Esc to close. Recently-used commands float to the top.
    """

    command_selected = pyqtSignal(str, dict)  # command_type, data

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Popup)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        self._commands: List[Dict[str, Any]] = []
        self._filtered: List[Dict[str, Any]] = []
        self._recents: List[str] = _load_recents()

        self._build_ui()
        self._apply_theme()
        theme_manager.theme_changed.connect(self._apply_theme)

    # -- UI ---------------------------------------------------------------

    def _build_ui(self):
        # Outer rounded card with shadow approximation (1px border, dark bg)
        self.setFixedWidth(PALETTE_WIDTH)

        self._card = QFrame(self)
        self._card.setObjectName("CmdkCard")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._card)

        v = QVBoxLayout(self._card)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        # Header row: search input + esc hint
        head = QWidget()
        h = QHBoxLayout(head)
        h.setContentsMargins(16, 10, 12, 10)
        h.setSpacing(10)

        glyph = QLabel("⌕")
        glyph.setStyleSheet(
            f"color: {theme_manager.get_color('text_secondary').name()}; font-size: 14pt;"
        )
        h.addWidget(glyph)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Type a command, block, or file…")
        sf = self.search.font()
        sf.setPointSize(12)
        self.search.setFont(sf)
        self.search.setFixedHeight(SEARCH_HEIGHT - 12)
        self.search.setFrame(False)
        self.search.setStyleSheet("QLineEdit { border: 0; background: transparent; }")
        self.search.textChanged.connect(self._on_search_changed)
        self.search.returnPressed.connect(self._run_selected)
        self.search.installEventFilter(self)
        h.addWidget(self.search, 1)

        esc = QLabel("esc")
        esc.setFont(_mono(8))
        esc.setStyleSheet(
            f"color: {theme_manager.get_color('text_secondary').name()};"
            f" border: 1px solid {theme_manager.get_color('border_primary').name()};"
            f" border-radius: 3px; padding: 1px 6px;"
        )
        h.addWidget(esc)

        v.addWidget(head)

        # Divider
        div1 = QFrame(); div1.setFrameShape(QFrame.HLine); div1.setObjectName("CmdkDivider")
        v.addWidget(div1)

        # Results list (custom rows)
        self.list = QListWidget()
        self.list.setObjectName("CmdkList")
        self.list.setMaximumHeight(ROW_MAX_HEIGHT)
        self.list.setMinimumHeight(40)
        self.list.setSpacing(0)
        self.list.setFocusPolicy(Qt.NoFocus)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list.itemClicked.connect(lambda _it: self._run_selected())
        v.addWidget(self.list)

        # Divider 2
        div2 = QFrame(); div2.setFrameShape(QFrame.HLine); div2.setObjectName("CmdkDivider")
        v.addWidget(div2)

        # Footer
        foot = QWidget()
        f = QHBoxLayout(foot)
        f.setContentsMargins(14, 6, 14, 6)
        f.setSpacing(14)
        self._foot_text = QLabel("↑↓ navigate · ↵ run · esc close")
        self._foot_text.setFont(_mono(8))
        f.addWidget(self._foot_text, 1)
        self._foot_count = QLabel("0 results")
        self._foot_count.setFont(_mono(8))
        f.addWidget(self._foot_count)
        v.addWidget(foot)

    def _apply_theme(self):
        bg = theme_manager.get_color('surface_elevated')
        border = theme_manager.get_color('border_primary').name()
        text = theme_manager.get_color('text_primary').name()
        text_dim = theme_manager.get_color('text_secondary').name()
        accent = theme_manager.get_color('accent_primary').name()

        bg_rgba = f"rgba({bg.red()}, {bg.green()}, {bg.blue()}, 245)"

        self._card.setStyleSheet(f"""
            QFrame#CmdkCard {{
                background: {bg_rgba};
                border: 1px solid {border};
                border-radius: 12px;
            }}
            QFrame#CmdkDivider {{
                background: {border};
                max-height: 1px;
                border: 0;
            }}
            QListWidget#CmdkList {{
                background: transparent;
                border: 0;
                outline: 0;
                padding: 6px 4px;
            }}
            QListWidget#CmdkList::item {{
                background: transparent;
                border-radius: 7px;
                padding: 0px;
                margin: 2px 4px;
            }}
            QListWidget#CmdkList::item:selected {{
                background: {accent};
            }}
            QListWidget#CmdkList::item:hover {{
                background: rgba(96,165,250,0.18);
            }}
        """)

        if hasattr(self, '_foot_text'):
            self._foot_text.setStyleSheet(f"color: {text_dim};")
            self._foot_count.setStyleSheet(f"color: {text_dim};")
        if hasattr(self, 'search'):
            self.search.setStyleSheet(
                f"QLineEdit {{ border: 0; background: transparent; color: {text}; }}"
            )

    # -- Public API -------------------------------------------------------

    def set_commands(self, commands: List[Dict[str, Any]]):
        # Drop placeholder/example commands without a callback to avoid no-op rows
        self._commands = [c for c in commands if c.get('callback')]
        self._refresh()

    def show_palette(self):
        """Show palette centered at the top of the parent window."""
        self.search.clear()
        self._refresh()
        self.show()
        self._position_top()
        self.raise_()
        self.activateWindow()
        self.search.setFocus()

    # -- Filtering --------------------------------------------------------

    def _on_search_changed(self, _text: str):
        self._refresh()

    def _score(self, cmd: Dict[str, Any], query: str) -> int:
        """Simple fuzzy-ish score: substring + alias hits, recents bonus."""
        q = (query or '').strip().lower()
        if not q:
            base = 100
        else:
            name = (cmd.get('name', '') or '').lower()
            desc = (cmd.get('description', '') or '').lower()
            aliases = ' '.join(cmd.get('aliases', [])).lower()
            hay = f"{name} {desc} {aliases}"
            if q in name:
                base = 80 - name.find(q)  # earlier match scores higher
            elif q in hay:
                base = 40
            else:
                # All query chars must appear in name in order
                i = 0
                for ch in q:
                    j = name.find(ch, i)
                    if j < 0:
                        return -1
                    i = j + 1
                base = 10
        # Recents bonus
        try:
            r_idx = self._recents.index(cmd.get('name', ''))
            base += max(0, 20 - r_idx)
        except ValueError:
            pass
        return base

    def _refresh(self):
        q = self.search.text() if hasattr(self, 'search') else ''
        scored = []
        for cmd in self._commands:
            s = self._score(cmd, q)
            if s >= 0:
                scored.append((s, cmd))
        scored.sort(key=lambda t: -t[0])
        self._filtered = [c for _s, c in scored]

        self.list.clear()
        for cmd in self._filtered:
            item = QListWidgetItem()
            row = _PaletteRow(cmd)
            item.setSizeHint(QSize(0, row.sizeHint().height() + 2))
            item.setData(Qt.UserRole, cmd)
            self.list.addItem(item)
            self.list.setItemWidget(item, row)

        if self.list.count():
            self.list.setCurrentRow(0)
        self._foot_count.setText(f"{self.list.count()} results")

    # -- Execution --------------------------------------------------------

    def _run_selected(self):
        item = self.list.currentItem()
        if item is None and self.list.count() > 0:
            item = self.list.item(0)
        if item is None:
            return
        cmd = item.data(Qt.UserRole)
        self._execute(cmd)

    def _execute(self, cmd: Dict[str, Any]):
        # Bump in recents
        name = cmd.get('name', '')
        if name:
            self._recents = [name] + [n for n in self._recents if n != name]
            _save_recents(self._recents)

        cb = cmd.get('callback')
        if callable(cb):
            try:
                cb()
            except Exception as e:
                logger.exception("Command palette callback failed: %s", e)
        self.command_selected.emit(cmd.get('type', 'action'), cmd.get('data', {}))
        self.close()

    # -- Positioning / events --------------------------------------------

    def _position_top(self):
        parent = self.parent()
        if parent and hasattr(parent, 'geometry'):
            geo = parent.frameGeometry()
            x = geo.x() + (geo.width() - self.width()) // 2
            y = geo.y() + max(80, geo.height() // 8)
            self.move(QPoint(x, y))
            return
        # Fallback: center on cursor's screen
        from PyQt5.QtGui import QCursor
        screen = QApplication.desktop().availableGeometry(QCursor.pos())
        x = screen.x() + (screen.width() - self.width()) // 2
        y = screen.y() + max(80, screen.height() // 8)
        self.move(QPoint(x, y))

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        if key == Qt.Key_Escape:
            self.close()
        elif key == Qt.Key_Down:
            self._move_selection(+1)
        elif key == Qt.Key_Up:
            self._move_selection(-1)
        elif key in (Qt.Key_Return, Qt.Key_Enter):
            self._run_selected()
        else:
            super().keyPressEvent(event)

    def eventFilter(self, obj, event):
        if obj is self.search and event.type() == event.KeyPress:
            key = event.key()
            if key == Qt.Key_Down:
                self._move_selection(+1); return True
            if key == Qt.Key_Up:
                self._move_selection(-1); return True
            if key in (Qt.Key_Return, Qt.Key_Enter):
                self._run_selected(); return True
            if key == Qt.Key_Escape:
                self.close(); return True
        return super().eventFilter(obj, event)

    def _move_selection(self, delta: int):
        n = self.list.count()
        if n == 0:
            return
        row = self.list.currentRow()
        row = (row + delta) % n
        self.list.setCurrentRow(row)
