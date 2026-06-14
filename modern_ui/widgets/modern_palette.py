"""Modern Block Palette — V1 "Tightened" redesign

Drop-in replacement. Same public API as the original:
  - class ModernBlockPalette(QWidget) with __init__(dsim, parent=None)
  - Signal block_drag_started(object)
  - Methods: refresh_blocks(), get_available_blocks()

Layout change
-------------
Old:  2-column grid of 60×60 square cards inside a heavy "category" frame.
New:  Single-column list of ~28px rows — category dot · 18px glyph · name · drag hint.
      Roughly 3× the blocks visible without scrolling.

Drag MIME and signal are unchanged, so canvas drop logic doesn't need edits.
"""

import logging
import os
import sys
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, QLineEdit, QFrame
)
from PyQt5.QtCore import (
    Qt, pyqtSignal, QMimeData, QPoint, QRect, QRectF, QPointF, QSize, QSettings
)
from PyQt5.QtGui import QDrag, QPainter, QPixmap, QFont, QColor, QPen, QPainterPath

# QSettings org/app — sourced from the shared constants in lib.app_paths so
# palette UI state lives in the same store as the rest of the app's UI
# preferences (see main_window.py first-run flag).
from lib.app_paths import SETTINGS_ORG as _SETTINGS_ORG, SETTINGS_APP as _SETTINGS_APP

# Chevron glyphs for the collapsible category header (expanded / collapsed).
_CHEVRON_EXPANDED = "▾"
_CHEVRON_COLLAPSED = "▸"


def _collapsed_settings_key(category_name: str) -> str:
    """QSettings key under which a category's collapsed flag is persisted."""
    return f"palette/collapsed/{category_name}"

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modern_ui.themes.theme_manager import theme_manager

logger = logging.getLogger(__name__)


# Category color dots — driven by theme tokens that already exist.
def _category_dot_color(category: str) -> QColor:
    c = (category or '').lower()
    if 'source' in c:
        return theme_manager.get_color('block_source_accent')
    if 'math' in c:
        return theme_manager.get_color('block_process_accent')
    if 'control' in c or 'continuous' in c:
        return theme_manager.get_color('block_control_accent')
    if 'sink' in c:
        return theme_manager.get_color('block_sink_accent')
    if 'rout' in c:
        return theme_manager.get_color('text_secondary')
    if 'filter' in c:
        return theme_manager.get_color('accent_primary')
    if 'discrete' in c:
        return theme_manager.get_color('block_other_accent')
    return theme_manager.get_color('text_secondary')


# -----------------------------------------------------------------------------
# Block row — compact draggable item
# -----------------------------------------------------------------------------

class CompactBlockRow(QFrame):
    """Single-row palette item: dot · glyph · name."""

    block_drag_started = pyqtSignal(object, object)  # menu_block, position

    GLYPH_SIZE = 22
    ROW_HEIGHT = 30

    def __init__(self, menu_block, category_name, colors, parent=None):
        super().__init__(parent)
        self.menu_block = menu_block
        self.category_name = category_name
        self.colors = colors
        self.setObjectName("PaletteRow")
        self.setFixedHeight(self.ROW_HEIGHT)
        self.setCursor(Qt.OpenHandCursor)
        self.setFrameShape(QFrame.NoFrame)
        self.setAttribute(Qt.WA_StyledBackground, True)
        # Keyboard navigation: rows accept focus so Down/Up from the filter box
        # (and Tab) can land on them, and Enter activates the focused row.
        self.setFocusPolicy(Qt.StrongFocus)

        # Tooltip with doc / ports / params, same logic as the original widget
        self._build_tooltip()

        # Layout: 8px gutter · 6px dot · 6px gap · 22px glyph · 8px gap · name · stretch
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 0, 10, 0)
        lay.setSpacing(8)

        self.dot = _CategoryDot(category_name)
        lay.addWidget(self.dot, 0, Qt.AlignVCenter)

        self.glyph = _BlockGlyphLabel(menu_block, colors)
        lay.addWidget(self.glyph, 0, Qt.AlignVCenter)

        self.name_label = QLabel(getattr(menu_block, 'fn_name', '—'))
        font = QFont()
        font.setPointSize(9)
        font.setStyleHint(QFont.SansSerif)
        self.name_label.setFont(font)
        self.name_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        lay.addWidget(self.name_label, 1, Qt.AlignVCenter)

        self._apply_styling()
        # Theme changes are applied centrally by the owning palette
        # (ModernBlockPalette._on_theme_changed), not a per-row connection.

    def _build_tooltip(self):
        try:
            doc_lines = []
            block_cls = getattr(self.menu_block, 'block_class', None)
            inst = block_cls() if block_cls else None
            if inst and hasattr(inst, 'doc'):
                doc_lines.append(str(inst.doc))
            def _names(seq):
                try:
                    return ", ".join([i.get('name', '') for i in seq if isinstance(i, dict)]) or "—"
                except Exception:
                    return "—"
            if inst:
                doc_lines.append(f"Inputs:  {_names(getattr(inst, 'inputs', []))}")
                doc_lines.append(f"Outputs: {_names(getattr(inst, 'outputs', []))}")
                params = getattr(self.menu_block, 'param_meta', getattr(inst, 'params', {}))
                if isinstance(params, dict) and params:
                    keys = list(params.keys())
                    doc_lines.append("Params:  " + ", ".join(keys[:6]) + ("…" if len(keys) > 6 else ""))
            tip = "\n".join([l for l in doc_lines if l])
            if tip:
                self.setToolTip(tip)
        except Exception:
            pass

    # -- Styling ------------------------------------------------------------

    def _apply_styling(self):
        bg = theme_manager.get_color('palette_item_bg').name()
        bg_hover = theme_manager.get_color('palette_item_hover').name()
        text = theme_manager.get_color('text_primary').name()
        accent = theme_manager.get_color('accent_primary').name()
        # Slightly transparent border so dark/light read fine
        border = theme_manager.get_color('border_primary').name()

        self.setStyleSheet(f"""
            QFrame#PaletteRow {{
                background-color: transparent;
                border: 1px solid transparent;
                border-radius: 6px;
            }}
            QFrame#PaletteRow:hover {{
                background-color: {bg_hover};
                border: 1px solid {border};
            }}
            QFrame#PaletteRow:focus {{
                background-color: {bg_hover};
                border: 1px solid {accent};
            }}
            QLabel {{
                color: {text};
                background: transparent;
            }}
        """)
        # Per-Qt-version safety: set the label color directly too, since the
        # QSS cascade from a per-row stylesheet does not always reach the
        # child QLabel on Qt 5.9 (renders as default black).
        if hasattr(self, 'name_label'):
            self.name_label.setStyleSheet(f"color: {text}; background: transparent;")

    def _on_theme_changed(self):
        self._apply_styling()
        self.dot.update()
        self.glyph.update()

    # -- Drag ---------------------------------------------------------------

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.setCursor(Qt.ClosedHandCursor)
            self.drag_start_position = event.pos()

    def mouseMoveEvent(self, event):
        if not (event.buttons() & Qt.LeftButton):
            return
        if not hasattr(self, 'drag_start_position'):
            return
        if (event.pos() - self.drag_start_position).manhattanLength() < 10:
            return
        self._start_drag(event)

    def mouseReleaseEvent(self, event):
        self.setCursor(Qt.OpenHandCursor)

    # -- Keyboard navigation ------------------------------------------------

    def keyPressEvent(self, event):
        """Enter activates the row; Down/Up move focus to sibling rows.

        Focus traversal hops over hidden rows (filtered out, or inside a
        collapsed category) so the keyboard walk matches what is on screen.
        """
        key = event.key()
        if key in (Qt.Key_Return, Qt.Key_Enter):
            self.activate()
            event.accept()
            return
        if key == Qt.Key_Down:
            self._focus_sibling(+1)
            event.accept()
            return
        if key == Qt.Key_Up:
            self._focus_sibling(-1)
            event.accept()
            return
        super().keyPressEvent(event)

    def _focus_sibling(self, direction: int):
        """Move focus to the next/previous visible row in the owning palette."""
        palette = self._owning_palette()
        if palette is None:
            return
        rows = [r for r in palette.findChildren(CompactBlockRow) if r.isVisible()]
        if self not in rows:
            return
        idx = rows.index(self) + direction
        if 0 <= idx < len(rows):
            rows[idx].setFocus(Qt.OtherFocusReason)

    def _owning_palette(self):
        """Walk up to the ModernBlockPalette that owns this row, if any."""
        w = self.parent()
        while w is not None and not isinstance(w, ModernBlockPalette):
            w = w.parent()
        return w

    def activate(self):
        """Add this block to the canvas — same effect as a click/drag-add.

        Emits the row-level ``block_drag_started`` (carrying the menu_block and
        no specific position) and asks the owning palette to re-emit its
        public ``block_drag_started(object)`` signal so existing consumers fire
        without needing a real drag gesture.
        """
        self.block_drag_started.emit(self.menu_block, None)
        palette = self._owning_palette()
        if palette is not None:
            palette.block_drag_started.emit(self.menu_block)

    def _start_drag(self, event):
        try:
            drag = QDrag(self)
            mime = QMimeData()
            mime.setText(f"diablo_block:{getattr(self.menu_block, 'fn_name', 'Unknown')}")
            drag.menu_block = self.menu_block

            # Drag preview: render a full block via existing BlockRenderer
            pix = self._drag_pixmap()
            drag.setPixmap(pix)
            drag.setHotSpot(QPoint(pix.width() // 2, pix.height() // 2))
            drag.setMimeData(mime)
            drag.exec_(Qt.CopyAction | Qt.MoveAction, Qt.CopyAction)
        except Exception as e:
            logger.error(f"Drag start failed: {e}")

    def _drag_pixmap(self) -> QPixmap:
        try:
            from lib.simulation.block import DBlock
            from modern_ui.renderers.block_renderer import BlockRenderer

            w, h = 80, 48
            pix = QPixmap(w, h)
            pix.fill(Qt.transparent)
            p = QPainter(pix)
            p.setOpacity(0.85)
            p.setRenderHint(QPainter.Antialiasing, True)

            mb = self.menu_block
            tmp = DBlock(
                block_fn=mb.block_fn, sid=0,
                coords=QRect(6, 4, w - 12, h - 8),
                color=mb.b_color, in_ports=mb.ins, out_ports=mb.outs,
                b_type=mb.b_type, io_edit=mb.io_edit, fn_name=mb.fn_name,
                params=mb.params, external=mb.external, colors=self.colors,
                block_class=getattr(mb, 'block_class', None),
            )
            tmp.update_Block()
            BlockRenderer().draw_block(tmp, p, draw_name=False, draw_ports=True)
            p.end()
            return pix
        except Exception as e:
            logger.warning(f"drag pixmap fallback: {e}")
            pix = QPixmap(72, 28)
            pix.fill(theme_manager.get_color('accent_primary'))
            return pix


class _CategoryDot(QWidget):
    """6px dot showing the row's category color."""
    SIZE = 8

    def __init__(self, category, parent=None):
        super().__init__(parent)
        self._cat = category
        self.setFixedSize(QSize(self.SIZE, self.SIZE))
        self.setAttribute(Qt.WA_TransparentForMouseEvents)

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(Qt.NoPen)
        p.setBrush(_category_dot_color(self._cat))
        p.drawEllipse(0, 0, self.SIZE, self.SIZE)


class _BlockGlyphLabel(QWidget):
    """22px flat thumbnail — category-colored chip with a geometric glyph.

    Replaces the previous BlockRenderer-at-22px which painted a gradient +
    drop-shadow combo that turned to mud at this size, especially in dark
    theme. This widget paints a flat rounded chip in the block's category
    accent color, with a single-stroke geometric icon on top — readable in
    both themes.
    """

    SIZE = 22

    def __init__(self, menu_block, colors, parent=None):
        super().__init__(parent)
        self.menu_block = menu_block
        self.colors = colors
        # Resolve the category once (it may instantiate the block class) so
        # paintEvent does not reconstruct a block on every repaint.
        self._category = _block_category_name(menu_block)
        self._glyph_kind = _glyph_kind_for(getattr(menu_block, 'fn_name', '') or '')
        self.setFixedSize(QSize(self.SIZE, self.SIZE))
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        # Repainted on theme change by the owning CompactBlockRow.

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)

        cat = self._category
        bg, accent, fg = _category_chip_colors(cat)

        # Chip
        p.setPen(QPen(accent, 1))
        p.setBrush(bg)
        p.drawRoundedRect(QRectF(1, 1, self.SIZE - 2, self.SIZE - 2), 5, 5)

        # Glyph
        pen = QPen(fg)
        pen.setWidthF(1.4)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)

        kind = self._glyph_kind
        _draw_glyph(p, kind, fg, self.SIZE)
        p.end()


def _block_category_name(menu_block) -> str:
    """Best-effort category lookup from menu_block (class.category, else keyword match)."""
    cls = getattr(menu_block, 'block_class', None)
    if cls:
        try:
            inst = cls()
            cat = getattr(inst, 'category', None)
            if cat:
                return str(cat)
        except Exception:
            pass
    name = getattr(menu_block, 'fn_name', '').lower()
    if any(k in name for k in ['sine', 'step', 'ramp', 'impulse', 'constant', 'noise', 'prbs', 'wave']):
        return 'Sources'
    if any(k in name for k in ['sum', 'gain', 'product', 'abs', 'sqrt', 'matrix', 'mathfunction', 'exp']):
        return 'Math'
    if any(k in name for k in ['integ', 'deriv', 'pid', 'delay', 'tranfn', 'transfer', 'state', 'lqr', 'controller']):
        return 'Control'
    if any(k in name for k in ['saturation', 'rate', 'hysteresis', 'deadband', 'filter']):
        return 'Filters'
    if any(k in name for k in ['scope', 'display', 'bode', 'nyquist', 'rootlocus', 'fft', 'export', 'term', 'xygraph']):
        return 'Sinks'
    if any(k in name for k in ['mux', 'demux', 'switch', 'selector', 'goto', 'from', 'port', 'sub']):
        return 'Routing'
    return 'Other'


def _category_chip_colors(cat: str):
    """Return (background QColor, accent border QColor, foreground glyph QColor).

    Chip background is a low-alpha tint of the accent so it reads on both
    light and dark panel backgrounds without needing different palettes.
    """
    c = (cat or '').lower()
    accent_key = 'text_secondary'
    if 'source' in c:
        accent_key = 'block_source_accent'
    elif 'math' in c:
        accent_key = 'block_process_accent'
    elif 'control' in c or 'continuous' in c:
        accent_key = 'block_control_accent'
    elif 'filter' in c:
        accent_key = 'block_other_accent'
    elif 'sink' in c:
        accent_key = 'block_sink_accent'
    elif 'rout' in c or 'discrete' in c:
        accent_key = 'block_other_accent'

    accent = theme_manager.get_color(accent_key)
    bg = QColor(accent); bg.setAlpha(48)
    # Glyph color: on dark theme use a lightened accent for max readability;
    # on light theme use a darker variant.
    if theme_manager.current_theme.value == 'dark':
        fg = QColor(accent).lighter(125)
    else:
        fg = QColor(accent).darker(140)
    return bg, accent, fg


def _glyph_kind_for(fn_name: str) -> str:
    """Map an fn_name to a glyph kind. Falls back to letter:<initials>."""
    n = (fn_name or '').lower()
    table = [
        ('sine', 'sine'), ('wave', 'sine'),
        ('noise', 'noise'), ('prbs', 'noise'),
        ('step', 'step'), ('ramp', 'ramp'),
        ('impulse', 'impulse'), ('constant', 'const'),
        ('sum', 'sum'), ('gain', 'gain'), ('product', 'product'),
        ('abs', 'abs'), ('sqrt', 'sqrt'), ('matrixgain', 'matrix'),
        ('mathfunction', 'fn'), ('exp', 'exp'),
        ('integ', 'integ'), ('deriv', 'deriv'),
        ('pid', 'pid'), ('tranfn', 'tranfn'), ('transfer', 'tranfn'),
        ('state', 'state'), ('lqr', 'lqr'),
        ('delay', 'delay'),
        ('saturation', 'sat'), ('rate', 'rate'), ('hysteresis', 'hys'),
        ('deadband', 'dead'), ('filter', 'filter'),
        ('scope', 'scope'), ('display', 'display'),
        ('bode', 'bode'), ('nyquist', 'nyq'), ('rootlocus', 'roots'),
        ('fft', 'fft'), ('xygraph', 'xy'),
        ('export', 'export'), ('term', 'term'),
        ('zoh', 'zoh'), ('firstorder', 'foh'), ('zero', 'zoh'), ('hold', 'zoh'),
        ('mux', 'mux'), ('demux', 'demux'),
        ('switch', 'switch'), ('selector', 'sel'),
        ('goto', 'goto'), ('from', 'from'),
        ('inport', 'in'), ('outport', 'out'),
        ('sub', 'sub'),
    ]
    for key, glyph in table:
        if key in n:
            return glyph
    return 'letter:' + (n[:2].upper() if n else '?')


def _draw_glyph(p: QPainter, kind: str, color: QColor, s: int):
    """Single-stroke geometric glyph in an s×s box. Pen is already configured."""
    pad = 5

    def L(x1, y1, x2, y2):
        p.drawLine(QPointF(x1, y1), QPointF(x2, y2))

    R = QRectF(pad, pad, s - 2 * pad, s - 2 * pad)

    if kind == 'sine':
        path = QPainterPath()
        path.moveTo(pad, s / 2)
        path.quadTo(pad + (s - 2 * pad) * 0.25, pad, s / 2, s / 2)
        path.quadTo(s - pad - (s - 2 * pad) * 0.25, s - pad, s - pad, s / 2)
        p.drawPath(path)
    elif kind == 'noise':
        pts = [(pad, s * 0.7), (pad + 2.5, s * 0.35), (pad + 4.5, s * 0.6),
               (pad + 6.5, s * 0.3), (pad + 8.5, s * 0.65), (s - pad, s * 0.45)]
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            L(x1, y1, x2, y2)
    elif kind == 'step':
        L(pad, s - pad, s / 2, s - pad); L(s / 2, s - pad, s / 2, pad); L(s / 2, pad, s - pad, pad)
    elif kind == 'ramp':
        L(pad, s - pad, s - pad, pad); L(pad, s - pad, s - pad, s - pad)
    elif kind == 'impulse':
        L(pad, s - pad, s - pad, s - pad); L(s / 2, pad, s / 2, s - pad)
    elif kind == 'sum':
        p.drawEllipse(R)
        _draw_text(p, color, s, '±', italic=False, weight=QFont.Bold)
    elif kind == 'gain':
        path = QPainterPath()
        path.moveTo(pad, pad); path.lineTo(s - pad, s / 2); path.lineTo(pad, s - pad); path.closeSubpath()
        p.drawPath(path)
    elif kind == 'product':
        p.drawEllipse(R)
        L(pad + 2, pad + 2, s - pad - 2, s - pad - 2)
        L(s - pad - 2, pad + 2, pad + 2, s - pad - 2)
    elif kind == 'integ':
        L(pad + 1, pad, pad + 1, s - pad)
        _draw_text(p, color, s, '1/s', italic=True, x_off=2.5, size_factor=0.4)
    elif kind == 'tranfn':
        _draw_text(p, color, s, 'B', italic=True, y_off=-3.5, size_factor=0.42)
        L(pad + 1.5, s / 2, s - pad - 1.5, s / 2)
        _draw_text(p, color, s, 'A', italic=True, y_off=4.5, size_factor=0.42)
    elif kind == 'delay':
        p.drawEllipse(R); _draw_text(p, color, s, '→', italic=False)
    elif kind == 'sat':
        L(pad, s - pad, pad + 4, s - pad); L(pad + 4, s - pad, s - pad - 4, pad); L(s - pad - 4, pad, s - pad, pad)
    elif kind == 'rate':
        L(pad, s - pad, s / 2, pad); L(s / 2, pad, s - pad, pad)
    elif kind in ('hys', 'dead'):
        L(pad, s / 2, s / 2 - 1, s / 2); L(s / 2 - 1, s / 2, s / 2 + 1, pad + 2)
        L(s / 2 + 1, pad + 2, s - pad, pad + 2)
    elif kind == 'filter':
        path = QPainterPath()
        path.moveTo(pad, s - pad); path.quadTo(s / 2, s - pad, s / 2, s / 2)
        path.quadTo(s / 2, pad, s - pad, pad)
        p.drawPath(path)
    elif kind == 'scope':
        p.drawRoundedRect(QRectF(pad - 1, pad + 1, s - 2 * pad + 2, s - 2 * pad - 2), 2, 2)
        path = QPainterPath()
        path.moveTo(pad + 1, s / 2 + 2)
        path.quadTo(pad + 4, pad + 2, s / 2, s / 2)
        path.quadTo(s - pad - 3, s - pad - 1, s - pad - 1, s / 2 - 1)
        p.drawPath(path)
    elif kind == 'bode':
        L(pad, pad, pad, s - pad); L(pad, s - pad, s - pad, s - pad)
        path = QPainterPath()
        path.moveTo(pad + 1, pad + 3); path.quadTo(s / 2, pad + 4, s / 2 + 1, s / 2)
        path.quadTo(s - pad - 2, s - pad - 4, s - pad, s - pad - 2)
        p.drawPath(path)
    elif kind == 'nyq':
        p.drawEllipse(R); L(pad, s / 2, s - pad, s / 2); L(s / 2, pad, s / 2, s - pad)
    elif kind == 'roots':
        p.drawEllipse(R); L(pad, s / 2, s - pad, s / 2)
        L(s * 0.3, s / 2 - 1.5, s * 0.3 + 1.5, s / 2 + 0.5)
        L(s * 0.3, s / 2 + 0.5, s * 0.3 + 1.5, s / 2 - 1.5)
    elif kind == 'fft':
        for i, h in enumerate([4, 7, 3, 8, 5, 6, 2, 5]):
            x = pad + i * (s - 2 * pad) / 8.0
            L(x, s - pad, x, s - pad - h)
    elif kind == 'xy':
        L(pad, s - pad, s - pad, s - pad); L(pad, s - pad, pad, pad)
        L(pad + 1, s - pad - 1, s - pad - 1, pad + 1)
    elif kind == 'export':
        p.drawRoundedRect(QRectF(pad, pad + 1, (s - 2 * pad) * 0.55, s - 2 * pad - 2), 1.2, 1.2)
        L(pad + 7, s / 2, s - pad - 1, s / 2)
        L(s - pad - 3, s / 2 - 2, s - pad - 1, s / 2)
        L(s - pad - 3, s / 2 + 2, s - pad - 1, s / 2)
    elif kind == 'term':
        L(pad + 2, s - pad - 1, s - pad - 2, s - pad - 1)
        L(pad + 4, s - pad - 3, s - pad - 4, s - pad - 3)
        L(s / 2, pad + 1, s / 2, s - pad - 3)
    elif kind == 'zoh':
        L(pad, s * 0.72, pad + 4, s * 0.72)
        L(pad + 4, s * 0.72, pad + 4, s / 2)
        L(pad + 4, s / 2, s - pad - 4, s / 2)
        L(s - pad - 4, s / 2, s - pad - 4, pad + 2)
        L(s - pad - 4, pad + 2, s - pad, pad + 2)
    elif kind == 'foh':
        L(pad, s * 0.7, pad + 5, s * 0.4); L(pad + 5, s * 0.4, s - pad - 5, s * 0.7)
        L(s - pad - 5, s * 0.7, s - pad, s * 0.3)
    elif kind == 'mux':
        L(pad, pad + 1, s / 2, s / 2); L(pad, s - pad - 1, s / 2, s / 2); L(s / 2, s / 2, s - pad, s / 2)
    elif kind == 'demux':
        L(s - pad, pad + 1, s / 2, s / 2); L(s - pad, s - pad - 1, s / 2, s / 2); L(s / 2, s / 2, pad, s / 2)
    elif kind == 'switch':
        L(pad, s / 2, s / 2 - 1, s / 2); L(s / 2 + 1, pad + 2, s - pad, s / 2)
    elif kind in ('in', 'out'):
        L(pad, s / 2, s - pad, s / 2)
        if kind == 'in':
            L(s - pad - 3, s / 2 - 2, s - pad, s / 2); L(s - pad - 3, s / 2 + 2, s - pad, s / 2)
        else:
            L(pad + 3, s / 2 - 2, pad, s / 2); L(pad + 3, s / 2 + 2, pad, s / 2)
    elif kind == 'sub':
        p.drawRoundedRect(QRectF(pad, pad, s - 2 * pad, s - 2 * pad), 1.5, 1.5)
        p.drawRoundedRect(QRectF(pad + 2, pad + 2, (s - 2 * pad) - 4, (s - 2 * pad) - 4), 1.5, 1.5)
    elif kind.startswith('letter:'):
        letters = kind.split(':', 1)[1] or '?'
        _draw_text(p, color, s, letters, italic=False, weight=QFont.Bold, size_factor=0.42)
    else:
        # Final fallback — short label from kind
        _draw_text(p, color, s, kind[:3], italic=False, weight=QFont.Bold, size_factor=0.36)


def _draw_text(p: QPainter, color: QColor, s: int, text: str,
               italic: bool = False, weight=None, size_factor: float = 0.55,
               x_off: float = 0, y_off: float = 0):
    f = QFont()
    if weight is not None:
        f.setWeight(weight)
    f.setItalic(italic)
    f.setPointSizeF(max(7.0, s * size_factor))
    if italic:
        f.setStyleHint(QFont.Serif)
    p.setFont(f)
    p.setPen(QPen(color))
    p.drawText(QRectF(0 + x_off, 0 + y_off, s, s), Qt.AlignCenter, text)


# -----------------------------------------------------------------------------
# Category section — slim header + rows
# -----------------------------------------------------------------------------

class _CategorySection(QWidget):
    """Collapsible category: a clickable header toggles its rows.

    The collapsed state is persisted per-category via QSettings (keyed by
    category name) and restored when the section is rebuilt, so a user's
    expand/collapse choices survive ``refresh_blocks()`` and app restarts.
    """

    def __init__(self, category_name, menu_blocks, colors, parent=None):
        super().__init__(parent)
        self.category_name = category_name
        self.menu_blocks = menu_blocks
        self.colors = colors
        # Last filter text, so a header toggle re-applies the active filter
        # instead of unconditionally showing every row.
        self._filter_text = ''

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 6, 0, 2)
        lay.setSpacing(0)

        # Restore the persisted collapsed flag before building the header so
        # the chevron and initial row visibility match the stored state.
        self._collapsed = self._load_collapsed()

        # Header — clickable label that toggles the section. The chevron prefix
        # indicates state (▾ expanded / ▸ collapsed).
        header = _CategoryHeaderLabel(self)
        f = QFont()
        f.setPointSize(8)
        f.setBold(True)
        f.setLetterSpacing(QFont.PercentageSpacing, 110)
        header.setFont(f)
        header.setContentsMargins(12, 2, 8, 4)
        header.setObjectName("PaletteCategoryHeader")
        header.setCursor(Qt.PointingHandCursor)
        lay.addWidget(header)
        self.header = header

        # Rows
        self.rows = []
        for mb in menu_blocks:
            row = CompactBlockRow(mb, category_name, colors)
            lay.addWidget(row)
            self.rows.append(row)

        self._refresh_header_text()
        self._apply_row_visibility()
        self._apply_styling()
        # Restyled on theme change by ModernBlockPalette._on_theme_changed.

    # -- Collapse / persistence --------------------------------------------

    def _load_collapsed(self) -> bool:
        try:
            settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
            return bool(settings.value(
                _collapsed_settings_key(self.category_name), False, type=bool
            ))
        except Exception:
            return False

    def _save_collapsed(self):
        try:
            settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
            settings.setValue(
                _collapsed_settings_key(self.category_name), self._collapsed
            )
        except Exception:
            pass

    def toggle_collapsed(self):
        """Flip collapsed state, persist it, and update the header + rows."""
        self._collapsed = not self._collapsed
        self._save_collapsed()
        self._refresh_header_text()
        self._apply_row_visibility()

    def is_collapsed(self) -> bool:
        return self._collapsed

    def _refresh_header_text(self):
        chevron = _CHEVRON_COLLAPSED if self._collapsed else _CHEVRON_EXPANDED
        self.header.setText(f"{chevron}  {self.category_name.upper()}")

    def _apply_row_visibility(self) -> bool:
        """Show rows that pass the active filter, unless the section is collapsed.

        Returns whether any row matches the filter (independent of collapse), so
        ``filter`` can set section visibility from the same single pass.
        """
        text = self._filter_text
        any_match = False
        for r in self.rows:
            name = getattr(r.menu_block, 'fn_name', '').lower()
            matches = (not text) or (text in name)
            r.setVisible(matches and not self._collapsed)
            any_match = any_match or matches
        return any_match

    # -- Styling ------------------------------------------------------------

    def _apply_styling(self):
        muted = theme_manager.get_color('text_secondary').name()
        self.findChild(QLabel, "PaletteCategoryHeader").setStyleSheet(
            f"QLabel#PaletteCategoryHeader {{ color: {muted}; background: transparent; }}"
        )

    def filter(self, text: str) -> bool:
        """Apply a filter; return True if the section stays visible.

        A collapsed section keeps its rows hidden but still reports whether any
        row matches, so the header remains visible (and toggleable) under an
        active filter.
        """
        self._filter_text = text
        any_match = self._apply_row_visibility()
        self.setVisible(any_match)
        return any_match


class _CategoryHeaderLabel(QLabel):
    """Category header label that toggles its owning section when clicked."""

    def __init__(self, section, parent=None):
        super().__init__(parent)
        self._section = section

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._section.toggle_collapsed()
            event.accept()
            return
        super().mousePressEvent(event)


# -----------------------------------------------------------------------------
# Top-level palette
# -----------------------------------------------------------------------------

class ModernBlockPalette(QWidget):
    """Modern block palette — compact single-column layout."""

    block_drag_started = pyqtSignal(object)

    def __init__(self, dsim, parent=None):
        super().__init__(parent)
        self.dsim = dsim
        self._sections = []
        self._setup_widget()
        self._load_blocks()
        self._apply_styling()
        theme_manager.theme_changed.connect(self._on_theme_changed)
        logger.info("Modern block palette initialized (compact V1)")

    def _on_theme_changed(self, *_):
        """Single theme-change handler for the whole palette.

        Restyles the palette chrome, then fans out to live children. Rows,
        glyphs, dots and section headers no longer subscribe to
        theme_manager.theme_changed individually: one connection here replaces
        ~150 per-widget connections (with ~70 blocks) and removes the
        dangling-connection lifecycle that refresh_blocks() had to clean up.
        findChildren reflects the current tree, so dynamically added/removed
        rows are handled automatically.
        """
        self._apply_styling()
        for section in self.findChildren(_CategorySection):
            section._apply_styling()
        for row in self.findChildren(CompactBlockRow):
            row._on_theme_changed()

    # -- Layout -------------------------------------------------------------

    def _setup_widget(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Header — "Library" + count
        head = QFrame()
        head.setObjectName("PaletteHead")
        hl = QHBoxLayout(head)
        hl.setContentsMargins(12, 10, 12, 6)
        hl.setSpacing(8)
        self.title = QLabel("Library")
        tf = QFont(); tf.setPointSize(9); tf.setBold(True)
        tf.setLetterSpacing(QFont.PercentageSpacing, 105)
        self.title.setFont(tf)
        hl.addWidget(self.title)
        hl.addStretch(1)
        self.count_label = QLabel("")
        cf = QFont(); cf.setPointSize(9)
        self.count_label.setFont(cf)
        hl.addWidget(self.count_label)
        outer.addWidget(head)

        # Search
        search_wrap = QFrame()
        sw = QHBoxLayout(search_wrap)
        sw.setContentsMargins(10, 0, 10, 8)
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Filter blocks…")
        self.search_bar.setClearButtonEnabled(True)
        self.search_bar.textChanged.connect(self._filter_blocks)
        # Down/Up from the search box step into the visible rows (keyboard nav).
        self.search_bar.installEventFilter(self)
        sw.addWidget(self.search_bar)
        outer.addWidget(search_wrap)

        # Scrollable list
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll.setFrameShape(QFrame.NoFrame)

        self.blocks_container = QWidget()
        self.blocks_container.setObjectName("PaletteBlocksContainer")
        self.blocks_container.setAttribute(Qt.WA_StyledBackground, True)
        self.blocks_layout = QVBoxLayout(self.blocks_container)
        self.blocks_layout.setContentsMargins(4, 0, 4, 12)
        self.blocks_layout.setSpacing(0)

        self.scroll.setWidget(self.blocks_container)
        self.scroll.viewport().setAutoFillBackground(False)
        outer.addWidget(self.scroll, 1)

    # -- Blocks -------------------------------------------------------------

    def _load_blocks(self):
        try:
            menu_blocks = getattr(self.dsim, 'menu_blocks', []) or []
            if not menu_blocks:
                logger.warning("No menu blocks found in DSim")
                self._add_placeholder()
                return

            categories = self._categorize_blocks(menu_blocks)
            total = 0
            for cat_name, blocks in categories.items():
                section = _CategorySection(cat_name, blocks, self.dsim.colors)
                self.blocks_layout.addWidget(section)
                self._sections.append(section)
                total += len(blocks)

            self.blocks_layout.addStretch(1)
            self.count_label.setText(str(total))
            logger.info(f"Loaded {total} blocks in {len(categories)} categories")
        except Exception as e:
            logger.error(f"Error loading blocks: {e}")
            self._add_placeholder()

    def _categorize_blocks(self, menu_blocks):
        # Same logic as the original
        categories = {
            "Sources": [],
            "Math": [],
            "Control": [],
            "Filters": [],
            "Sinks": [],
            "Routing": [],
            "Optimization Primitives": [],
            "Other": [],
        }
        source_kw = ['step', 'ramp', 'sine', 'square', 'constant', 'source', 'noise', 'prbs', 'impulse']
        math_kw = ['sum', 'gain', 'multiply', 'add', 'subtract', 'divide', 'abs', 'sqrt', 'product', 'matrix']
        control_kw = ['integrator', 'derivative', 'pid', 'controller', 'delay', 'transfer', 'state', 'lqr']
        filter_kw = ['filter', 'lowpass', 'highpass', 'bandpass', 'hysteresis', 'saturation', 'deadband', 'rate']
        sink_kw = ['scope', 'display', 'sink', 'plot', 'output', 'export', 'term', 'bode', 'nyquist', 'rootlocus', 'fft', 'xygraph']
        routing_kw = ['mux', 'demux', 'switch', 'selector', 'goto', 'from', 'inport', 'outport', 'sub']

        for b in menu_blocks:
            cat = None
            cls = getattr(b, 'block_class', None)
            if cls:
                try:
                    inst = cls()
                    if hasattr(inst, 'category'):
                        cat = inst.category
                except Exception:
                    pass
            if not cat:
                name = getattr(b, 'fn_name', '').lower()
                for kws, label in [
                    (source_kw, "Sources"), (math_kw, "Math"),
                    (control_kw, "Control"), (filter_kw, "Filters"),
                    (sink_kw, "Sinks"), (routing_kw, "Routing"),
                ]:
                    if any(k in name for k in kws):
                        cat = label
                        break
                cat = cat or "Other"
            categories.setdefault(cat, []).append(b)

        return {k: v for k, v in categories.items() if v}

    def _add_placeholder(self):
        p = QLabel("No blocks available.\nCheck DSim initialization.")
        p.setAlignment(Qt.AlignCenter)
        p.setWordWrap(True)
        p.setStyleSheet(
            f"color:{theme_manager.get_color('text_secondary').name()}; "
            f"font-style:italic; padding:20px;"
        )
        self.blocks_layout.addWidget(p)

    # -- Filtering ----------------------------------------------------------

    def _filter_blocks(self, text):
        text = (text or '').strip().lower()
        for s in self._sections:
            s.filter(text)

    # -- Keyboard navigation ------------------------------------------------

    def visible_rows(self):
        """Currently-visible block rows, in on-screen (top-to-bottom) order."""
        return [r for r in self.findChildren(CompactBlockRow) if r.isVisible()]

    def eventFilter(self, obj, event):
        """Route Down/Up from the search box into the visible row list.

        Down focuses the first visible row, Up the last, so the user can leave
        the filter box and walk the results without the mouse. Once on a row,
        CompactBlockRow.keyPressEvent handles further arrow/Enter navigation.
        """
        from PyQt5.QtCore import QEvent
        if obj is self.search_bar and event.type() == QEvent.KeyPress:
            key = event.key()
            if key in (Qt.Key_Down, Qt.Key_Up):
                rows = self.visible_rows()
                if rows:
                    target = rows[0] if key == Qt.Key_Down else rows[-1]
                    target.setFocus(Qt.OtherFocusReason)
                    return True
        return super().eventFilter(obj, event)

    # -- Styling ------------------------------------------------------------

    def _apply_styling(self):
        bg = theme_manager.get_color('palette_bg').name()
        text = theme_manager.get_color('text_primary').name()
        text2 = theme_manager.get_color('text_secondary').name()
        border = theme_manager.get_color('border_primary').name()
        search_bg = theme_manager.get_color('surface').name()
        scroll_track = theme_manager.get_color('surface_secondary').name()
        scroll_thumb = theme_manager.get_color('border_secondary').name()

        self.setStyleSheet(f"""
            ModernBlockPalette {{
                background-color: {bg};
                border: 1px solid {border};
                border-radius: 8px;
            }}
            QLabel {{ color: {text}; background: transparent; }}
            QFrame#PaletteHead {{ background: transparent; }}
            QLineEdit {{
                background-color: {search_bg};
                color: {text};
                border: 1px solid {border};
                border-radius: 6px;
                padding: 5px 9px;
                selection-background-color: {theme_manager.get_color('accent_primary').name()};
            }}
            QLineEdit:focus {{ border-color: {theme_manager.get_color('border_focus').name()}; }}
            QScrollArea {{ border: none; background-color: {bg}; }}
            QWidget#PaletteBlocksContainer {{ background-color: {bg}; }}
            QScrollArea > QWidget > QWidget {{ background-color: {bg}; }}
            QScrollBar:vertical {{
                background: {scroll_track};
                width: 10px;
                border-radius: 5px;
                margin: 0px;
            }}
            QScrollBar::handle:vertical {{
                background: {scroll_thumb};
                border-radius: 5px;
                min-height: 24px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {theme_manager.get_color('border_hover').name()};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}
        """)
        # Style the small count badge
        self.count_label.setStyleSheet(
            f"color:{text2}; background: transparent; padding: 2px 6px;"
        )

    # -- Public API (preserved) --------------------------------------------

    def refresh_blocks(self):
        try:
            for i in reversed(range(self.blocks_layout.count())):
                item = self.blocks_layout.itemAt(i)
                w = item.widget()
                if w:
                    # Detach from the layout and schedule for deletion. Theme
                    # restyling is now driven centrally via findChildren in
                    # _on_theme_changed, so detached widgets (parent cleared
                    # below) are simply excluded from the next fan-out.
                    self.blocks_layout.removeWidget(w)
                    w.setParent(None)
                    w.deleteLater()
                else:
                    self.blocks_layout.removeItem(item)
            self._sections.clear()
            self._load_blocks()
        except Exception as e:
            logger.error(f"Error refreshing palette: {e}")

    def get_available_blocks(self):
        return getattr(self.dsim, 'menu_blocks', [])
