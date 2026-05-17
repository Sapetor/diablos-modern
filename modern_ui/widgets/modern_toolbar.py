"""
Modern Toolbar Widget for DiaBloS — V1 "Tightened" redesign

Drop-in replacement preserving the public API of the original:
  Signals  : new_diagram, open_diagram, save_diagram,
             play_simulation, pause_simulation, stop_simulation, step_simulation,
             plot_results, capture_screen, auto_route_wires,
             zoom_changed (float), theme_toggled,
             command_palette_requested  (NEW — optional, safe to ignore)
  Methods  : set_simulation_state(running: bool, paused: bool = False)
             set_status(msg: str)        — also updates the right-side pill
             get_zoom_factor() -> float
             set_zoom_factor(factor: float)

Visual changes
--------------
* Icons are drawn as QPainterPath shapes (crisp on Retina, identical on Win/macOS/Linux).
* Compact 28×28 icon buttons replace 60×40 emoji+label buttons.
* Transport group (Play/Pause/Stop/Step) is centered, with a monospace
  `t = 0.000 / 10.0 s` readout right next to it.
* A colored status pill sits on the right ("Ready", "Simulating…", "Paused").
* Zoom is a tight rocker (−  100% +) instead of a 70px slider + label.
* ⌘K command-palette button at the far right (emits a signal).
"""

from __future__ import annotations

from PyQt5.QtWidgets import (
    QToolBar, QAction, QWidget, QHBoxLayout, QLabel, QPushButton, QToolButton,
    QSizePolicy, QFrame
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QRectF, QPointF
from PyQt5.QtGui import (
    QIcon, QPixmap, QPainter, QColor, QPen, QBrush, QPainterPath, QFont, QPolygonF
)
from modern_ui.themes.theme_manager import theme_manager


# ----------------------------------------------------------------------------- 
# Icon builder — geometric, theme-aware, DPI-safe
# -----------------------------------------------------------------------------

def _make_icon(kind: str, size: int = 18, color: str | None = None) -> QIcon:
    """Build a QIcon from a small geometric path. ``kind`` is one of:
    new, open, save, play, pause, stop, step, plot, capture, route,
    sun, moon, search, plus, minus.
    """
    if color is None:
        color = theme_manager.get_color('text_primary').name()

    # Render at 2× for DPI safety, let Qt downscale.
    px = QPixmap(size * 2, size * 2)
    px.fill(Qt.transparent)
    p = QPainter(px)
    p.setRenderHint(QPainter.Antialiasing, True)
    p.scale(2.0, 2.0)

    pen = QPen(QColor(color))
    pen.setWidthF(1.6)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)

    s = size
    pad = 3
    rect = QRectF(pad, pad, s - 2 * pad, s - 2 * pad)

    def line(x1, y1, x2, y2):
        p.drawLine(QPointF(x1, y1), QPointF(x2, y2))

    if kind == 'new':
        # Document with folded corner
        path = QPainterPath()
        path.moveTo(pad + 1, pad)
        path.lineTo(s - pad - 3, pad)
        path.lineTo(s - pad, pad + 3)
        path.lineTo(s - pad, s - pad)
        path.lineTo(pad + 1, s - pad)
        path.closeSubpath()
        p.drawPath(path)
        line(s - pad - 3, pad, s - pad - 3, pad + 3)
        line(s - pad - 3, pad + 3, s - pad, pad + 3)

    elif kind == 'open':
        # Folder
        path = QPainterPath()
        path.moveTo(pad, pad + 3)
        path.lineTo(pad + 4, pad + 3)
        path.lineTo(pad + 5.5, pad + 1.5)
        path.lineTo(s - pad, pad + 1.5)
        path.lineTo(s - pad, s - pad)
        path.lineTo(pad, s - pad)
        path.closeSubpath()
        p.drawPath(path)

    elif kind == 'save':
        # Floppy disk-ish
        p.drawRoundedRect(rect, 1.2, 1.2)
        # Top slot
        line(pad + 2, pad, pad + 2, pad + 3)
        line(s - pad - 2, pad, s - pad - 2, pad + 3)
        # Bottom label box
        p.drawRect(QRectF(pad + 1.5, s - pad - 4, s - 2 * pad - 3, 3))

    elif kind == 'play':
        # Right-pointing triangle, filled
        tri = QPolygonF([
            QPointF(pad + 1, pad),
            QPointF(s - pad, s / 2),
            QPointF(pad + 1, s - pad),
        ])
        p.setBrush(QColor(color))
        p.setPen(Qt.NoPen)
        p.drawPolygon(tri)

    elif kind == 'pause':
        p.setBrush(QColor(color))
        p.setPen(Qt.NoPen)
        bar_w = (s - 2 * pad - 2) / 2 - 0.5
        p.drawRoundedRect(QRectF(pad, pad, bar_w, s - 2 * pad), 0.6, 0.6)
        p.drawRoundedRect(QRectF(s - pad - bar_w, pad, bar_w, s - 2 * pad), 0.6, 0.6)

    elif kind == 'stop':
        p.setBrush(QColor(color))
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(rect, 1.0, 1.0)

    elif kind == 'step':
        # ▶ + bar
        tri = QPolygonF([
            QPointF(pad, pad),
            QPointF(s - pad - 3, s / 2),
            QPointF(pad, s - pad),
        ])
        p.setBrush(QColor(color))
        p.setPen(Qt.NoPen)
        p.drawPolygon(tri)
        p.drawRoundedRect(QRectF(s - pad - 1.6, pad, 1.6, s - 2 * pad), 0.5, 0.5)

    elif kind == 'plot':
        # Axes + curve
        line(pad, pad, pad, s - pad)
        line(pad, s - pad, s - pad, s - pad)
        path = QPainterPath()
        path.moveTo(pad + 1, s - pad - 1)
        path.quadTo(pad + (s - 2 * pad) * 0.4, pad - 0.5,
                    (s - pad), pad + 1)
        p.drawPath(path)

    elif kind == 'capture':
        # Camera
        p.drawRoundedRect(QRectF(pad, pad + 2, s - 2 * pad, s - 2 * pad - 2), 1.5, 1.5)
        # Top hump
        p.drawRoundedRect(QRectF(s / 2 - 1.8, pad, 3.6, 2.2), 0.6, 0.6)
        # Lens
        p.drawEllipse(QPointF(s / 2, s / 2 + 0.8), 2.2, 2.2)

    elif kind == 'route':
        # Routing zig-zag
        path = QPainterPath()
        path.moveTo(pad, pad + 2)
        path.lineTo(pad + 4, pad + 2)
        path.lineTo(pad + 4, s / 2)
        path.lineTo(s - pad - 4, s / 2)
        path.lineTo(s - pad - 4, s - pad - 2)
        path.lineTo(s - pad, s - pad - 2)
        p.drawPath(path)
        # Endpoint dots
        p.setBrush(QColor(color))
        p.drawEllipse(QPointF(pad, pad + 2), 1.0, 1.0)
        p.drawEllipse(QPointF(s - pad, s - pad - 2), 1.0, 1.0)

    elif kind == 'sun':
        p.drawEllipse(QPointF(s / 2, s / 2), 2.2, 2.2)
        for i in range(8):
            import math
            a = i * math.pi / 4
            r1, r2 = 3.4, 4.6
            line(s / 2 + math.cos(a) * r1, s / 2 + math.sin(a) * r1,
                 s / 2 + math.cos(a) * r2, s / 2 + math.sin(a) * r2)

    elif kind == 'moon':
        path = QPainterPath()
        path.moveTo(s - pad - 1, pad + 1)
        path.arcTo(QRectF(pad, pad, s - 2 * pad, s - 2 * pad), 60, 240)
        path.closeSubpath()
        p.setBrush(QColor(color))
        p.setPen(Qt.NoPen)
        p.drawPath(path)

    elif kind == 'search':
        p.drawEllipse(QPointF(s / 2 - 1, s / 2 - 1), s / 3 - 0.5, s / 3 - 0.5)
        line(s / 2 + 1.4, s / 2 + 1.4, s - pad, s - pad)

    elif kind == 'plus':
        line(s / 2, pad + 1, s / 2, s - pad - 1)
        line(pad + 1, s / 2, s - pad - 1, s / 2)

    elif kind == 'minus':
        line(pad + 1, s / 2, s - pad - 1, s / 2)

    p.end()
    return QIcon(px)


# -----------------------------------------------------------------------------
# Small reusable widgets
# -----------------------------------------------------------------------------

class _StatusPill(QFrame):
    """Colored-dot widget + text ('Ready', 'Simulating…', etc.).

    The dot is a separately-painted child so its color is independent of
    text color — idle goes grey, running goes green, paused amber, error red,
    regardless of QSS text-color rules.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("StatusPill")
        self.setProperty("state", "idle")
        self.setMinimumHeight(22)
        self.setAttribute(Qt.WA_StyledBackground, True)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 0, 10, 0)
        lay.setSpacing(6)

        self._dot = _StateDot(self)
        self._label = QLabel("Ready", self)
        self._label.setObjectName("StatusPillLabel")
        lay.addWidget(self._dot, 0, Qt.AlignVCenter)
        lay.addWidget(self._label, 0, Qt.AlignVCenter)

    def set_state(self, state: str, label: str | None = None):
        if state not in ('idle', 'running', 'paused', 'error'):
            state = 'idle'
        self.setProperty("state", state)
        self._dot.set_state(state)
        text = label or {
            'idle': 'Ready',
            'running': 'Simulating…',
            'paused': 'Paused',
            'error': 'Error',
        }[state]
        self._label.setText(text)
        # Force re-polish so the [state=…] selector reapplies on dark/light swap.
        self.style().unpolish(self)
        self.style().polish(self)


class _StateDot(QWidget):
    """6px filled circle painted in the state color (green / amber / red / grey)."""
    _COLORS = {
        'idle':    ('text_secondary', None),
        'running': ('success', 'success'),
        'paused':  ('warning', 'warning'),
        'error':   ('error', 'error'),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._state = 'idle'
        self.setFixedSize(QSize(8, 8))
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        theme_manager.theme_changed.connect(self.update)

    def set_state(self, s: str):
        self._state = s if s in self._COLORS else 'idle'
        self.update()

    def paintEvent(self, _ev):
        col_key, glow_key = self._COLORS[self._state]
        col = theme_manager.get_color(col_key)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        # Soft glow ring for active states
        if glow_key:
            glow = QColor(col); glow.setAlpha(70)
            p.setBrush(glow); p.setPen(Qt.NoPen)
            p.drawEllipse(0, 0, 8, 8)
            p.setBrush(col)
            p.drawEllipse(1, 1, 6, 6)
        else:
            p.setBrush(col); p.setPen(Qt.NoPen)
            p.drawEllipse(1, 1, 6, 6)
        p.end()


class _ZoomRocker(QWidget):
    """Compact   −   100%   +   cluster (replaces the wide slider)."""

    zoom_changed = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._zoom = 1.0
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self.minus_btn = QToolButton()
        self.minus_btn.setObjectName("ZoomRockerBtn")
        self.minus_btn.setIcon(_make_icon('minus', 14, theme_manager.get_color('text_secondary').name()))
        self.minus_btn.setIconSize(QSize(14, 14))
        self.minus_btn.setAutoRaise(True)
        self.minus_btn.setFixedSize(QSize(22, 22))
        self.minus_btn.clicked.connect(self._on_minus)

        self.label = QLabel("100%")
        self.label.setObjectName("ZoomRockerLabel")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setMinimumWidth(40)

        self.plus_btn = QToolButton()
        self.plus_btn.setObjectName("ZoomRockerBtn")
        self.plus_btn.setIcon(_make_icon('plus', 14, theme_manager.get_color('text_secondary').name()))
        self.plus_btn.setIconSize(QSize(14, 14))
        self.plus_btn.setAutoRaise(True)
        self.plus_btn.setFixedSize(QSize(22, 22))
        self.plus_btn.clicked.connect(self._on_plus)

        lay.addWidget(self.minus_btn)
        lay.addWidget(self.label)
        lay.addWidget(self.plus_btn)

    def _on_minus(self):
        self.set_zoom(max(0.25, round((self._zoom - 0.1) * 10) / 10))

    def _on_plus(self):
        self.set_zoom(min(2.0, round((self._zoom + 0.1) * 10) / 10))

    def set_zoom(self, factor: float):
        factor = max(0.25, min(2.0, factor))
        if abs(factor - self._zoom) < 1e-6:
            return
        self._zoom = factor
        self.label.setText(f"{int(round(factor * 100))}%")
        self.zoom_changed.emit(factor)

    def zoom(self) -> float:
        return self._zoom

    def refresh_icons(self):
        c = theme_manager.get_color('text_secondary').name()
        self.minus_btn.setIcon(_make_icon('minus', 14, c))
        self.plus_btn.setIcon(_make_icon('plus', 14, c))


class _TransportGroup(QWidget):
    """Play/Pause/Stop/Step + monospace t-readout, centered."""

    play = pyqtSignal()
    pause = pyqtSignal()
    stop = pyqtSignal()
    step = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)

        self.play_btn = self._mk_btn('play', 'TransportPlay', "Run (F5)")
        self.pause_btn = self._mk_btn('pause', 'TransportPause', "Pause (F6)")
        self.stop_btn = self._mk_btn('stop', 'TransportStop', "Stop (F7)")
        self.step_btn = self._mk_btn('step', 'TransportStep', "Step (F8)")

        self.play_btn.clicked.connect(self.play)
        self.pause_btn.clicked.connect(self.pause)
        self.stop_btn.clicked.connect(self.stop)
        self.step_btn.clicked.connect(self.step)

        self.time_label = QLabel("t = 0.000 / 10.000 s")
        self.time_label.setObjectName("TransportTimeLabel")
        mono = QFont("Menlo")
        mono.setStyleHint(QFont.Monospace)
        if hasattr(mono, 'setFamilies'):  # Qt 5.13+
            mono.setFamilies(["Menlo", "Consolas", "DejaVu Sans Mono", "monospace"])
        mono.setPointSize(9)
        self.time_label.setFont(mono)
        self.time_label.setMinimumWidth(140)
        self.time_label.setAlignment(Qt.AlignCenter)

        for w in (self.play_btn, self.pause_btn, self.stop_btn, self.step_btn):
            lay.addWidget(w)
        sep = QFrame()
        sep.setObjectName("TransportSep")
        sep.setFrameShape(QFrame.VLine)
        sep.setFixedWidth(1)
        lay.addSpacing(4)
        lay.addWidget(sep)
        lay.addSpacing(4)
        lay.addWidget(self.time_label)

    def _mk_btn(self, icon_kind: str, obj_name: str, tip: str) -> QToolButton:
        b = QToolButton()
        b.setObjectName(obj_name)
        b.setIcon(_make_icon(icon_kind, 16))
        b.setIconSize(QSize(16, 16))
        b.setFixedSize(QSize(28, 26))
        b.setToolTip(tip)
        b.setCursor(Qt.PointingHandCursor)
        return b

    def refresh_icons(self):
        self.play_btn.setIcon(_make_icon('play', 16))
        self.pause_btn.setIcon(_make_icon('pause', 16))
        self.stop_btn.setIcon(_make_icon('stop', 16))
        self.step_btn.setIcon(_make_icon('step', 16))

    def set_state(self, running: bool, paused: bool):
        self.play_btn.setEnabled(not running or paused)
        self.pause_btn.setEnabled(running and not paused)
        self.stop_btn.setEnabled(running)
        self.step_btn.setEnabled(not running or paused)

    def set_time(self, t: float, t_end: float):
        self.time_label.setText(f"t = {t:6.3f} / {t_end:.3f} s")


# -----------------------------------------------------------------------------
# Main toolbar
# -----------------------------------------------------------------------------

class ModernToolBar(QToolBar):
    """Modern styled toolbar — tightened V1 redesign."""

    # File / sim / view actions (kept identical to original)
    new_diagram = pyqtSignal()
    open_diagram = pyqtSignal()
    save_diagram = pyqtSignal()
    play_simulation = pyqtSignal()
    pause_simulation = pyqtSignal()
    stop_simulation = pyqtSignal()
    step_simulation = pyqtSignal()
    plot_results = pyqtSignal()
    capture_screen = pyqtSignal()
    auto_route_wires = pyqtSignal()
    zoom_changed = pyqtSignal(float)
    theme_toggled = pyqtSignal()
    # New, optional
    command_palette_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("Main Toolbar", parent)
        self.setObjectName("ModernToolBar")
        self.setMovable(False)
        self.setFloatable(False)
        self.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.setIconSize(QSize(18, 18))

        self._build_actions()
        self._build_layout()

        theme_manager.theme_changed.connect(self._update_theme)
        self._update_theme()

    # -- Actions ------------------------------------------------------------

    def _build_actions(self):
        def mk(kind: str, label: str, shortcut: str | None, tip: str, sig):
            a = QAction(_make_icon(kind, 18), label, self)
            if shortcut:
                a.setShortcut(shortcut)
            a.setToolTip(f"{tip}" + (f"  ({shortcut})" if shortcut else ""))
            a.triggered.connect(sig)
            return a

        self.new_action = mk('new', "New", "Ctrl+N", "New diagram", self.new_diagram.emit)
        self.open_action = mk('open', "Open", "Ctrl+O", "Open diagram", self.open_diagram.emit)
        self.save_action = mk('save', "Save", "Ctrl+S", "Save diagram", self.save_diagram.emit)
        self.plot_action = mk('plot', "Plot", None, "Show waveform inspector", self.plot_results.emit)
        self.capture_action = mk('capture', "Capture", None, "Take screenshot", self.capture_screen.emit)
        self.auto_route_action = mk('route', "Auto-route", None, "Auto-route wires", self.auto_route_wires.emit)
        self.theme_action = mk('sun', "Theme", None, "Toggle theme", self._toggle_theme)

    # -- Layout -------------------------------------------------------------

    def _build_layout(self):
        # Left group: file actions
        self.addAction(self.new_action)
        self.addAction(self.open_action)
        self.addAction(self.save_action)
        self.addSeparator()

        # Centered cluster needs to push left + right groups apart. The
        # cleanest way in a QToolBar is two stretch spacers around a single
        # widget that hosts the transport group.
        left_stretch = QWidget()
        left_stretch.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.addWidget(left_stretch)

        self.transport = _TransportGroup(self)
        self.transport.play.connect(self.play_simulation)
        self.transport.pause.connect(self.pause_simulation)
        self.transport.stop.connect(self.stop_simulation)
        self.transport.step.connect(self.step_simulation)
        self.addWidget(self.transport)

        right_stretch = QWidget()
        right_stretch.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.addWidget(right_stretch)

        # Right group: status pill, view actions, zoom, ⌘K, theme
        self.status_pill = _StatusPill(self)
        self.addWidget(self.status_pill)

        self.addSeparator()
        self.addAction(self.plot_action)
        self.addAction(self.capture_action)
        self.addAction(self.auto_route_action)
        self.addSeparator()

        self.zoom_rocker = _ZoomRocker(self)
        self.zoom_rocker.zoom_changed.connect(self.zoom_changed)
        self.addWidget(self.zoom_rocker)
        self.addSeparator()

        self.cmdk_btn = QPushButton("Search…  ⌘K")
        self.cmdk_btn.setObjectName("CommandPaletteBtn")
        self.cmdk_btn.setFlat(True)
        self.cmdk_btn.setCursor(Qt.PointingHandCursor)
        self.cmdk_btn.clicked.connect(self.command_palette_requested)
        self.addWidget(self.cmdk_btn)

        self.addAction(self.theme_action)

    # -- Theme --------------------------------------------------------------

    def _toggle_theme(self):
        theme_manager.toggle_theme()
        self.theme_toggled.emit()

    def _update_theme(self):
        # Re-render all icons with new theme colors
        self.new_action.setIcon(_make_icon('new', 18))
        self.open_action.setIcon(_make_icon('open', 18))
        self.save_action.setIcon(_make_icon('save', 18))
        self.plot_action.setIcon(_make_icon('plot', 18))
        self.capture_action.setIcon(_make_icon('capture', 18))
        self.auto_route_action.setIcon(_make_icon('route', 18))

        if theme_manager.current_theme.value == "dark":
            self.theme_action.setIcon(_make_icon('sun', 18))
            self.theme_action.setToolTip("Switch to light theme")
        else:
            self.theme_action.setIcon(_make_icon('moon', 18))
            self.theme_action.setToolTip("Switch to dark theme")

        self.transport.refresh_icons()
        self.zoom_rocker.refresh_icons()

    # -- Public API (preserved) --------------------------------------------

    def set_status(self, message: str):
        """Compatibility shim — also drives the status pill color."""
        m = (message or "").lower()
        if 'run' in m or 'simulat' in m and 'pause' not in m:
            self.status_pill.set_state('running')
        elif 'paus' in m:
            self.status_pill.set_state('paused')
        elif 'error' in m or 'fail' in m:
            self.status_pill.set_state('error')
        else:
            self.status_pill.set_state('idle', message if message else None)

    def set_simulation_state(self, running: bool, paused: bool = False):
        self.transport.set_state(running, paused)
        if running and not paused:
            self.status_pill.set_state('running')
        elif running and paused:
            self.status_pill.set_state('paused')
        else:
            self.status_pill.set_state('idle')

    def set_simulation_time(self, t: float, t_end: float = 10.0):
        """NEW — hook this up from your sim tick if you want live t-readout.
        Safe to ignore; defaults to 0/10 until called."""
        self.transport.set_time(t, t_end)

    def get_zoom_factor(self) -> float:
        return self.zoom_rocker.zoom()

    def set_zoom_factor(self, factor: float):
        self.zoom_rocker.set_zoom(factor)

    # Back-compat: old code referenced `zoom_slider` and `zoom_label`.
    # Forward those attribute accesses to the rocker so nothing breaks.
    @property
    def zoom_slider(self):
        # Returns the rocker for any code that calls .value() etc.
        return self.zoom_rocker

    @property
    def zoom_label(self):
        return self.zoom_rocker.label
