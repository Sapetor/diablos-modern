"""
Connection Renderer Module
Responsible for drawing connection lines, arrowheads, and labels.
Separates rendering logic from the DLine data model.
"""

import math
import logging
import numpy as np
from PyQt6.QtGui import QPainter, QPen, QColor, QPolygonF

from PyQt6.QtCore import Qt, QPoint, QPointF, QRect
from modern_ui.renderers.sample_time_colors import rate_color
from modern_ui.themes.theme_manager import (
    theme_manager,
    get_ui_font,
    get_mono_font,
    font_metrics,
    text_width,
    TYPE,
)

logger = logging.getLogger(__name__)

# Wires are drawn with round caps and joins everywhere; unpack into QPen(...).
_ROUND_ENDS = (Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)

# The chip / label fonts are constants; building them (and their metrics) per
# frame showed up in the paint profile, so they are created lazily once.
_CHIP_FONT = None
_LABEL_FONT = None


def _chip_font():
    """Monospace font for the live port-value chips (built once)."""
    global _CHIP_FONT
    if _CHIP_FONT is None:
        _CHIP_FONT = get_mono_font(TYPE["caption"])
    return _CHIP_FONT


def _label_font():
    """UI font for connection labels (built once)."""
    global _LABEL_FONT
    if _LABEL_FONT is None:
        _LABEL_FONT = get_ui_font(TYPE["body"])
    return _LABEL_FONT


class ConnectionRenderer:
    """Stateless renderer for connection lines."""

    # Optional ``callable(base_alpha) -> int`` set by the owning canvas to its
    # ``glow_pulse_alpha``. When present, the active/selected connection's outer
    # glow breathes with the canvas animation phase; otherwise it uses the flat
    # base alpha so the renderer stays usable standalone. Class default keeps
    # every instance safe to read without an __init__.
    pulse_alpha = None

    # Base alpha of the selected-connection outer glow (the value modulated by
    # the canvas phase while a simulation/connection drag keeps the timer live).
    _ACTIVE_GLOW_BASE_ALPHA = 40

    # Extra width (px) of the canvas-coloured halo painted under every wire.
    # Where a later-drawn wire crosses an earlier one it cuts a small gap into
    # it, so crossings read as "over/under" rather than as junctions. Set to 0
    # to disable.
    CROSSING_GAP = 3.0

    # Distance (px) the arrowhead tip is pulled back from the port centre so
    # it stays visible in front of the port disc (ports paint on top of wires).
    ARROW_PORT_INSET = 7

    # Arrowhead length (px) along the wire and half-angle of its sides.
    ARROW_SIZE = 10
    ARROW_HALF_ANGLE = math.pi / 7

    def _glow_alpha(self, base_alpha: int) -> int:
        """Resolve a glow alpha, applying the canvas pulse callable if set."""
        fn = self.pulse_alpha
        return fn(base_alpha) if callable(fn) else base_alpha

    # ── Live-state V1: port-value chips ────────────────────────────

    def draw_port_value_chips(self, painter: QPainter, blocks) -> None:
        """Draw a small chip next to each output port showing its current scalar value.

        Called from the canvas paintEvent only when simulation is running.
        Tries `block.get_held_output(port)`, falls back to `block._held_outputs[port]`.
        Vector values render as `vec[N]`.
        """
        if not painter or not painter.isActive():
            return
        try:
            chip_bg = theme_manager.get_color("surface_elevated")
            success = theme_manager.get_color("success")
            border = QColor(success)
            border.setAlpha(90)

            font = _chip_font()
            painter.setFont(font)
            fm = font_metrics(font)

            for block in blocks:
                coords = getattr(block, "out_coords", None)
                if not coords:
                    continue
                for port_idx, p in enumerate(coords):
                    val = self._port_value(block, port_idx)
                    if val is None:
                        continue
                    text = self._format_value(val)
                    if not text:
                        continue

                    text_w = text_width(fm, text)
                    text_h = fm.height()
                    pad_x, pad_y = 6, 2
                    chip_w = text_w + 2 * pad_x
                    chip_h = text_h + 2 * pad_y

                    # Position to the right of the port; flip to left if it would
                    # exceed the visible area (best-effort — caller can clip).
                    x = p.x() + 8
                    y = p.y() - chip_h // 2
                    rect = QRect(x, y, chip_w, chip_h)

                    painter.setBrush(chip_bg)
                    painter.setPen(QPen(border, 1))
                    painter.drawRoundedRect(rect, 3, 3)

                    painter.setPen(success)
                    painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)
        except Exception as e:  # never let a render error tank the paint
            logger.debug("Live port-value chip draw skipped: %s", e)

    def _port_value(self, block, port_idx: int):
        # Try the canonical API first
        getter = getattr(block, "get_held_output", None)
        if callable(getter):
            try:
                return getter(port_idx)
            except Exception:
                logger.debug(
                    "get_held_output raised while reading port value; falling back", exc_info=True
                )
        held = getattr(block, "_held_outputs", None)
        if isinstance(held, dict) and port_idx in held:
            return held[port_idx]
        return None

    def _format_value(self, val) -> str:
        # Scalars → 3-significant-figure float
        try:
            if isinstance(val, (int, float, np.floating, np.integer)):
                f = float(val)
                if abs(f) < 1e-3 or abs(f) >= 1e4:
                    return f"{f:.2e}"
                return f"{f:.3f}"
            if isinstance(val, np.ndarray):
                if val.ndim == 0 or val.size == 1:
                    return self._format_value(float(val.item()))
                return f"vec[{val.size}]"
            if isinstance(val, (list, tuple)):
                if len(val) == 1:
                    return self._format_value(val[0])
                return f"vec[{len(val)}]"
        except Exception:
            logger.debug("Failed to format port value for live chip", exc_info=True)
        return ""

    # Radius (px) of the dot drawn where wires from one output port split.
    JUNCTION_RADIUS = 3.5

    @staticmethod
    def _line_width(line) -> float:
        """Stroke width: thicker for vector (MIMO) signals and when selected."""
        base_width = 2.0 if getattr(line, "signal_width", 1) <= 1 else 3.5
        selected = line.selected and line.selected_segment == -1
        return base_width + 0.5 if selected else base_width

    def draw_halo(self, line, painter: QPainter):
        """Paint the canvas-coloured crossing-gap halo under ``line``.

        Separate from :meth:`draw_line` so the canvas can paint the halos of
        all wires leaving one port before any of their strokes: siblings then
        never cut gaps into each other where they run together and split.
        """
        if self.CROSSING_GAP <= 0 or not painter or not painter.isActive():
            return
        if not line.path or line.path.isEmpty():
            return
        painter.save()
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setPen(
                QPen(
                    theme_manager.get_color("canvas_background"),
                    self._line_width(line) + self.CROSSING_GAP,
                    Qt.PenStyle.SolidLine,
                    Qt.PenCapStyle.FlatCap,
                    Qt.PenJoinStyle.RoundJoin,
                )
            )
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(line.path)
        finally:
            painter.restore()

    @staticmethod
    def _polyline(line):
        """The wire as straight segments, or None when it is drawn as a curve.

        Orthogonal and waypoint wires are polylines (their fillets are tiny);
        a two-point bezier wire is a cubic unless its ends are level.
        """
        pts = list(getattr(line, "points", None) or [])
        if len(pts) < 2:
            return None
        if getattr(line, "routing_mode", "bezier") == "orthogonal" or len(pts) > 2:
            return pts
        a, b = pts
        return pts if a.x() == b.x() or a.y() == b.y() else None

    @staticmethod
    def _on_polyline(p, pts, tol=1.0) -> bool:
        px, py = p.x(), p.y()
        for a, b in zip(pts, pts[1:]):
            ax, ay, bx, by = a.x(), a.y(), b.x(), b.y()
            dx, dy = bx - ax, by - ay
            seg2 = dx * dx + dy * dy
            t = 0.0 if seg2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg2))
            if math.hypot(px - (ax + t * dx), py - (ay + t * dy)) <= tol:
                return True
        return False

    @classmethod
    def junction_points(cls, lines):
        """Points where wires leaving the same output port part ways.

        For each pair of sibling polylines, the junction is the last vertex of
        one that still lies on the other. Wires that diverge right at the port
        (e.g. a fan of bezier curves) get no dot.
        """
        groups = {}
        for line in lines:
            if getattr(line, "hidden", False):
                continue
            pts = cls._polyline(line)
            if pts is not None:
                groups.setdefault((line.srcblock, line.srcport), []).append(pts)

        def split(a, b):
            """Last vertex of ``b`` still on ``a``, with its distance along
            ``b`` (None when ``b`` leaves ``a`` at the port)."""
            k, dist = 0, 0.0
            while k + 1 < len(b) and cls._on_polyline(b[k + 1], a):
                p, q = b[k], b[k + 1]
                dist += math.hypot(q.x() - p.x(), q.y() - p.y())
                k += 1
            return (dist, b[k]) if 0 < k < len(b) - 1 else None

        found = {}
        for siblings in groups.values():
            for i, a in enumerate(siblings):
                for b in siblings[i + 1 :]:
                    # Both directions: whichever wire turns away first marks
                    # only the start of the shared run; the split is the
                    # point furthest along it.
                    cands = [c for c in (split(a, b), split(b, a)) if c is not None]
                    if cands:
                        q = max(cands, key=lambda c: c[0])[1]
                        found[(q.x(), q.y())] = QPointF(q)
        return list(found.values())

    def draw_junctions(self, painter: QPainter, points) -> None:
        """Dot each branch point (Simulink-style) in the wire colour."""
        if not points or not painter or not painter.isActive():
            return
        painter.save()
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(theme_manager.get_color("connection_default"))
            r = self.JUNCTION_RADIUS
            for p in points:
                painter.drawEllipse(p, r, r)
        finally:
            painter.restore()

    def draw_line(self, line, painter: QPainter, sample_time: float = -1.0, halo: bool = True):
        """
        Draw the connection line with smooth Bezier curves and modern styling.

        Args:
            line: DLine object containing path and properties
            painter: QPainter instance
            sample_time: rate the wire carries (see sample_time_colors);
                a discrete wire (> 0) is tinted with the block dot's color.
            halo: paint the crossing-gap halo first (False when the caller
                already painted it with :meth:`draw_halo`).
        """
        if not painter or not painter.isActive():
            return

        if not line.path or line.path.isEmpty():
            return

        # Save painter state
        painter.save()

        try:
            # Enable antialiasing for smooth curves
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

            # Use theme_manager for connection colors
            default_connection_color = theme_manager.get_color("connection_default")
            active_connection_color = theme_manager.get_color("connection_active")

            # Determine line color and width based on selection state and signal width
            # Note: selected_segment logic handled below for specific highlighting
            is_selected = line.selected and line.selected_segment == -1
            rate_tint = rate_color(sample_time) if sample_time > 0 else None
            idle_color = rate_tint or default_connection_color
            pen_color = active_connection_color if is_selected else idle_color

            line_width = self._line_width(line)

            # Crossing gap: a halo in the canvas colour under the wire so it
            # visibly passes over wires drawn before it.
            if halo:
                self.draw_halo(line, painter)

            # Draw subtle glow/shadow for selected connections
            if is_selected:
                # Draw outer glow — alpha gently pulses with the canvas phase
                # while the animation timer runs (stable alpha when idle).
                glow_color = QColor(active_connection_color)
                glow_color.setAlpha(self._glow_alpha(self._ACTIVE_GLOW_BASE_ALPHA))
                glow_pen = QPen(
                    glow_color,
                    line_width + 4,
                    Qt.PenStyle.SolidLine,
                    *_ROUND_ENDS,
                )
                painter.setPen(glow_pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawPath(line.path)

            # Determine line style based on signal type
            # Discrete signals use dashed lines
            is_discrete = getattr(line, "discrete_signal", False) or sample_time > 0
            line_style = Qt.PenStyle.DashLine if is_discrete else Qt.PenStyle.SolidLine

            # Draw main connection line
            pen = QPen(
                pen_color,
                line_width,
                line_style,
                *_ROUND_ENDS,
            )
            if is_discrete:
                # Set dash pattern for discrete signals: dash-space-dash-space
                pen.setDashPattern([6, 3])  # 6 pixels dash, 3 pixels space
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(line.path)

            # If a specific segment is selected, highlight it
            if line.selected and line.selected_segment != -1:
                if line.selected_segment < len(line.points) - 1:
                    p1 = line.points[line.selected_segment]
                    p2 = line.points[line.selected_segment + 1]

                    # Draw glow for segment
                    segment_glow_color = QColor(active_connection_color)
                    segment_glow_color.setAlpha(60)
                    glow_pen = QPen(
                        segment_glow_color,
                        6,
                        Qt.PenStyle.SolidLine,
                        *_ROUND_ENDS,
                    )
                    painter.setPen(glow_pen)
                    painter.drawLine(p1, p2)

                    # Draw segment highlight
                    highlight_pen = QPen(
                        active_connection_color,
                        3,
                        Qt.PenStyle.SolidLine,
                        *_ROUND_ENDS,
                    )
                    painter.setPen(highlight_pen)
                    painter.drawLine(p1, p2)

            # Draw waypoint handles when the line is selected (skip the two
            # port endpoints, which are not draggable).
            if line.selected:
                handle_fill = QColor(theme_manager.get_color("canvas_background"))
                painter.setBrush(handle_fill)
                painter.setPen(QPen(active_connection_color, 2))
                for point in line.points[1:-1]:
                    painter.drawEllipse(point, 4, 4)

            # Draw connection label if present
            if getattr(line, "label", ""):
                self._draw_label(line, painter)

            # Draw arrowhead
            self._draw_arrowhead(
                line,
                painter,
                active_connection_color if line.selected else idle_color,
            )

        finally:
            # Restore painter state
            painter.restore()

    @staticmethod
    def label_anchor(line) -> QPoint:
        """Scene point where the wire label is centred: halfway along the path."""
        path = getattr(line, "path", None)
        if path is not None and not path.isEmpty() and path.length() > 0:
            mid = path.pointAtPercent(0.5)
            return QPoint(int(mid.x()), int(mid.y()))
        pts = getattr(line, "points", None) or []
        if len(pts) >= 2:
            a, b = pts[0], pts[-1]
            return QPoint((a.x() + b.x()) // 2, (a.y() + b.y()) // 2)
        return QPoint(0, 0)

    def _draw_label(self, line, painter: QPainter):
        """Draw the connection label at the midpoint of the drawn path."""
        if len(line.points) < 2:
            return

        label_pos = self.label_anchor(line)

        # Draw label background (canonical UI stack instead of fixed Arial)
        font = _label_font()
        painter.setFont(font)

        metrics = font_metrics(font)
        text = str(line.label)
        label_width = text_width(metrics, text)
        text_height = metrics.height()

        # Background rectangle
        padding = 4
        bg_rect = QRect(
            label_pos.x() - label_width // 2 - padding,
            label_pos.y() - text_height // 2 - padding,
            label_width + 2 * padding,
            text_height + 2 * padding,
        )

        # Draw semi-transparent background
        bg_color = theme_manager.get_color("canvas_background")
        bg_color.setAlpha(230)
        painter.setBrush(bg_color)
        painter.setPen(QPen(theme_manager.get_color("border_primary"), 1))
        painter.drawRoundedRect(bg_rect, 3, 3)

        # Draw text
        text_color = theme_manager.get_color("text_primary")
        painter.setPen(text_color)
        painter.drawText(bg_rect, Qt.AlignmentFlag.AlignCenter, text)

    def _draw_arrowhead(self, line, painter: QPainter, color: QColor):
        """Draw arrowhead at the end of the line."""
        if not line.path or line.path.isEmpty():
            return

        arrow_size = self.ARROW_SIZE
        length = line.path.length()
        if length <= 0:
            return

        end_point = line.path.pointAtPercent(1.0)
        # Direction of the final approach: the path's tangent at the port.
        # (A chord over the last arrow_size px tilted the arrow whenever a
        # corner fillet ended that close to the port.) Qt measures the angle
        # in degrees, counter-clockwise with y up.
        angle = -math.radians(line.path.angleAtPercent(1.0))

        # Pull the tip back so it is not hidden under the port disc, but never
        # further than the wire is long.
        inset = min(self.ARROW_PORT_INSET, max(0.0, length - arrow_size))
        tip = QPointF(
            end_point.x() - inset * math.cos(angle), end_point.y() - inset * math.sin(angle)
        )

        # Arrowhead points
        half = self.ARROW_HALF_ANGLE
        arrow_p1 = QPointF(
            tip.x() - arrow_size * math.cos(angle - half),
            tip.y() - arrow_size * math.sin(angle - half),
        )
        arrow_p2 = QPointF(
            tip.x() - arrow_size * math.cos(angle + half),
            tip.y() - arrow_size * math.sin(angle + half),
        )

        arrow_polygon = QPolygonF([tip, arrow_p1, arrow_p2])

        painter.setBrush(color)  # Fill arrowhead with line color
        painter.setPen(Qt.PenStyle.NoPen)  # No border for arrowhead
        painter.drawPolygon(arrow_polygon)
