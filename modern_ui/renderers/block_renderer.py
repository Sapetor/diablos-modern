"""
Block Renderer Module
Handles the rendering of DBlock objects on the canvas.
Separates the view/drawing logic from the data model.
"""

import logging
import math
from typing import Optional
from PyQt5.QtGui import (
    QColor, QPen, QPainter, QPolygonF, QLinearGradient, QPainterPath,
    QRadialGradient, QTransform, QFont, QFontMetrics,
)
from PyQt5.QtCore import Qt, QRect, QPoint, QPointF
from modern_ui.themes.theme_manager import theme_manager, get_ui_font, TYPE

logger = logging.getLogger(__name__)

_CATEGORY_KEY_MAP = (
    ('source',    'block_source'),
    ('math',      'block_process'),
    ('control',   'block_control'),
    ('sink',      'block_sink'),
    ('routing',   'block_routing'),
    ('analysis',  'block_analysis'),
    ('pde',       'block_pde'),
    ('optim',     'block_optimization'),
)


def _category_token(block) -> str:
    """Return a lowercased category string from a block, or '' if absent."""
    cat = getattr(block, 'category', None)
    if cat is None:
        return ''
    return cat.lower() if isinstance(cat, str) else str(cat).lower()


# Memoizes the (category-token, suffix) -> color-key substring scan. The
# mapping is purely theme-independent (it resolves a category string to a
# theme_manager color *key*, not a QColor), so it never needs invalidation on
# theme/palette change and can be cached for the process lifetime. This keeps
# the linear scan over _CATEGORY_KEY_MAP off the per-frame paint path; the
# category set is tiny and bounded, so the cache cannot grow unbounded.
_CATEGORY_KEY_CACHE: dict = {}


def _category_color_key(block, suffix: str = '') -> str:
    """Resolve the theme-manager color key for a block's category.

    suffix='' returns the fill key (e.g. 'block_source'); '_border' returns
    the border key. Falls back to 'block_other...' for unknown categories.
    """
    cat = _category_token(block)
    cache_key = (cat, suffix)
    cached = _CATEGORY_KEY_CACHE.get(cache_key)
    if cached is not None:
        return cached
    for token, key in _CATEGORY_KEY_MAP:
        if token in cat:
            result = key + suffix
            break
    else:
        result = 'block_other' + suffix
    _CATEGORY_KEY_CACHE[cache_key] = result
    return result


# Soft-elevation shadow recipe. Each tuple is (offset, expand, alpha_scale) for
# one stacked layer drawn behind the block, from the broadest/faintest layer
# down to the tightest/strongest. Progressively larger offset+expand with
# decreasing alpha fakes a blurred drop shadow using only plain QPainter fills
# (no per-pixel blur), reading as modern soft elevation rather than a hard 1990s
# offset rectangle. alpha_scale is a fraction of the theme base alpha so
# light/dark themes keep their intended shadow strength.
_SOFT_SHADOW_LAYERS = (
    # (offset, expand, alpha_scale)
    (5, 4, 0.25),
    (4, 2, 0.45),
    (3, 0, 0.70),
    (2, -1, 1.00),
)


class BlockRenderer:
    """
    Stateless renderer for DBlock objects.
    """

    # Soft-elevation shadow recipe, exposed on the class so it is easy to
    # assert on in tests (see _draw_soft_shadow for the per-layer semantics).
    _SOFT_SHADOW_LAYERS = _SOFT_SHADOW_LAYERS

    def __init__(self):
        # Tracks block_fns whose draw_icon has already raised, so the failure
        # is logged once at warning level rather than every paint frame.
        self._draw_icon_warned = set()

    def draw_block(self, block, painter: Optional[QPainter], draw_name: bool = True,
                   draw_ports: bool = True, hovered_port=None) -> None:
        """
        Draw a block on the canvas with modern styling, shadows, and depth.

        Args:
            block: The DBlock instance to draw
            painter: QPainter instance for rendering
            draw_name: Whether to draw the block name/label
            draw_ports: Whether to draw the input/output port connectors
            hovered_port: Optional (port_idx, is_output) tuple identifying a
                port on *this* block that is currently hovered, so it can be
                drawn larger/brighter. None when nothing on this block is hovered.
        """
        if painter is None:
            return

        # Wrap the whole body so font/pen/brush/transform mutations (including
        # those applied by the icon helpers) cannot leak to the next block or
        # back to the caller, even if an exception is raised mid-paint.
        painter.save()
        try:
            self._draw_block_body(block, painter, draw_name, draw_ports, hovered_port)
        finally:
            painter.restore()

    def _draw_block_body(self, block, painter: QPainter, draw_name: bool,
                         draw_ports: bool, hovered_port=None) -> None:
        """Body of draw_block; runs inside a painter.save()/restore() guard."""
        # Draw a soft multi-layer shadow for depth. The shadow follows the block
        # outline (triangle for Gain, rounded rect otherwise) and is built from
        # several stacked fills with growing offset/spread and fading alpha.
        self._draw_soft_shadow(block, painter)

        # Resolve colors from the *current* theme every paint. The block's
        # cached b_color is set at startup; when the user toggles dark/light
        # later, that cache goes stale and produces light fills on a dark
        # canvas (or vice versa). Going through theme_manager keeps fill
        # and border in sync with the active theme.
        if hasattr(block, 'category') and block.category:
            border_color = theme_manager.get_color(_category_color_key(block, '_border'))
            base_color = theme_manager.get_color(_category_color_key(block, ''))
        else:
            border_color = theme_manager.get_color('border_primary')
            base_color = block.b_color

        if block.selected:
            border_color = theme_manager.get_color('block_selected')

        # Draw main block shape with gradient fill.
        # Subtle lighten (+12 RGB) keeps the top close to base so icons over
        # the top of the block don't wash out against a too-bright gradient.
        gradient = QLinearGradient(block.left, block.top, block.left, block.top + block.height)
        lighter_color = QColor(base_color)
        lighter_color.setRed(min(255, base_color.red() + 12))
        lighter_color.setGreen(min(255, base_color.green() + 12))
        lighter_color.setBlue(min(255, base_color.blue() + 12))
        gradient.setColorAt(0, lighter_color)
        gradient.setColorAt(1, base_color)
        
        painter.setBrush(gradient)
        painter.setPen(QPen(border_color, 3 if block.selected else 2.5))

        if block.block_fn == "Gain":
            # Draw a triangle for the Gain block
            points = QPolygonF()
            if not block.flipped:
                points.append(QPoint(block.left, block.top))
                points.append(QPoint(block.left + block.width, int(block.top + block.height / 2)))
                points.append(QPoint(block.left, block.top + block.height))
            else:
                points.append(QPoint(block.left + block.width, block.top))
                points.append(QPoint(block.left, int(block.top + block.height / 2)))
                points.append(QPoint(block.left + block.width, block.top + block.height))
            painter.drawPolygon(points)
        else:
            radius = 12
            painter.drawRoundedRect(QRect(block.left, block.top, block.width, block.height), radius, radius)

        # Draw block-specific icon
        icon_pen = QPen(theme_manager.get_color('block_icon_color'), 2)
        painter.setPen(icon_pen)
        
        path = QPainterPath()
        
        # Try polymorphic draw_icon first
        if block.block_instance and hasattr(block.block_instance, 'draw_icon'):
            try:
                custom_path = block.block_instance.draw_icon(block.rect)
                if custom_path is not None:
                    path = custom_path
            except Exception as e:
                # Log once per block type at warning; subsequent identical
                # failures drop to debug to avoid per-frame log spam. The
                # legacy-icon fallback below recovers cleanly (it guards on
                # path.isEmpty()), so this is non-fatal.
                if block.block_fn not in self._draw_icon_warned:
                    self._draw_icon_warned.add(block.block_fn)
                    logger.warning(f"draw_icon failed for {block.block_fn}: {e}")
                else:
                    logger.debug(f"draw_icon failed for {block.block_fn}: {e}")
        
        # Fallback to legacy switch statement
        self._draw_legacy_icon(block, painter, path)

        if not path.isEmpty():
            margin = block.width * 0.2
            transform = QTransform()
            if block.flipped:
                transform.translate(block.left + block.width - margin, block.top + margin)
                transform.scale(-(block.width - 2 * margin), block.height - 2 * margin)
            else:
                transform.translate(block.left + margin, block.top + margin)
                transform.scale(block.width - 2 * margin, block.height - 2 * margin)
            
            scaled_path = transform.map(path)
            painter.drawPath(scaled_path)

        # Draw ports
        if draw_ports:
            self.draw_ports(block, painter, hovered_port)
            self.draw_port_labels(block, painter)

        if draw_name:
            text_color = theme_manager.get_color('text_primary')
            painter.setPen(text_color)
            # Copy the model's QFont before mutating; this renderer is stateless
            # and must not alter shared model state (block.font is persistent).
            font = QFont(block.font)
            font.setWeight(QFont.Normal)
            painter.setFont(font)
            text_rect = QRect(block.left, block.top + block.height + 2, block.width, 28)
            painter.drawText(text_rect, Qt.AlignHCenter | Qt.AlignTop, block.username)

        # Enhanced selection visualization
        if block.selected:
            selection_color = theme_manager.get_color('block_selected')
            painter.setPen(QPen(selection_color, 3, Qt.SolidLine))
            painter.setBrush(Qt.NoBrush)
            padding = 4
            selection_rect = QRect(
                block.left - padding,
                block.top - padding,
                block.width + 2 * padding,
                block.height + 2 * padding
            )
            painter.drawRoundedRect(selection_rect, 14, 14)

        # Draw sample rate indicator for discrete blocks
        self._draw_sample_rate_indicator(block, painter)

    def _draw_soft_shadow(self, block, painter: QPainter) -> None:
        """Draw a soft, multi-layer drop shadow behind the block.

        Stacks the _SOFT_SHADOW_LAYERS recipe (growing offset/spread, fading
        alpha) into a few plain fills so the result reads as soft elevation
        without any per-pixel blur. The shadow tracks the block outline: a
        triangle for the Gain block, a rounded rectangle otherwise. The base
        color/alpha come from the active theme's 'block_shadow' token so
        light/dark themes stay consistent.
        """
        base_shadow = theme_manager.get_color('block_shadow')
        base_alpha = base_shadow.alpha()

        painter.setPen(Qt.NoPen)
        radius = 12
        is_gain = block.block_fn == "Gain"

        for offset, expand, alpha_scale in self._SOFT_SHADOW_LAYERS:
            layer_color = QColor(base_shadow)
            layer_color.setAlpha(int(base_alpha * alpha_scale))
            painter.setBrush(layer_color)

            if is_gain:
                # Shadow for the Gain triangle, grown outward by `expand`.
                shadow_points = QPolygonF()
                if not block.flipped:
                    shadow_points.append(QPoint(block.left + offset - expand, block.top + offset - expand))
                    shadow_points.append(QPoint(block.left + block.width + offset + expand, int(block.top + block.height / 2) + offset))
                    shadow_points.append(QPoint(block.left + offset - expand, block.top + block.height + offset + expand))
                else:
                    shadow_points.append(QPoint(block.left + block.width + offset + expand, block.top + offset - expand))
                    shadow_points.append(QPoint(block.left + offset - expand, int(block.top + block.height / 2) + offset))
                    shadow_points.append(QPoint(block.left + block.width + offset + expand, block.top + block.height + offset + expand))
                painter.drawPolygon(shadow_points)
            else:
                # Shadow for the rounded rectangle, grown outward by `expand`.
                shadow_rect = QRect(
                    block.left + offset - expand,
                    block.top + offset - expand,
                    block.width + 2 * expand,
                    block.height + 2 * expand,
                )
                painter.drawRoundedRect(shadow_rect, radius + expand, radius + expand)

    def draw_ports(self, block, painter: Optional[QPainter], hovered_port=None) -> None:
        """Draw input and output ports with modern styling.

        Args:
            block: The DBlock instance whose ports are drawn.
            painter: QPainter instance for rendering.
            hovered_port: Optional (port_idx, is_output) tuple identifying a
                port on *this* block to emphasize (drawn larger/brighter with
                the 'port_hover' color). None when no port on this block is
                hovered.
        """
        if painter is None:
            return

        port_input_color = theme_manager.get_color('port_input')
        port_output_color = theme_manager.get_color('port_output')
        port_draw_radius = block.port_radius - 1

        # Resolve which (index, is_output) on this block is hovered, if any.
        hover_idx, hover_is_output = self._resolve_hovered_port(hovered_port)

        # Input ports
        for i, port_in_location in enumerate(block.in_coords):
            is_hovered = hover_idx == i and hover_is_output is False
            self._draw_port(painter, port_in_location, port_input_color, port_draw_radius, is_hovered)

        # Output ports
        for i, port_out_location in enumerate(block.out_coords):
            is_hovered = hover_idx == i and hover_is_output is True
            self._draw_port(painter, port_out_location, port_output_color, port_draw_radius, is_hovered)

    @staticmethod
    def _resolve_hovered_port(hovered_port):
        """Normalize a hovered_port spec to (port_idx, is_output).

        Accepts the (port_idx, is_output) tuple passed down per block (or None)
        and returns (None, None) when nothing on this block is hovered, so the
        port loops can compare cheaply without re-validating the shape.
        """
        if not hovered_port:
            return None, None
        try:
            port_idx, is_output = hovered_port
        except (TypeError, ValueError):
            return None, None
        return port_idx, bool(is_output)

    def _draw_port(self, painter: QPainter, location, port_color: QColor,
                   port_draw_radius: int, is_hovered: bool) -> None:
        """Draw a single port disc, emphasized when hovered.

        A hovered port is drawn slightly larger and brighter using the
        'port_hover' color so the user can see which connector they are about
        to grab; unhovered ports keep the standard input/output gradient.
        """
        # A hovered port grows by ~40% and recolors to the brighter hover token.
        if is_hovered:
            base_color = theme_manager.get_color('port_hover')
            radius = int(port_draw_radius * 1.4)
        else:
            base_color = port_color
            radius = port_draw_radius

        gradient = QRadialGradient(location.x(), location.y(), radius)
        gradient.setColorAt(0.0, base_color.lighter(130))
        gradient.setColorAt(0.7, base_color)
        gradient.setColorAt(1.0, base_color.darker(110))

        painter.setBrush(gradient)
        painter.setPen(QPen(base_color.darker(140), 2.0))
        painter.drawEllipse(location, radius, radius)

        # Highlight
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 255, 255, 50))
        highlight_offset = int(radius * 0.3)
        highlight_size = int(radius * 0.4)
        painter.drawEllipse(location.x() - highlight_offset, location.y() - highlight_offset, highlight_size, highlight_size)

    def draw_port_labels(self, block, painter: Optional[QPainter], show_labels: bool = True) -> None:
        """
        Draw labels next to input and output ports for blocks with multiple ports.

        Args:
            block: The DBlock instance
            painter: QPainter instance for rendering
            show_labels: Whether to show port labels (can be toggled in settings)
        """
        if painter is None or not show_labels:
            return

        # Only show labels for blocks with multiple ports on either side
        if block.in_ports <= 1 and block.out_ports <= 1:
            return

        # Skip blocks where port meaning is obvious from the icon
        skip_blocks = {'Sum', 'SgProd', 'Product', 'Mux', 'Demux'}
        if block.block_fn in skip_blocks:
            return

        # Get port names from block
        input_names, output_names = block.get_port_names()

        # Setup font for labels (canonical UI stack instead of fixed Arial)
        label_font = get_ui_font(TYPE['caption'])
        label_font.setBold(False)
        painter.setFont(label_font)
        font_metrics = QFontMetrics(label_font)

        # Qt 5.11+ uses horizontalAdvance, older versions use width
        def get_text_width(text):
            if hasattr(font_metrics, 'horizontalAdvance'):
                return font_metrics.horizontalAdvance(text)
            return font_metrics.width(text)

        # Label colors
        text_color = theme_manager.get_color('text_primary')
        bg_color = theme_manager.get_color('background_secondary')
        bg_color.setAlpha(200)

        port_radius = block.port_radius
        label_offset = port_radius + 4  # Offset from port edge

        # Draw input port labels (on left side, text inside block)
        for i, port_coord in enumerate(block.in_coords):
            if i < len(input_names):
                label = input_names[i]
                # Position label to the right of the port (inside block)
                text_width = get_text_width(label)
                text_height = font_metrics.height()

                x = port_coord.x() + label_offset
                y = port_coord.y() + text_height // 4

                # Draw background for readability
                bg_rect = QRect(int(x - 2), int(y - text_height + 2), int(text_width + 4), int(text_height))
                painter.setPen(Qt.NoPen)
                painter.setBrush(bg_color)
                painter.drawRoundedRect(bg_rect, 2, 2)

                # Draw text
                painter.setPen(text_color)
                painter.drawText(int(x), int(y), label)

        # Draw output port labels (on right side, text inside block)
        for i, port_coord in enumerate(block.out_coords):
            if i < len(output_names):
                label = output_names[i]
                # Position label to the left of the port (inside block)
                text_width = get_text_width(label)
                text_height = font_metrics.height()

                x = port_coord.x() - label_offset - text_width
                y = port_coord.y() + text_height // 4

                # Draw background for readability
                bg_rect = QRect(int(x - 2), int(y - text_height + 2), int(text_width + 4), int(text_height))
                painter.setPen(Qt.NoPen)
                painter.setBrush(bg_color)
                painter.drawRoundedRect(bg_rect, 2, 2)

                # Draw text
                painter.setPen(text_color)
                painter.drawText(int(x), int(y), label)

    def draw_resize_handles(self, block, painter):
        """Draw resize handles on the corners and edges of selected blocks."""
        if not block.selected:
            return

        try:
            from config.block_sizes import RESIZE_HANDLE_SIZE
            handle_size = RESIZE_HANDLE_SIZE
        except ImportError:
            handle_size = 8

        handle_color = theme_manager.get_color('accent_primary')
        border_color = theme_manager.get_color('border_primary')

        painter.save()
        painter.setPen(QPen(border_color, 1))
        painter.setBrush(handle_color)

        half_handle = handle_size // 2
        handles = {
            'top_left': (block.left - half_handle, block.top - half_handle),
            'top_right': (block.left + block.width - half_handle, block.top - half_handle),
            'bottom_left': (block.left - half_handle, block.top + block.height - half_handle),
            'bottom_right': (block.left + block.width - half_handle, block.top + block.height - half_handle),
            'top': (block.left + block.width//2 - half_handle, block.top - half_handle),
            'bottom': (block.left + block.width//2 - half_handle, block.top + block.height - half_handle),
            'left': (block.left - half_handle, block.top + block.height//2 - half_handle),
            'right': (block.left + block.width - half_handle, block.top + block.height//2 - half_handle),
        }

        for _, (x, y) in handles.items():
            painter.drawRect(int(x), int(y), int(handle_size), int(handle_size))

        painter.restore()

    def get_resize_handle_at(self, block, pos: QPoint):
        """
        Check if a position is over a resize handle.

        Args:
            block: The block to check handles for
            pos: QPoint position to check

        Returns:
            Handle name if over a handle, None otherwise
        """
        if not block.selected:
            return None

        try:
            from config.block_sizes import RESIZE_HANDLE_SIZE
            handle_size = RESIZE_HANDLE_SIZE
        except ImportError:
            handle_size = 8

        # Define handle positions (same as draw_resize_handles)
        half_handle = handle_size // 2
        handles = {
            'top_left': (block.left - half_handle, block.top - half_handle),
            'top_right': (block.left + block.width - half_handle, block.top - half_handle),
            'bottom_left': (block.left - half_handle, block.top + block.height - half_handle),
            'bottom_right': (block.left + block.width - half_handle, block.top + block.height - half_handle),
            'top': (block.left + block.width//2 - half_handle, block.top - half_handle),
            'bottom': (block.left + block.width//2 - half_handle, block.top + block.height - half_handle),
            'left': (block.left - half_handle, block.top + block.height//2 - half_handle),
            'right': (block.left + block.width - half_handle, block.top + block.height//2 - half_handle),
        }

        # Check if position is within any handle
        for handle_name, (x, y) in handles.items():
            if (x <= pos.x() <= x + handle_size and
                y <= pos.y() <= y + handle_size):
                return handle_name

        return None

    def _draw_legacy_icon(self, block, painter, path):
        """Helper to draw legacy icons that use direct calls or mess with fonts."""
        # Using if/elif chain copied from original block.py
        if path.isEmpty() and block.block_fn == "Step":
            path.moveTo(0.1, 0.7); path.lineTo(0.5, 0.7); path.lineTo(0.5, 0.3); path.lineTo(0.9, 0.3)
        elif path.isEmpty() and block.block_fn == "Ramp":
            path.moveTo(0.1, 0.9); path.lineTo(0.9, 0.1)
        elif path.isEmpty() and block.block_fn == "Sine":
            path.moveTo(0.1, 0.5); path.quadTo(0.3, 0.1, 0.5, 0.5); path.quadTo(0.7, 0.9, 0.9, 0.5)
        elif block.block_fn == "SgProd":
            path.moveTo(0.2, 0.2); path.lineTo(0.8, 0.8)
            path.moveTo(0.2, 0.8); path.lineTo(0.8, 0.2)
        elif block.block_fn == "TranFn":
            self._draw_text_icon(block, painter, ["B(s)", "A(s)"], italic=True)
        elif block.block_fn == "Demux":
            path.moveTo(0.2, 0.5); path.lineTo(0.4, 0.5)
            path.moveTo(0.4, 0.2); path.lineTo(0.4, 0.8); path.lineTo(0.8, 0.8); path.lineTo(0.8, 0.2); path.lineTo(0.4, 0.2)
            path.moveTo(0.8, 0.3); path.lineTo(1.0, 0.3)
            path.moveTo(0.8, 0.7); path.lineTo(1.0, 0.7)
        elif block.block_fn == "Mux":
            path.moveTo(0.2, 0.3); path.lineTo(0.4, 0.3)
            path.moveTo(0.2, 0.7); path.lineTo(0.4, 0.7)
            path.moveTo(0.4, 0.2); path.lineTo(0.8, 0.4); path.lineTo(0.8, 0.6); path.lineTo(0.4, 0.8); path.lineTo(0.4, 0.2)
            path.moveTo(0.8, 0.5); path.lineTo(1.0, 0.5)
        elif block.block_fn == "BodeMag":
            path.moveTo(0.1, 0.9); path.lineTo(0.9, 0.9)
            path.moveTo(0.1, 0.9); path.lineTo(0.1, 0.1)
            path.moveTo(0.1, 0.4); path.lineTo(0.4, 0.4); path.lineTo(0.6, 0.7); path.lineTo(0.9, 0.7)
        elif block.block_fn == "RootLocus":
            path.moveTo(0.1, 0.5); path.lineTo(0.9, 0.5)
            path.moveTo(0.5, 0.1); path.lineTo(0.5, 0.9)
            p_x, p_y = 0.4, 0.3; path.moveTo(p_x-0.03, p_y-0.03); path.lineTo(p_x+0.03, p_y+0.03); path.moveTo(p_x+0.03, p_y-0.03); path.lineTo(p_x-0.03, p_y+0.03)
            p_x, p_y = 0.4, 0.7; path.moveTo(p_x-0.03, p_y-0.03); path.lineTo(p_x+0.03, p_y+0.03); path.moveTo(p_x+0.03, p_y-0.03); path.lineTo(p_x-0.03, p_y+0.03)
            path.addEllipse(QPointF(0.2*block.width, 0.5*block.height), 3, 3) 
            path.moveTo(0.4, 0.3); path.quadTo(0.3, 0.3, 0.2, 0.5)
            path.moveTo(0.4, 0.7); path.quadTo(0.3, 0.7, 0.2, 0.5)
            path.moveTo(0.2, 0.5); path.lineTo(0.1, 0.5)
        elif block.block_fn == "LQR":
            self._draw_centered_text(block, painter, "LQR", bold=True, size_delta=2)
        elif block.block_fn == "Deriv":
            self._draw_text_icon(block, painter, ["dy", "dt"], italic=True)
        elif block.block_fn == "DiscreteTranFn":
            self._draw_text_icon(block, painter, ["B(z)", "A(z)"], italic=True)
        elif block.block_fn == "Integrator":
             self._draw_text_icon(block, painter, ["1", "s"], italic=True, size_delta=4)
        elif block.block_fn == "Scope":
            path.moveTo(0.1, 0.9); path.lineTo(0.9, 0.9); path.moveTo(0.1, 0.9); path.lineTo(0.1, 0.1)
            path.moveTo(0.1, 0.6); path.quadTo(0.3, 0.2, 0.5, 0.6); path.quadTo(0.7, 1.0, 0.9, 0.6)
        elif block.block_fn == "Sum":
             sign_text = block.params.get('sign', '++')
             if isinstance(sign_text, dict): sign_text = sign_text.get('default', '++')
             self._draw_centered_text(block, painter, str(sign_text), size_delta=4)
        elif block.block_fn == "Noise":
             path.moveTo(0.1, 0.5); path.lineTo(0.2, 0.3); path.lineTo(0.3, 0.7); path.lineTo(0.4, 0.4)
             path.lineTo(0.5, 0.6); path.lineTo(0.6, 0.2); path.lineTo(0.7, 0.8); path.lineTo(0.8, 0.5); path.lineTo(0.9, 0.6)
        elif block.block_fn == "Exp":
             self._draw_centered_text(block, painter, "eˣ", italic=True, size_delta=4)
        elif block.block_fn == "Display":
             params_source = getattr(block, 'exec_params', block.params) or block.params
             display_val = params_source.get('_display_value_', '---')
             # Dynamic character limit based on block width (approx 8 pixels per char)
             block_width = getattr(block, 'width', 80)
             max_chars = max(10, int(block_width / 8))
             if len(str(display_val)) > max_chars: display_val = str(display_val)[:max_chars-1] + "…"
             self._draw_centered_text(block, painter, str(display_val), bold=True, size_delta=2)
        elif block.block_fn == "Term":
             path.moveTo(0.5, 0.2); path.lineTo(0.5, 0.6)
             path.moveTo(0.2, 0.6); path.lineTo(0.8, 0.6)
             path.moveTo(0.3, 0.75); path.lineTo(0.7, 0.75)
             path.moveTo(0.4, 0.9); path.lineTo(0.6, 0.9)
        elif block.block_fn == "Export":
             path.moveTo(0.2, 0.2); path.lineTo(0.8, 0.2); path.lineTo(0.8, 0.8); path.lineTo(0.2, 0.8); path.lineTo(0.2, 0.2)
             path.moveTo(0.5, 0.5); path.lineTo(1.0, 0.5); path.moveTo(0.8, 0.3); path.lineTo(1.0, 0.5); path.lineTo(0.8, 0.7)
        elif block.block_fn == "ZeroOrderHold":
             path.moveTo(0.1, 0.8); path.lineTo(0.3, 0.8); path.lineTo(0.3, 0.5); path.lineTo(0.6, 0.5)
             path.lineTo(0.6, 0.2); path.lineTo(0.9, 0.2)
        elif block.block_fn == "PRBS":
             path.moveTo(0.1, 0.7); path.lineTo(0.18, 0.7); path.lineTo(0.18, 0.3); path.lineTo(0.32, 0.3)
             path.lineTo(0.32, 0.7); path.lineTo(0.45, 0.7); path.lineTo(0.45, 0.4); path.lineTo(0.6, 0.4)
             path.lineTo(0.6, 0.7); path.lineTo(0.78, 0.7); path.lineTo(0.78, 0.3); path.lineTo(0.9, 0.3); path.lineTo(0.9, 0.7)
        elif block.block_fn == "Hysteresis":
             path.moveTo(0.15, 0.75); path.lineTo(0.75, 0.75); path.lineTo(0.75, 0.25); path.lineTo(0.85, 0.25)
             path.moveTo(0.85, 0.25); path.lineTo(0.25, 0.25); path.lineTo(0.25, 0.75); path.lineTo(0.15, 0.75)
             path.moveTo(0.45, 0.75); path.lineTo(0.45, 0.72); path.lineTo(0.51, 0.75); path.lineTo(0.45, 0.78); path.lineTo(0.45, 0.75)
             path.moveTo(0.55, 0.25); path.lineTo(0.55, 0.22); path.lineTo(0.49, 0.25); path.lineTo(0.55, 0.28); path.lineTo(0.55, 0.25)
        elif block.block_fn == "Deadband":
             path.moveTo(0.15, 0.80); path.lineTo(0.35, 0.50); path.lineTo(0.65, 0.50); path.lineTo(0.85, 0.20)
             path.moveTo(0.2, 0.5); path.lineTo(0.8, 0.5)
        elif block.block_fn == "Switch":
             path.moveTo(0.5, 0.10); path.lineTo(0.5, 0.35)
             path.moveTo(0.47, 0.30); path.lineTo(0.5, 0.35); path.lineTo(0.53, 0.30)
             path.moveTo(0.30, 0.35); path.lineTo(0.70, 0.35); path.lineTo(0.70, 0.75); path.lineTo(0.30, 0.75); path.lineTo(0.30, 0.35)
             path.moveTo(0.30, 0.45); path.lineTo(0.45, 0.45); path.moveTo(0.30, 0.65); path.lineTo(0.45, 0.65)
             path.moveTo(0.45, 0.45); path.lineTo(0.55, 0.55); path.lineTo(0.70, 0.55)
             path.moveTo(0.70, 0.55); path.lineTo(0.90, 0.55)
        elif block.block_fn == "Saturation":
             path.moveTo(0.1, 0.8); path.lineTo(0.9, 0.8); path.moveTo(0.1, 0.2); path.lineTo(0.9, 0.2)
             path.moveTo(0.15, 0.5); path.quadTo(0.3, 0.2, 0.45, 0.2); path.lineTo(0.55, 0.2); path.quadTo(0.7, 0.8, 0.85, 0.8)
        elif block.block_fn == "RateLimiter":
             path.moveTo(0.15, 0.75); path.lineTo(0.35, 0.75); path.lineTo(0.65, 0.25); path.lineTo(0.85, 0.25)
             path.moveTo(0.35, 0.75); path.lineTo(0.35, 0.25); path.lineTo(0.40, 0.25)
             self._draw_corner_label(block, painter, "du/dt")
        elif block.block_fn == "PID":
             self._draw_centered_text(block, painter, "PID", bold=True, size_delta=3)
             self._draw_corner_labels(block, painter, "sp", "pv")
        elif block.block_fn == "StateSpace":
             self._draw_centered_text(block, painter, "x' = Ax+Bu\ny = Cx+Du", size_delta=-1)
        elif block.block_fn == "DiscreteStateSpace":
             self._draw_centered_text(block, painter, "x[k+1]=Ax+Bu\ny[k]=Cx+Du", size_delta=-2)
        elif block.block_fn == "External":
             path.moveTo(0.2, 0.2); path.lineTo(0.8, 0.2); path.moveTo(0.2, 0.5); path.lineTo(0.6, 0.5)
             path.moveTo(0.2, 0.8); path.lineTo(0.8, 0.8); path.moveTo(0.2, 0.2); path.lineTo(0.2, 0.8)
        elif block.block_fn == "Constant":
             self._draw_centered_text(block, painter, "K", bold=True, size_delta=4)
        elif block.block_fn == "Delay":
             self._draw_centered_text(block, painter, "z⁻ⁿ", size_delta=2)
        elif block.block_fn == "Abs":
             self._draw_centered_text(block, painter, "|u|", bold=True, size_delta=4)
        elif block.block_fn == "TransportDelay":
             font = painter.font()
             orig = font.pointSize()
             font.setPointSize(orig + 3); font.setItalic(True); painter.setFont(font)
             painter.setPen(theme_manager.get_color('block_icon_color'))
             cx, cy = block.left + block.width // 2, block.top + block.height // 2
             painter.drawText(cx - 12, cy + 4, "e")
             font.setPointSize(orig); painter.setFont(font)
             painter.drawText(cx - 2, cy - 4, "-τs")
             font.setItalic(False); painter.setFont(font)
        elif block.block_fn == "XYGraph":
             path.moveTo(0.15, 0.85); path.lineTo(0.85, 0.85); path.moveTo(0.15, 0.85); path.lineTo(0.15, 0.15)
             path.moveTo(0.82, 0.82); path.lineTo(0.85, 0.85); path.lineTo(0.82, 0.88)
             path.moveTo(0.12, 0.18); path.lineTo(0.15, 0.15); path.lineTo(0.18, 0.18)
             path.moveTo(0.25, 0.75); path.quadTo(0.35, 0.35, 0.55, 0.45); path.quadTo(0.75, 0.55, 0.70, 0.30)
        elif block.block_fn == "Assert":
             self._draw_centered_text(block, painter, "⚠", bold=True, size_delta=6, color=theme_manager.get_current_theme()['error'])
        elif block.block_fn == "Selector":
             path.moveTo(0.15, 0.3); path.lineTo(0.15, 0.7); path.lineTo(0.35, 0.7); path.lineTo(0.35, 0.3); path.lineTo(0.15, 0.3)
             path.moveTo(0.17, 0.4); path.lineTo(0.33, 0.4); path.moveTo(0.17, 0.5); path.lineTo(0.33, 0.5); path.moveTo(0.17, 0.6); path.lineTo(0.33, 0.6)
             path.moveTo(0.35, 0.5); path.lineTo(0.65, 0.5); path.moveTo(0.60, 0.45); path.lineTo(0.65, 0.5); path.lineTo(0.60, 0.55)
             path.moveTo(0.70, 0.45); path.lineTo(0.85, 0.45); path.lineTo(0.85, 0.55); path.lineTo(0.70, 0.55); path.lineTo(0.70, 0.45)
        elif block.block_fn == "Subsystem":
             # Nested rectangles icon
             path.moveTo(0.2, 0.2); path.lineTo(0.8, 0.2); path.lineTo(0.8, 0.8); path.lineTo(0.2, 0.8); path.lineTo(0.2, 0.2)
             path.moveTo(0.3, 0.3); path.lineTo(0.7, 0.3); path.lineTo(0.7, 0.7); path.lineTo(0.3, 0.7); path.lineTo(0.3, 0.3)
        elif block.block_fn == "Inport":
             self._draw_centered_text(block, painter, "In", bold=True, size_delta=2)
             # Draw arrow?
             # path.moveTo(0.2, 0.5); path.lineTo(0.8, 0.5); path.lineTo(0.6, 0.3)...
        elif block.block_fn == "Outport":
             self._draw_centered_text(block, painter, "Out", bold=True, size_delta=2)
             
        elif block.block_fn == "FFT":
             path.moveTo(0.15, 0.80); path.lineTo(0.15, 0.20); path.moveTo(0.15, 0.80); path.lineTo(0.85, 0.80)
             bar_positions = [0.22, 0.32, 0.42, 0.52, 0.62, 0.72]
             bar_heights = [0.30, 0.55, 0.70, 0.45, 0.25, 0.15]
             bar_w = 0.06
             for x, h in zip(bar_positions, bar_heights):
                 path.moveTo(x, 0.80); path.lineTo(x, 0.80 - h * 0.55); path.lineTo(x + bar_w, 0.80 - h * 0.55); path.lineTo(x + bar_w, 0.80)
        elif block.block_fn == "MathFunction":
             self._draw_centered_text(block, painter, "f(u)", italic=True, size_delta=4)

    # --- Helper methods for common icon patterns ---

    def _draw_text_icon(self, block, painter, lines, italic=False, size_delta=2):
        font = painter.font()
        orig = font.pointSize()
        font.setPointSize(orig + size_delta)
        font.setItalic(italic)
        painter.setFont(font)
        painter.setPen(theme_manager.get_color('block_icon_color'))

        rect_top = QRect(block.left, block.top, block.width, block.height // 2)
        painter.drawText(rect_top, Qt.AlignCenter, lines[0])
        
        line_y = block.top + block.height // 2
        painter.drawLine(block.left + 10, line_y, block.left + block.width - 10, line_y)
        
        rect_bot = QRect(block.left, block.top + block.height // 2, block.width, block.height // 2)
        painter.drawText(rect_bot, Qt.AlignCenter, lines[1])
        
        font.setItalic(False)
        font.setPointSize(orig)
        painter.setFont(font)

    def _draw_centered_text(self, block, painter, text, bold=False, italic=False, size_delta=0, color=None):
        font = painter.font()
        orig = font.pointSize()
        font.setPointSize(orig + size_delta)
        font.setBold(bold)
        font.setItalic(italic)
        painter.setFont(font)
        painter.setPen(QColor(color) if color else theme_manager.get_color('block_icon_color'))
        rect = QRect(block.left, block.top, block.width, block.height)
        painter.drawText(rect, Qt.AlignCenter, text)
        font.setBold(False); font.setItalic(False); font.setPointSize(orig)
        painter.setFont(font)

    def _draw_corner_label(self, block, painter, text):
        font = painter.font()
        orig = font.pointSize()
        font.setPointSize(orig - 1)
        painter.setFont(font)
        painter.setPen(theme_manager.get_color('block_icon_color'))
        painter.drawText(QRect(block.left + int(block.width*0.4), block.top + int(block.height*0.4), int(block.width*0.6), int(block.height*0.6)), Qt.AlignCenter, text)
        font.setPointSize(orig)
        painter.setFont(font)

    def _draw_corner_labels(self, block, painter, tl, bl):
        font = painter.font()
        orig = font.pointSize()
        orig_bold = font.bold()
        orig_italic = font.italic()
        font.setPointSize(orig - 1)
        # Normalize bold/italic so this helper does not inherit state left on
        # the painter font by a prior icon helper.
        font.setBold(False)
        font.setItalic(False)
        painter.setFont(font)
        painter.drawText(QRect(block.left + 4, block.top + 2, block.width // 2, block.height // 2), Qt.AlignLeft | Qt.AlignTop, tl)
        painter.drawText(QRect(block.left + 4, block.top + block.height // 2, block.width // 2, block.height // 2), Qt.AlignLeft | Qt.AlignBottom, bl)
        font.setPointSize(orig)
        font.setBold(orig_bold)
        font.setItalic(orig_italic)
        painter.setFont(font)

    def _draw_sample_rate_indicator(self, block, painter: Optional[QPainter]) -> None:
        """
        Draw a small colored indicator for discrete blocks showing their sample rate.

        Indicator colors:
        - No indicator: Continuous block (sample_time < 0 or not set)
        - Gray dot: Inherited sample time (sample_time = 0)
        - Colored dot: Fixed discrete rate (sample_time > 0)
          - Red = fast (small sample time)
          - Blue = slow (large sample time)
          - Gradient between based on log scale

        Args:
            block: The DBlock instance
            painter: QPainter instance for rendering
        """
        if painter is None:
            return

        # Get sample time from block params
        sample_time = block.params.get('sampling_time',
                      block.params.get('sample_time',
                      block.params.get('output_sample_time', -1.0)))

        try:
            sample_time = float(sample_time)
        except (ValueError, TypeError):
            sample_time = -1.0

        # No indicator for continuous blocks
        if sample_time < 0:
            return

        # Indicator properties
        indicator_radius = 5
        indicator_x = block.left + block.width - indicator_radius - 3
        indicator_y = block.top + indicator_radius + 3

        # Determine indicator color
        if sample_time == 0:
            # Inherited rate - gray
            indicator_color = QColor(128, 128, 128)
        else:
            # Fixed discrete rate - color based on sample time
            # Use log scale: 0.001s (1kHz) = red, 1s (1Hz) = blue
            log_min = math.log10(0.001)  # 1ms = fast (red)
            log_max = math.log10(1.0)    # 1s = slow (blue)
            log_sample = math.log10(max(0.001, min(1.0, sample_time)))

            # Normalize to 0-1 range
            t = (log_sample - log_min) / (log_max - log_min)
            t = max(0.0, min(1.0, t))

            # Interpolate from red (fast) to blue (slow)
            r = int(255 * (1 - t))
            g = int(100 * (1 - abs(t - 0.5) * 2))  # Green peak in middle
            b = int(255 * t)
            indicator_color = QColor(r, g, b)

        # Draw the indicator dot
        painter.save()
        painter.setPen(QPen(QColor(50, 50, 50), 1))
        painter.setBrush(indicator_color)
        painter.drawEllipse(
            QPoint(int(indicator_x), int(indicator_y)),
            indicator_radius,
            indicator_radius
        )
        painter.restore()
