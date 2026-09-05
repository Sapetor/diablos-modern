"""Pixel-identity and behaviour tests for the renderer paint-object caches.

``modern_ui/renderers/block_renderer.py`` memoizes the objects it used to
rebuild for every block on every frame (outline + shadow QPainterPaths, the
body/port gradients, pens, the icon path from ``draw_icon``, the icon
QTransform and the label fonts). The contract those caches must honour is
simply: *the painted result is bit-for-bit what an uncached renderer would
produce*, for every block type, shape, flip and selection state.

The tests below render a wide diagram twice -- once with the caches flushed
before every single frame (the "cold"/uncached behaviour) and once warm -- and
compare the raw QImage bytes. Anything that makes a cache key too coarse (a
missing geometry component, a shared mutable QColor, a gradient whose centre
was not re-set) shows up as a byte difference.

Run with:
    QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python -m pytest \
        tests/modern_ui/test_renderer_paint_caches.py -p no:cacheprovider -o addopts=""
"""

import hashlib
import time

import pytest
from PyQt5.QtCore import QPoint, QRect
from PyQt5.QtGui import QImage, QPainter

from lib.lib import DSim
from lib.simulation.block import DBlock
from lib.simulation.connection import DLine
from modern_ui.renderers import block_renderer as br
from modern_ui.themes.theme_manager import theme_manager

pytestmark = pytest.mark.qt


# A spread of block types that between them exercise every icon route through
# the renderer: draw_icon paths, legacy path branches, direct-to-painter text
# icons (TranFn/PID/Display), the gain-value fitter, the Sum/Product port
# glyphs, and the four outline shapes (rect / triangle / circle / tag).
SAMPLE_BLOCK_FNS = (
    "Step",
    "Ramp",
    "Sine",
    "Gain",
    "MatrixGain",
    "Sum",
    "Product",
    "SgProd",
    "Scope",
    "Integrator",
    "TranFn",
    "Mux",
    "Demux",
    "Constant",
    "Display",
    "Term",
    "Export",
    "ZeroOrderHold",
    "Saturation",
    "RateLimiter",
    "PID",
    "StateSpace",
    "Abs",
    "Delay",
    "Noise",
    "Exp",
    "Goto",
    "From",
    "Deriv",
    "BodeMag",
    "RootLocus",
    "MathFunction",
    "Selector",
    "Subsystem",
    "Inport",
    "Outport",
    "FFT",
    "Switch",
    "Deadband",
    "Hysteresis",
    "PRBS",
    "XYGraph",
    "Assert",
    "LQR",
    "TransportDelay",
    "DiscreteTranFn",
    "DiscreteStateSpace",
    "External",
)

CANVAS_W, CANVAS_H = 1200, 900

_BLOCK_CLASSES = None


def _block_class(block_fn):
    """Resolve ``block_fn`` -> BaseBlock subclass so DBlock gets a real icon.

    Without a ``block_class`` a DBlock has ``block_instance = None`` and the
    polymorphic ``draw_icon`` path -- one of the things being cached -- is never
    reached, so these tests would pass vacuously.
    """
    global _BLOCK_CLASSES
    if _BLOCK_CLASSES is None:
        from lib.block_loader import load_blocks

        _BLOCK_CLASSES = {}
        for cls in load_blocks():
            try:
                _BLOCK_CLASSES[cls().block_name] = cls
            except Exception:  # a block that cannot be default-constructed
                continue
    return _BLOCK_CLASSES.get(block_fn)


def _build_dsim():
    """A dense diagram: every sample block type, some flipped, some selected."""
    dsim = DSim()
    blocks = []
    cols = 8
    for i, fn in enumerate(SAMPLE_BLOCK_FNS):
        block = DBlock(
            block_fn=fn,
            sid=i,
            coords=QRect(60 + (i % cols) * 130, 60 + (i // cols) * 120, 90, 70),
            color="#4CAF50",
            in_ports=2,
            out_ports=1,
            b_type=2,
            io_edit="both",
            fn_name=fn.lower(),
            params={"gain": 2.5, "tag": "A", "value": 1.0},
            external=False,
            colors=None,
            block_class=_block_class(fn),
        )
        block.name = "%s%d" % (fn, i)
        block.flipped = i % 3 == 0
        block.selected = i % 7 == 0
        blocks.append(block)
    dsim.blocks_list.extend(blocks)
    for i in range(0, len(blocks) - 1, 2):
        src, dst = blocks[i], blocks[i + 1]
        dsim.line_list.append(
            DLine(
                i,
                src.name,
                0,
                dst.name,
                0,
                [QPoint(src.left + 90, src.top + 35), QPoint(dst.left, dst.top + 35)],
            )
        )
    return dsim


@pytest.fixture
def canvas(qapp):
    from modern_ui.widgets.modern_canvas import ModernCanvas

    widget = ModernCanvas(_build_dsim())
    widget.resize(CANVAS_W, CANVAS_H)
    return widget


def _flush_caches():
    """Put the renderer back in the state it would be in with no caching."""
    br.clear_render_caches()
    theme_manager._color_cache.clear()


def _render(canvas):
    image = QImage(CANVAS_W, CANVAS_H, QImage.Format_ARGB32)
    image.fill(0)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing, True)
    canvas.canvas_renderer.draw_grid(
        painter, QRect(0, 0, CANVAS_W, CANVAS_H), CANVAS_W, CANVAS_H, True
    )
    canvas.rendering_manager.render_content(painter)
    painter.end()
    return image


def _digest(image):
    bits = image.bits()
    bits.setsize(image.byteCount())
    return hashlib.sha256(bytes(bits)).hexdigest()


class TestCachedRenderIsPixelIdentical:
    def test_warm_cache_matches_cold_cache(self, canvas):
        """A cached frame is byte-identical to an uncached one."""
        _flush_caches()
        cold = _digest(_render(canvas))
        warm = _digest(_render(canvas))  # second frame reuses every cache
        assert warm == cold

        # ...and flushing mid-stream must not change anything either.
        _flush_caches()
        assert _digest(_render(canvas)) == cold

    def test_moving_a_block_repaints_it_at_the_new_place(self, canvas):
        """Geometry is part of every cache key, so a move must not be stale."""
        before = _digest(_render(canvas))

        block = canvas.dsim.blocks_list[0]
        block.relocate_Block(QPoint(block.left + 37, block.top + 23))
        moved = _digest(_render(canvas))
        assert moved != before

        # And the moved frame equals what a cold renderer produces.
        _flush_caches()
        assert _digest(_render(canvas)) == moved

    def test_flipping_a_block_repaints_it(self, canvas):
        before = _digest(_render(canvas))
        block = canvas.dsim.blocks_list[1]
        block.flipped = not block.flipped
        after = _digest(_render(canvas))
        assert after != before
        _flush_caches()
        assert _digest(_render(canvas)) == after

    def test_theme_change_repaints_with_new_colors(self, canvas):
        """The get_color memo must be invalidated when the theme changes."""
        from modern_ui.themes.theme_manager import ThemeType

        original = theme_manager.current_theme
        try:
            theme_manager.set_theme(ThemeType.DARK)
            dark = _digest(_render(canvas))
            theme_manager.set_theme(ThemeType.LIGHT)
            light = _digest(_render(canvas))
            assert dark != light
            theme_manager.set_theme(ThemeType.DARK)
            assert _digest(_render(canvas)) == dark
        finally:
            theme_manager.set_theme(original)


class TestCacheMechanics:
    def test_outline_path_is_reused_for_identical_geometry(self, canvas):
        block = canvas.dsim.blocks_list[0]
        _flush_caches()
        first = br.cached_outline_path(block, "rect")
        assert br.cached_outline_path(block, "rect") is first

    def test_outline_cache_keys_on_shape_offset_and_expand(self, canvas):
        block = canvas.dsim.blocks_list[0]
        _flush_caches()
        rect = br.cached_outline_path(block, "rect")
        assert br.cached_outline_path(block, "triangle") is not rect
        assert br.cached_outline_path(block, "rect", offset=3) is not rect
        assert br.cached_outline_path(block, "rect", expand=2) is not rect

    def test_caches_are_bounded(self, canvas):
        """Dragging mints a fresh key per frame; the caches must not grow forever."""
        block = canvas.dsim.blocks_list[0]
        _flush_caches()
        for dx in range(br._CACHE_MAX + 50):
            block.left = dx
            br.cached_outline_path(block, "rect")
        assert len(br._OUTLINE_CACHE) <= br._CACHE_MAX

    def test_icon_source_path_is_memoized_per_class_and_geometry(self, canvas):
        block = next(b for b in canvas.dsim.blocks_list if b.block_instance is not None)
        renderer = br.BlockRenderer()
        _flush_caches()
        first = renderer._icon_source_path(block)
        assert renderer._icon_source_path(block) is first
        block.relocate_Block(QPoint(block.left + 11, block.top))
        assert renderer._icon_source_path(block) is not first

    def test_draw_icon_failure_is_not_memoized(self, canvas):
        """A transient draw_icon error must not freeze an empty icon in place."""
        block = next(b for b in canvas.dsim.blocks_list if b.block_instance is not None)
        renderer = br.BlockRenderer()
        _flush_caches()

        calls = {"n": 0}
        good = block.block_instance.draw_icon

        def flaky(rect):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("boom")
            return good(rect)

        block.block_instance.draw_icon = flaky
        try:
            assert renderer._icon_source_path(block).isEmpty()
            assert not renderer._icon_source_path(block).isEmpty()
        finally:
            del block.block_instance.draw_icon


@pytest.mark.slow
def test_repeated_offscreen_render_completes(canvas):
    """Render the sample diagram 50x offscreen; a regression here is a hang."""
    _flush_caches()
    _render(canvas)  # warm up fonts / caches
    start = time.perf_counter()
    for _ in range(50):
        _render(canvas)
    elapsed = time.perf_counter() - start
    # Extremely loose: this asserts "still finishes", not a perf number.
    assert elapsed < 60.0
