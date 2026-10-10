"""Wires are tinted with their sample rate (same scale as the block dot).

The engine resolves rates only when a run starts, so the canvas repeats the
propagation rule in ``modern_ui/renderers/sample_time_colors.py``.
"""

import types

import pytest
from PyQt6.QtCore import QPoint
from PyQt6.QtTest import QTest

from modern_ui.renderers.sample_time_colors import (
    block_sample_times,
    rate_color,
    wire_sample_times,
)


def _blk(name, ts, **params):
    return types.SimpleNamespace(name=name, params=params, resolve_sample_time=lambda: ts)


def _line(src, dst):
    return types.SimpleNamespace(srcblock=src, dstblock=dst)


class TestPropagation:
    def test_inherited_block_takes_the_fastest_input_rate(self):
        blocks = [_blk("a", 0.1), _blk("b", 0.01), _blk("sum", 0.0), _blk("out", 0.0)]
        lines = [_line("a", "sum"), _line("b", "sum"), _line("sum", "out")]
        rates = block_sample_times(blocks, lines)
        assert rates["sum"] == 0.01
        assert rates["out"] == 0.01

    def test_inherited_without_discrete_input_is_continuous(self):
        blocks = [_blk("tf", -1.0), _blk("g", 0.0)]
        assert block_sample_times(blocks, [_line("tf", "g")])["g"] == -1.0

    def test_wire_carries_its_source_rate(self):
        blocks = [_blk("zoh", 0.05), _blk("tf", -1.0), _blk("scope", -1.0)]
        a, b = _line("zoh", "tf"), _line("tf", "scope")
        rates = wire_sample_times(blocks, [a, b])
        assert rates[id(a)] == 0.05
        assert rates[id(b)] == -1.0

    def test_rate_transition_output_uses_output_sample_time(self):
        blocks = [_blk("rt", -1.0, output_sample_time=0.2), _blk("scope", -1.0)]
        a = _line("rt", "scope")
        assert wire_sample_times(blocks, [a])[id(a)] == 0.2

    def test_rate_colors(self):
        assert rate_color(-1.0) is None
        fast, slow = rate_color(0.001), rate_color(1.0)
        assert fast.red() > fast.blue() and slow.blue() > slow.red()


@pytest.fixture
def window(qapp):
    from modern_ui.main_window import ModernDiaBloSWindow

    w = ModernDiaBloSWindow()
    w.resize(1200, 800)
    w.show()
    qapp.processEvents()
    yield w
    w.close()


@pytest.mark.qt
def test_discrete_wire_is_drawn_in_its_rate_color(window, qapp):
    canvas = window.canvas
    mb = {m.block_fn: m for m in window.dsim.menu_blocks}
    zoh = canvas.add_block_from_palette(mb["ZeroOrderHold"], QPoint(200, 300))
    zoh.params["sampling_time"] = 0.001  # fastest end of the scale: red
    scope = canvas.add_block_from_palette(mb["Scope"], QPoint(600, 300))
    window.dsim.add_line((zoh.name, 0, zoh.out_coords[0]), (scope.name, 0, scope.in_coords[0]))
    for b in window.dsim.blocks_list:
        b.selected = False
    for ln in window.dsim.line_list:
        ln.selected = False
    canvas.update()
    # The welcome overlay hides on a deferred timer; let it go first.
    for _ in range(4):
        qapp.processEvents()
        QTest.qWait(5)
    assert not canvas.welcome_overlay.isVisible()

    img = canvas.grab().toImage()
    a, b = canvas.world_to_screen(zoh.out_coords[0]), canvas.world_to_screen(scope.in_coords[0])
    y = (a.y() + b.y()) // 2
    xs = range(a.x() + 20, b.x() - 20)
    reds = [img.pixelColor(x, yy) for x in xs for yy in range(y - 2, y + 3)]
    assert any(c.red() > 150 and c.blue() < 90 and c.green() < 120 for c in reds), (
        "no red (1 ms rate) pixels along the ZOH -> Scope wire"
    )
