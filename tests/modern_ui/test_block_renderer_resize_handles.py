"""Tests for BlockRenderer resize-handle geometry.

draw_resize_handles and get_resize_handle_at previously each hand-rolled the
same 8-handle position dict (a verbatim duplicate). They now share
_resize_handle_rects; these tests pin that the hit-test resolves exactly the
geometry the draw path uses, and the selection/miss guards.
"""
import pytest
from PyQt5.QtCore import QPoint

from modern_ui.renderers.block_renderer import BlockRenderer


@pytest.fixture(autouse=True)
def _qt(qapp):
    return qapp


class _Blk:
    def __init__(self, selected=True):
        self.left = 100
        self.top = 80
        self.width = 90
        self.height = 60
        self.selected = selected


@pytest.mark.qt
class TestResizeHandleGeometry:
    def test_hit_test_resolves_each_handle_at_its_center(self):
        r = BlockRenderer()
        block = _Blk(selected=True)
        size = r._resize_handle_size()
        rects = r._resize_handle_rects(block)
        assert set(rects) == {
            'top_left', 'top_right', 'bottom_left', 'bottom_right',
            'top', 'bottom', 'left', 'right',
        }
        # The center of each handle rect must hit-test back to that handle.
        for name, (x, y) in rects.items():
            center = QPoint(int(x + size / 2), int(y + size / 2))
            assert r.get_resize_handle_at(block, center) == name

    def test_point_away_from_handles_returns_none(self):
        r = BlockRenderer()
        block = _Blk(selected=True)
        # Dead center of the block is far from every edge/corner handle.
        center = QPoint(block.left + block.width // 2, block.top + block.height // 2)
        assert r.get_resize_handle_at(block, center) is None

    def test_unselected_block_has_no_handles(self):
        r = BlockRenderer()
        block = _Blk(selected=False)
        assert r.get_resize_handle_at(block, QPoint(block.left, block.top)) is None
