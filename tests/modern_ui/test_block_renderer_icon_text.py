"""BlockRenderer._draw_icon_text: the painter-drawn text half of a block icon.

A block's draw_icon() path is stroked first; text that a QPainterPath cannot
express (1/s, PID, the Gain value, a Goto tag ...) is painted afterwards from
the _FRACTION_TEXT_ICONS / _CENTERED_TEXT_ICONS tables and the
_DYNAMIC_TEXT_ICONS dispatch. Blocks whose whole icon is a path draw no text,
and nothing here appends to the (shared, memoized) draw_icon path any more.
"""

import pytest
from PyQt5.QtGui import QPainterPath

from modern_ui.renderers import block_renderer as br
from modern_ui.renderers.block_renderer import BlockRenderer


@pytest.fixture(autouse=True)
def _qt(qapp):
    return qapp


class _Blk:
    """Minimal stand-in: the text overlay reads block_fn, geometry and params."""

    def __init__(self, block_fn, **params):
        self.block_fn = block_fn
        self.left = 0
        self.top = 0
        self.width = 80
        self.height = 60
        self.flipped = False
        self.params = params
        self.exec_params = {}
        self.block_instance = None
        self.in_coords = []


def _spy(monkeypatch, renderer):
    calls = []
    for name in (
        "_draw_centered_text",
        "_draw_text_icon",
        "_draw_corner_label",
        "_draw_corner_labels",
        "_draw_port_glyphs",
        "_draw_gain_value",
    ):
        monkeypatch.setattr(renderer, name, lambda *a, _n=name, **k: calls.append((_n, a[2:], k)))
    return calls


@pytest.mark.qt
class TestIconTextTables:
    @pytest.mark.parametrize("fn", ["Scope", "Constant", "Step", "Hysteresis", "Abs", "LQR"])
    def test_path_only_blocks_draw_no_text(self, monkeypatch, fn):
        r = BlockRenderer()
        calls = _spy(monkeypatch, r)
        r._draw_icon_text(_Blk(fn), object())
        assert calls == []

    def test_fraction_icon(self, monkeypatch):
        r = BlockRenderer()
        calls = _spy(monkeypatch, r)
        r._draw_icon_text(_Blk("Integrator"), object())
        assert calls == [("_draw_text_icon", (["1", "s"],), {"italic": True, "size_delta": 4})]

    def test_centered_icon(self, monkeypatch):
        r = BlockRenderer()
        calls = _spy(monkeypatch, r)
        r._draw_icon_text(_Blk("MathFunction"), object())
        assert calls == [("_draw_centered_text", ("f(u)",), {"italic": True, "size_delta": 4})]

    def test_pid_draws_label_and_corner_ports(self, monkeypatch):
        r = BlockRenderer()
        calls = _spy(monkeypatch, r)
        r._draw_icon_text(_Blk("PID"), object())
        assert [c[0] for c in calls] == ["_draw_centered_text", "_draw_corner_labels"]
        assert calls[0][1] == ("PID",)
        assert calls[1][1] == ("sp", "pv")

    def test_rate_limiter_only_adds_corner_label(self, monkeypatch):
        r = BlockRenderer()
        calls = _spy(monkeypatch, r)
        r._draw_icon_text(_Blk("RateLimiter"), object())
        assert calls == [("_draw_corner_label", ("du/dt",), {})]

    def test_goto_tag_reads_param(self, monkeypatch):
        r = BlockRenderer()
        calls = _spy(monkeypatch, r)
        r._draw_icon_text(_Blk("Goto", tag="x"), object())
        assert calls == [("_draw_centered_text", ("[x]",), {"bold": True, "size_delta": 2})]

    def test_display_value_is_truncated_to_block_width(self, monkeypatch):
        r = BlockRenderer()
        calls = _spy(monkeypatch, r)
        blk = _Blk("Display")
        blk.params["_display_value_"] = "1234567890123456789"
        r._draw_icon_text(blk, object())
        (name, (text,), _kw) = calls[0]
        assert name == "_draw_centered_text"
        assert text.endswith("…") and len(text) == 10

    def test_sum_signs_go_to_port_glyphs(self, monkeypatch):
        r = BlockRenderer()
        calls = _spy(monkeypatch, r)
        r._draw_icon_text(_Blk("Sum", sign="+-"), object())
        assert calls == [("_draw_port_glyphs", (["+", "-"],), {})]

    def test_mask_wins_over_block_text(self, monkeypatch):
        r = BlockRenderer()
        calls = _spy(monkeypatch, r)
        monkeypatch.setattr(br, "_block_mask", lambda b: {"icon": "Σ", "name": "My"})
        r._draw_icon_text(_Blk("Integrator"), object())
        assert calls == [("_draw_centered_text", ("Σ",), {"bold": True, "size_delta": 1})]


@pytest.mark.qt
class TestIntrinsicSubsystemIcon:
    def test_plain_subsystem_gets_nested_squares(self, monkeypatch):
        monkeypatch.setattr(br, "_block_mask", lambda b: None)
        path = BlockRenderer()._icon_source_path(_Blk("Subsystem"))
        assert path.elementCount() == 10
        assert path is br._SUBSYSTEM_ICON_PATH

    def test_masked_subsystem_gets_no_path(self, monkeypatch):
        monkeypatch.setattr(br, "_block_mask", lambda b: {"name": "M"})
        assert BlockRenderer()._icon_source_path(_Blk("Subsystem")).isEmpty()

    def test_other_instanceless_block_gets_empty_path(self):
        path = BlockRenderer()._icon_source_path(_Blk("Inport"))
        assert isinstance(path, QPainterPath) and path.isEmpty()
