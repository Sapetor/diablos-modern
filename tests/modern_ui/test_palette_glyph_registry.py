"""
Palette glyph registry: every kind the block-name mapping can emit resolves
to a painter (or to the documented text fallback), unknown kinds fall back the
same way the old if/elif switch did, and each registered painter actually
puts ink on the tile.

``_draw_glyph`` used to be one 36-branch switch over ``kind``; it is now a
lookup into ``_GLYPHS``. These tests lock in the contract of that table so a
kind can't silently drop off the registry and start rendering as a 3-letter
label.
"""

import pytest

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QImage, QPainter, QPen

from modern_ui.widgets import modern_palette as mp

_BG = QColor(17, 34, 51, 255)
_FG = QColor(30, 60, 200)
_TILE = mp._BlockGlyphLabel.SIZE

# Every glyph kind `_glyph_kind_for` can return for a known block name. Kept
# explicit (not derived from the mapping) so a mapping edit that renames a
# kind shows up here as a deliberate change.
_MAPPED_KINDS = {
    "sine",
    "noise",
    "step",
    "ramp",
    "impulse",
    "const",
    "sum",
    "gain",
    "product",
    "abs",
    "sqrt",
    "matrix",
    "fn",
    "exp",
    "integ",
    "deriv",
    "pid",
    "tranfn",
    "state",
    "lqr",
    "delay",
    "sat",
    "rate",
    "hys",
    "dead",
    "filter",
    "scope",
    "display",
    "bode",
    "nyq",
    "roots",
    "fft",
    "xy",
    "export",
    "term",
    "zoh",
    "foh",
    "mux",
    "demux",
    "switch",
    "sel",
    "goto",
    "from",
    "in",
    "out",
    "sub",
}

# Kinds that intentionally have no dedicated painter and render as a short
# label through the final fallback.
_LABEL_KINDS = {
    "const",
    "abs",
    "sqrt",
    "matrix",
    "fn",
    "exp",
    "deriv",
    "pid",
    "state",
    "lqr",
    "display",
    "sel",
    "goto",
    "from",
}


def _render(kind, s=_TILE, colour=_FG):
    """Render ``kind`` the way _BlockGlyphLabel.paintEvent does, into an ARGB image."""
    img = QImage(s, s, QImage.Format_ARGB32)
    img.fill(_BG)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing, True)
    pen = QPen(colour)
    pen.setWidthF(1.4)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    mp._draw_glyph(p, kind, colour, s)
    p.end()
    return img


def _blank(s=_TILE):
    img = QImage(s, s, QImage.Format_ARGB32)
    img.fill(_BG)
    return img


# The substring keys of the `_glyph_kind_for` table, in table order. Feeding
# each key back through the mapping yields the kind a block whose name
# contains that key actually gets.
_MAPPING_KEYS = [
    "sine",
    "wave",
    "noise",
    "prbs",
    "step",
    "ramp",
    "impulse",
    "constant",
    "sum",
    "gain",
    "product",
    "abs",
    "sqrt",
    "matrixgain",
    "mathfunction",
    "exp",
    "integ",
    "deriv",
    "pid",
    "tranfn",
    "transfer",
    "state",
    "lqr",
    "delay",
    "saturation",
    "rate",
    "hysteresis",
    "deadband",
    "filter",
    "scope",
    "display",
    "bode",
    "nyquist",
    "rootlocus",
    "fft",
    "xygraph",
    "export",
    "term",
    "zoh",
    "firstorder",
    "zero",
    "hold",
    "mux",
    "demux",
    "switch",
    "selector",
    "goto",
    "from",
    "inport",
    "outport",
    "sub",
]

# Kinds the table declares but can never emit because an earlier, shorter key
# is a substring of theirs ("gain" in "matrixgain", "exp" in "export", "mux" in
# "demux"). Pre-existing first-match behaviour, pinned here so a fix to the
# mapping shows up as a deliberate test change rather than a silent one.
_SHADOWED_KINDS = {"matrix", "export", "demux"}


def _mapping_kinds():
    """Kinds actually emitted by `_glyph_kind_for` for each mapping key."""
    return {mp._glyph_kind_for(key) for key in _MAPPING_KEYS}


@pytest.mark.unit
class TestGlyphRegistryCoverage:
    def test_mapping_kinds_are_registered_or_documented_labels(self):
        # Every kind the mapping can emit either has a painter or is one of the
        # kinds that intentionally render as a text label.
        missing = _MAPPED_KINDS - set(mp._GLYPHS) - _LABEL_KINDS
        assert not missing, f"kinds with neither painter nor label fallback: {sorted(missing)}"

    def test_label_kinds_have_no_painter(self):
        # The label set documents the fallback; if a painter is added for one of
        # them this list must be updated so the two stay in sync.
        assert not (_LABEL_KINDS & set(mp._GLYPHS))

    def test_mapping_emits_exactly_the_declared_kinds_minus_shadowed(self):
        # Every kind the table can really emit is declared, and the only
        # declared kinds it never emits are the known shadowed ones.
        emitted = _mapping_kinds()
        assert emitted == _MAPPED_KINDS - _SHADOWED_KINDS
        for key in _MAPPING_KEYS:
            assert mp._glyph_kind_for(key) in _MAPPED_KINDS

    def test_registry_dead_entries_are_only_the_shadowed_kinds(self):
        # `_glyph_export` and `_glyph_demux` are registered but unreachable
        # through the mapping today (see _SHADOWED_KINDS); nothing else is.
        unreachable = set(mp._GLYPHS) - _mapping_kinds()
        assert unreachable == _SHADOWED_KINDS & set(mp._GLYPHS) == {"export", "demux"}

    def test_hys_and_dead_share_one_painter(self):
        assert mp._GLYPHS["hys"] is mp._GLYPHS["dead"]


@pytest.mark.unit
@pytest.mark.qt
class TestGlyphPainting:
    @pytest.mark.parametrize("kind", sorted(mp._GLYPHS))
    def test_registered_painter_paints_something(self, qapp, kind):
        assert _render(kind) != _blank()

    @pytest.mark.parametrize("kind", sorted(_LABEL_KINDS))
    def test_label_fallback_paints_something(self, qapp, kind):
        assert _render(kind) != _blank()

    def test_letter_kind_paints_something(self, qapp):
        assert _render("letter:AB") != _blank()

    def test_unknown_kind_falls_back_to_three_letter_label(self, qapp):
        # Same fallback the old switch had: first three characters as a bold
        # label, so "zzzzzz" and "zzz" render identically and "zzz" != "abc".
        assert _render("zzzzzz") == _render("zzz")
        assert _render("zzz") != _render("abc")
        assert _render("zzz") != _blank()

    def test_empty_letter_kind_renders_question_mark(self, qapp):
        # `letter:` with no initials draws "?" (the mapping emits this for an
        # empty fn_name), and must not raise.
        assert _render("letter:") == _render("letter:?")
        assert _render("letter:") != _blank()

    def test_glyph_tile_uses_registry_kind(self, qapp):
        # The tile widget resolves its kind through the same mapping the
        # registry is verified against.
        class _MB:
            fn_name = "Integrator"
            block_class = None

        tile = mp._BlockGlyphLabel(_MB(), {})
        try:
            assert tile._glyph_kind == "integ"
            assert tile._glyph_kind in mp._GLYPHS
        finally:
            tile.deleteLater()
