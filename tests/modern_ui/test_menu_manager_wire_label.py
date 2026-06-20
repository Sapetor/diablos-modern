"""Focused test for MenuManager._wire_label.

menu_manager.py had zero coverage. _wire_label/_detach_wires carried dead
getattr fallbacks for src_block/dst_block/src_port/dst_port -- attribute names
DLine never defines (it uses srcblock/srcport/dstblock/dstport). This pins the
label format against the real DLine attribute names after that simplification.
"""
import pytest

from modern_ui.managers.menu_manager import MenuManager


class _StubLine:
    """Carries the attribute names DLine actually defines."""

    def __init__(self, srcblock, srcport, dstblock, dstport):
        self.srcblock = srcblock
        self.srcport = srcport
        self.dstblock = dstblock
        self.dstport = dstport


@pytest.mark.qt
def test_wire_label_uses_dline_attribute_names(qapp):
    mm = MenuManager(canvas=object())
    label = mm._wire_label(_StubLine('gain0', 0, 'sum0', 1))
    assert label == "wire: gain0.out[0] → sum0.in[1]"


@pytest.mark.qt
def test_wire_label_defaults_when_attrs_missing(qapp):
    mm = MenuManager(canvas=object())
    # A bare object exposes none of the wire attributes -> empty names, port 0.
    label = mm._wire_label(object())
    assert label == "wire: .out[0] → .in[0]"
