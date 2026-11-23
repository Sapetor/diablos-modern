"""Editing a block's port counts (canvas menu > Edit Ports).

``DBlock.change_port_numbers`` used to open the PyQt6 dialog itself, which
tied the engine to Qt; the GUI now runs the dialog and the block only
applies the values. Cancelling also used to push an undo step and mark the
diagram modified although nothing had changed.
"""

import pytest
from PyQt6.QtCore import QPoint

from diablos.gui.widgets import simulation_dialogs

pytestmark = pytest.mark.qt


def _block(dsim, fn):
    menu = {b.fn_name: b for b in dsim.menu_blocks}
    return dsim.add_block(menu[fn], QPoint(100, 100))


def test_change_port_numbers_applies_only_what_io_edit_allows(qapp):
    from diablos.core.lib import DSim

    dsim = DSim()
    mux = _block(dsim, "mux")  # io_edit "input"
    assert mux.editable_ports() == {"inputs": mux.in_ports}
    outs = mux.out_ports
    assert mux.change_port_numbers({"inputs": 4, "outputs": 9}) is True
    assert (mux.in_ports, mux.out_ports) == (4, outs)
    assert mux.params["_inputs_"] == 4 and len(mux.in_coords) == 4
    mux.change_port_numbers({"inputs": "-2"})  # invalid: keep the count
    assert mux.in_ports == 4

    step = _block(dsim, "step")  # fixed ports
    assert step.editable_ports() == {}
    assert step.change_port_numbers({"outputs": 3}) is False


class _FakeDialog:
    accept = True
    values = {"inputs": 5}

    def __init__(self, name, current):
        self.current = current

    def exec(self):
        return self.accept

    def get_values(self):
        return dict(self.values)


@pytest.fixture
def window(qapp):
    from diablos.gui.main_window import ModernDiaBloSWindow

    w = ModernDiaBloSWindow()
    yield w
    w.dsim.dirty = False
    w.close()


@pytest.mark.parametrize("accept", [True, False])
def test_edit_ports_from_the_canvas_menu(window, monkeypatch, accept):
    canvas = window.canvas
    mux = _block(canvas.dsim, "mux")
    canvas.dsim.dirty = False
    undo_before = len(canvas.history_manager.undo_stack)
    monkeypatch.setattr(_FakeDialog, "accept", accept)
    monkeypatch.setattr(simulation_dialogs, "PortDialog", _FakeDialog)

    canvas.menu_manager._edit_ports(mux)

    if accept:
        assert mux.in_ports == 5
        assert len(canvas.history_manager.undo_stack) == undo_before + 1
        assert canvas.dsim.dirty is True
    else:
        assert mux.in_ports != 5
        assert len(canvas.history_manager.undo_stack) == undo_before
        assert canvas.dsim.dirty is False
