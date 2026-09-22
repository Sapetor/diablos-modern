"""
A flipped block must come back from paste and from a save/load round trip with
its ports mirrored, exactly as the canvas flip action leaves them.

Both paths used to set ``block.flipped`` *after* the constructor had already laid
the ports out unflipped, without re-running ``update_Block()``, so wires attached
to the wrong edge until the block was next moved. For load that meant every saved
diagram containing a flipped block reopened wrong. ``DBlock.flipped`` is now a
property whose setter re-lays the ports.
"""

from types import SimpleNamespace

import pytest
from PyQt6.QtCore import QPoint, QRect

from lib.simulation.block import DBlock
from modern_ui.managers.clipboard_manager import ClipboardManager


@pytest.fixture(autouse=True)
def _qt(qapp):
    """DBlock builds a QPixmap, which needs a live QApplication."""
    return qapp


def _gain():
    return DBlock(
        block_fn="Gain",
        sid=0,
        coords=QRect(100, 100, 60, 40),
        color="#4CAF50",
        in_ports=1,
        out_ports=1,
        b_type=2,
        io_edit=False,
        fn_name="Gain",
        params={"gain": 2.0},
        external=False,
        username="",
        block_class=None,
        colors=None,
        category="Math",
    )


def _ports(block):
    return block.in_coords[0], block.out_coords[0]


def _paste(flipped):
    dsim = SimpleNamespace(blocks_list=[], menu_blocks=[], colors=None)
    manager = ClipboardManager(SimpleNamespace(dsim=dsim))
    manager.clipboard_blocks = [
        {
            "block_fn": "Gain",
            "coords": QRect(100, 100, 60, 40),
            "color": "#4CAF50",
            "category": "Math",
            "in_ports": 1,
            "out_ports": 1,
            "b_type": 2,
            "io_edit": False,
            "fn_name": "Gain",
            "params": {"gain": 2.0},
            "external": False,
            "flipped": flipped,
        }
    ]
    (pasted,) = manager._instantiate_pasted_blocks(QPoint(0, 0))
    return pasted


def _round_trip(flipped):
    from lib.lib import DSim
    from lib.services.file_service import FileService
    from lib.workspace import WorkspaceManager

    WorkspaceManager._instance = None
    source = DSim()
    menu_gain = next(b for b in source.model.menu_blocks if b.block_fn == "Gain")
    block = source.model.add_block(menu_gain, QPoint(100, 100))
    block.flipped = flipped
    data = FileService(source.model).serialize()

    WorkspaceManager._instance = None
    target = DSim()
    FileService(target.model).apply_loaded_data(data)
    loaded = next(b for b in target.model.blocks_list if b.block_fn == "Gain")
    return _ports(block), loaded


def test_setting_flipped_mirrors_the_ports():
    plain, block = _gain(), _gain()
    block.flipped = True
    assert _ports(block) == (plain.out_coords[0], plain.in_coords[0])
    block.flipped = False
    assert _ports(block) == _ports(plain)


@pytest.mark.parametrize("flipped", [True, False])
def test_paste_keeps_the_flip(flipped):
    reference = _gain()
    reference.flipped = flipped
    pasted = _paste(flipped)
    assert pasted.flipped is flipped
    assert _ports(pasted) == _ports(reference)


@pytest.mark.parametrize("flipped", [True, False])
def test_save_and_load_keep_the_flip(flipped):
    want, loaded = _round_trip(flipped)
    assert loaded.flipped is flipped
    assert _ports(loaded) == want
