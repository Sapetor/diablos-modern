"""
A flipped block must come back from paste and from a save/load round trip with
its ports mirrored, exactly as the canvas flip action leaves them.

Both paths used to set ``block.flipped`` *after* the constructor had already laid
the ports out unflipped, without re-running ``update_Block()``. The flag said
"flipped" while ``in_coords``/``out_coords`` still sat on the unflipped sides, so
wires attached to the wrong edge until the block was next moved. For load that
meant every saved diagram containing a flipped block reopened wrong.
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


def _gain(x=100, y=100):
    return DBlock(
        block_fn="Gain",
        sid=0,
        coords=QRect(x, y, 60, 40),
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


def _flipped_reference(x=100, y=100):
    """Port coordinates of a block flipped the way the canvas does it."""
    block = _gain(x, y)
    block.flipped = True
    block.update_Block()
    return block.in_coords[0], block.out_coords[0]


def _clipboard_entry(x, y, flipped):
    return {
        "block_fn": "Gain",
        "coords": QRect(x, y, 60, 40),
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


def _paste(flipped):
    dsim = SimpleNamespace(blocks_list=[], menu_blocks=[], colors=None)
    manager = ClipboardManager(SimpleNamespace(dsim=dsim))
    manager.clipboard_blocks = [_clipboard_entry(100, 100, flipped)]
    (pasted,) = manager._instantiate_pasted_blocks(QPoint(0, 0))
    return pasted


class TestPastedFlippedBlock:
    def test_pasted_flipped_block_has_mirrored_ports(self):
        pasted = _paste(flipped=True)
        assert pasted.flipped is True
        assert (pasted.in_coords[0], pasted.out_coords[0]) == _flipped_reference()

    def test_flipped_paste_actually_swaps_the_sides(self):
        """Guard against a reference that is itself unflipped."""
        plain = _paste(flipped=False)
        flipped = _paste(flipped=True)
        assert flipped.in_coords[0] == plain.out_coords[0]
        assert flipped.out_coords[0] == plain.in_coords[0]

    def test_unflipped_paste_is_unchanged(self):
        pasted = _paste(flipped=False)
        reference = _gain()
        assert pasted.flipped is False
        assert pasted.in_coords[0] == reference.in_coords[0]
        assert pasted.out_coords[0] == reference.out_coords[0]


class TestLoadedFlippedBlock:
    def _round_trip(self, flipped):
        from lib.lib import DSim
        from lib.services.file_service import FileService
        from lib.workspace import WorkspaceManager

        WorkspaceManager._instance = None
        source = DSim()
        menu_gain = next(b for b in source.model.menu_blocks if b.block_fn == "Gain")
        block = source.model.add_block(menu_gain, QPoint(100, 100))
        if flipped:
            block.flipped = True
            block.update_Block()
        want = (block.in_coords[0], block.out_coords[0])
        data = FileService(source.model).serialize()

        WorkspaceManager._instance = None
        target = DSim()
        FileService(target.model).apply_loaded_data(data)
        loaded = next(b for b in target.model.blocks_list if b.block_fn == "Gain")
        return want, loaded

    def test_reloaded_flipped_block_keeps_mirrored_ports(self):
        want, loaded = self._round_trip(flipped=True)
        assert loaded.flipped is True
        assert (loaded.in_coords[0], loaded.out_coords[0]) == want

    def test_reloaded_unflipped_block_is_unchanged(self):
        want, loaded = self._round_trip(flipped=False)
        assert loaded.flipped is False
        assert (loaded.in_coords[0], loaded.out_coords[0]) == want
