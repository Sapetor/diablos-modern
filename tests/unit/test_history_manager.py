from unittest.mock import MagicMock

import pytest
from PyQt5.QtCore import QRect

from blocks.transfer_function import TransferFunctionBlock
from lib.simulation.block import DBlock
from modern_ui.managers.history_manager import HistoryManager


@pytest.fixture
def canvas_with_tranfn(qapp):
    canvas = MagicMock()
    dsim = MagicMock()

    menu_entry = MagicMock()
    menu_entry.block_fn = 'TranFn'
    menu_entry.block_class = TransferFunctionBlock
    dsim.menu_blocks = [menu_entry]

    block = DBlock(
        block_fn='TranFn',
        sid=0,
        coords=QRect(100, 100, 110, 80),
        color='#4CAF50',
        in_ports=1,
        out_ports=1,
        b_type=2,
        io_edit=True,
        fn_name='transfer_function',
        params={'numerator': '[1]', 'denominator': '[1, 1]'},
        external=False,
        username='',
        block_class=TransferFunctionBlock,
        colors=None,
    )

    dsim.blocks_list = [block]
    dsim.line_list = []
    dsim.add_line = MagicMock()

    canvas.dsim = dsim
    canvas.update = MagicMock()
    canvas.clear_validation = MagicMock()

    return canvas


def test_restore_preserves_block_instance_for_nonstandard_block_fn(canvas_with_tranfn):
    history = HistoryManager(canvas_with_tranfn)

    captured = history._capture_state()
    assert history._restore_state(captured) is True

    restored = canvas_with_tranfn.dsim.blocks_list[0]
    assert isinstance(restored.block_instance, TransferFunctionBlock)
