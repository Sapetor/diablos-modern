"""Unit tests for BodePhaseBlock.

BodePhase is an analysis block: it produces its plot via a right-click menu and
does no signal processing during simulation, so ``execute()`` must always
return an empty dict regardless of input. Mirrors the BodeMag coverage in
``tests/test_remaining.py``.
"""

import numpy as np
import pytest

from blocks.bodephase import BodePhaseBlock


@pytest.mark.unit
class TestBodePhaseBlock:
    def test_metadata(self):
        block = BodePhaseBlock()
        assert block.block_name == "BodePhase"
        assert block.category == "Analysis"
        # Analysis sink: consumes one input, produces no outputs.
        assert len(block.inputs) == 1
        assert block.outputs == []

    def test_execute_returns_empty(self):
        block = BodePhaseBlock()
        params = {'_init_start_': True}
        result = block.execute(0.0, {0: np.array([1.0])}, params)
        assert result == {}

    def test_execute_ignores_missing_input(self):
        block = BodePhaseBlock()
        result = block.execute(1.0, {}, {'_init_start_': True})
        assert result == {}

    def test_execute_stateless_across_calls(self):
        block = BodePhaseBlock()
        params = {'_init_start_': True}
        for t in (0.0, 0.1, 0.2):
            assert block.execute(t, {0: np.array([t])}, params) == {}

    def test_draw_icon_returns_path(self):
        from PyQt5.QtGui import QPainterPath
        block = BodePhaseBlock()
        path = block.draw_icon(None)
        assert isinstance(path, QPainterPath)
        assert not path.isEmpty()
