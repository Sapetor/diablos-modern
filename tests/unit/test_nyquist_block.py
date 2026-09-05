"""Unit tests for the Nyquist analysis-marker block (blocks/nyquist.py).

``NyquistBlock`` is a marker: it is dropped on the canvas, wired to a dynamic
block, and the plot is produced from the right-click menu. During simulation it
must do nothing at all -- and, crucially, must keep doing nothing when it is
handed inputs it never asked for, because the engine executes every block in the
diagram. Before this file the block had no test.
"""

import numpy as np
import pytest

from blocks.nyquist import NyquistBlock


@pytest.mark.unit
class TestNyquistBlockContract:
    def test_identity(self):
        block = NyquistBlock()
        assert block.block_name == "Nyquist"
        assert block.category == "Analysis"

    def test_ports(self):
        block = NyquistBlock()
        assert len(block.inputs) == 1
        assert block.inputs[0]["name"] == "in"
        # A marker consumes a signal and emits none.
        assert block.outputs == []

    def test_requires_its_input(self):
        """A marker with nothing wired to it has nothing to analyse, so
        validation should flag it rather than silently accept it."""
        assert NyquistBlock().requires_inputs is True

    def test_params_are_declarative_only(self):
        """Any parameter it declares must carry a usable default -- the block is
        configured entirely from the palette, never from an input signal."""
        params = NyquistBlock().params
        assert isinstance(params, dict)
        for name, spec in params.items():
            assert "default" in spec, name
        if "_init_start_" in params:
            assert params["_init_start_"]["default"] is True


@pytest.mark.unit
class TestNyquistBlockExecution:
    def test_execute_returns_nothing(self):
        block = NyquistBlock()
        params = {k: v["default"] for k, v in block.params.items()}

        assert block.execute(time=0.0, inputs={0: np.array([1.0])}, params=params) == {}

    def test_execute_is_inert_across_a_time_sweep(self):
        """No accumulation, no error dict, no state -- at any time or input."""
        block = NyquistBlock()
        params = {k: v["default"] for k, v in block.params.items()}
        snapshot = dict(params)

        for step in range(10):
            result = block.execute(
                time=step * 0.1,
                inputs={0: np.array([float(step)])},
                params=params,
                dtime=0.1,
            )
            assert result == {}
            assert "E" not in result

        assert params == snapshot, "marker block must not accumulate state in params"

    def test_execute_tolerates_a_missing_input(self):
        """Ports can be left unconnected while a diagram is being built."""
        block = NyquistBlock()
        params = {k: v["default"] for k, v in block.params.items()}

        assert block.execute(time=0.0, inputs={}, params=params) == {}

    def test_draw_icon_returns_a_painter_path(self, qapp):
        from PyQt5.QtCore import QRect
        from PyQt5.QtGui import QPainterPath

        path = NyquistBlock().draw_icon(QRect(0, 0, 100, 60))
        assert isinstance(path, QPainterPath)
        assert not path.isEmpty()
