"""Unit tests for the analysis *marker* blocks (BodeMag, BodePhase, RootLocus).

These blocks carry no simulation semantics: they mark a signal so the right-click
Analysis menu knows which dynamic block to linearize and plot. The only coverage
they had was ``assert block.execute(...) == {}`` in ``tests/test_remaining.py``,
which passes for a block that is broken in every other way.

What actually has to hold for a marker, and is asserted here:

* it is discoverable as a block (so it shows up in the palette / frozen build);
* it consumes exactly one input and produces no outputs;
* it still *requires* its input, so an unwired marker is flagged by validation;
* every declared parameter has a default, because nothing ever supplies one;
* ``execute()`` is inert -- no outputs, no error dict, no state written into
  ``params`` -- for any time, any input, and a missing input.
"""

import numpy as np
import pytest

from blocks.bodemag import BodeMagBlock
from blocks.bodephase import BodePhaseBlock
from blocks.rootlocus import RootLocusBlock

MARKERS = [
    pytest.param(BodeMagBlock, "BodeMag", id="bodemag"),
    pytest.param(BodePhaseBlock, "BodePhase", id="bodephase"),
    pytest.param(RootLocusBlock, "RootLocus", id="rootlocus"),
]


def _defaults(block):
    return {name: spec["default"] for name, spec in block.params.items()}


@pytest.mark.unit
@pytest.mark.parametrize("block_cls,expected_name", MARKERS)
class TestAnalysisMarkerContract:
    def test_identity(self, block_cls, expected_name):
        block = block_cls()
        assert block.block_name == expected_name
        assert block.category == "Analysis"

    def test_single_input_no_output(self, block_cls, expected_name):
        block = block_cls()
        assert len(block.inputs) == 1
        assert block.inputs[0]["name"] == "in"
        assert block.outputs == []

    def test_requires_its_input(self, block_cls, expected_name):
        """A marker with nothing wired to it has nothing to analyse, so
        validation should flag it rather than silently accept it."""
        assert block_cls().requires_inputs is True

    def test_zero_outputs_silences_the_dangling_output_warning(self, block_cls, expected_name):
        """``DiagramValidator._check_disconnected_outputs`` loops over
        ``range(block.out_ports)``, so declaring no output ports (rather than
        overriding ``requires_outputs``) is what keeps a marker from warning."""
        assert block_cls().outputs == []

    def test_documented(self, block_cls, expected_name):
        """The doc string is the block's only UI affordance -- it is what tells
        the user the plot comes from the right-click menu."""
        doc = block_cls().doc
        assert isinstance(doc, str) and doc.strip()
        assert "right-click" in doc.lower()

    def test_every_param_has_a_default(self, block_cls, expected_name):
        params = block_cls().params
        assert isinstance(params, dict)
        for name, spec in params.items():
            assert "default" in spec, f"{expected_name}.{name} has no default"

    def test_draw_icon_returns_a_painter_path(self, block_cls, expected_name, qapp):
        from PyQt5.QtCore import QRect
        from PyQt5.QtGui import QPainterPath

        path = block_cls().draw_icon(QRect(0, 0, 100, 60))
        assert isinstance(path, QPainterPath)
        assert not path.isEmpty()


@pytest.mark.unit
@pytest.mark.parametrize("block_cls,expected_name", MARKERS)
class TestAnalysisMarkerExecution:
    def test_inert_across_a_time_sweep(self, block_cls, expected_name):
        block = block_cls()
        params = _defaults(block)
        snapshot = dict(params)

        for step in range(10):
            result = block.execute(
                time=step * 0.1,
                inputs={0: np.array([float(step)])},
                params=params,
                dtime=0.1,
            )
            assert result == {}, f"{expected_name} produced output during simulation"
            assert "E" not in result, f"{expected_name} reported an error"

        assert params == snapshot, f"{expected_name} wrote state into params"

    def test_tolerates_a_missing_input(self, block_cls, expected_name):
        block = block_cls()
        assert block.execute(time=0.0, inputs={}, params=_defaults(block)) == {}

    def test_tolerates_a_vector_input(self, block_cls, expected_name):
        """Markers are often tapped off a MIMO signal."""
        block = block_cls()
        result = block.execute(
            time=0.0,
            inputs={0: np.array([1.0, 2.0, 3.0])},
            params=_defaults(block),
        )
        assert result == {}


@pytest.mark.unit
def test_markers_are_discoverable_blocks():
    """A marker missing from the loader silently disappears from the palette."""
    from lib.block_loader import load_blocks

    names = {cls().block_name for cls in load_blocks()}
    assert {"BodeMag", "BodePhase", "RootLocus", "Nyquist"} <= names
