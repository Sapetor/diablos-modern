import pytest
import numpy as np

@pytest.mark.unit
class TestTerminatorBlock:
    """Tests for Terminator block."""

    def test_block_properties(self):
        from blocks.terminator import TerminatorBlock
        block = TerminatorBlock()
        assert block.block_name == 'Term'
        assert block.category == 'Sinks'

    def test_accepts_any_input(self):
        from blocks.terminator import TerminatorBlock
        block = TerminatorBlock()
        # Should not raise for any input
        result = block.execute(0.0, {0: np.array([1.0, 2.0, 3.0])}, {})
        assert result == {}

    def test_returns_empty_dict(self):
        from blocks.terminator import TerminatorBlock
        block = TerminatorBlock()
        result = block.execute(0.0, {0: np.array([5.0])}, {})
        assert result == {}
        assert isinstance(result, dict)

    def test_no_side_effects(self):
        from blocks.terminator import TerminatorBlock
        block = TerminatorBlock()
        params = {}
        block.execute(0.0, {0: np.array([1.0])}, params)
        block.execute(0.1, {0: np.array([2.0])}, params)
        # Params should remain empty - no state stored
        assert params == {}

    def test_has_one_input_no_outputs(self):
        from blocks.terminator import TerminatorBlock
        block = TerminatorBlock()
        assert len(block.inputs) == 1
        assert len(block.outputs) == 0

    def test_no_params_needed(self):
        from blocks.terminator import TerminatorBlock
        block = TerminatorBlock()
        assert block.params == {}

    def test_scalar_input(self):
        from blocks.terminator import TerminatorBlock
        block = TerminatorBlock()
        result = block.execute(0.0, {0: 42.0}, {})
        assert result == {}

    def test_missing_input(self):
        from blocks.terminator import TerminatorBlock
        block = TerminatorBlock()
        result = block.execute(0.0, {}, {})
        assert result == {}
