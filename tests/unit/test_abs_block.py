import pytest
import numpy as np


@pytest.mark.unit
class TestAbsBlock:
    """Tests for Abs block."""

    def test_block_properties(self):
        from blocks.abs_block import AbsBlock
        block = AbsBlock()
        assert block.block_name == 'Abs'
        assert block.category == 'Math'

    def test_positive_input(self):
        from blocks.abs_block import AbsBlock
        block = AbsBlock()
        result = block.execute(0.0, {0: np.array([5.0])}, {})
        np.testing.assert_array_equal(result[0], np.array([5.0]))

    def test_negative_input(self):
        from blocks.abs_block import AbsBlock
        block = AbsBlock()
        result = block.execute(0.0, {0: np.array([-5.0])}, {})
        np.testing.assert_array_equal(result[0], np.array([5.0]))

    def test_zero_input(self):
        from blocks.abs_block import AbsBlock
        block = AbsBlock()
        result = block.execute(0.0, {0: np.array([0.0])}, {})
        np.testing.assert_array_equal(result[0], np.array([0.0]))

    def test_vector_input(self):
        from blocks.abs_block import AbsBlock
        block = AbsBlock()
        result = block.execute(0.0, {0: np.array([-3.0, 0.0, 4.0])}, {})
        np.testing.assert_array_equal(result[0], np.array([3.0, 0.0, 4.0]))

    def test_scalar_input_converted(self):
        from blocks.abs_block import AbsBlock
        block = AbsBlock()
        result = block.execute(0.0, {0: -7.0}, {})
        assert result[0][0] == 7.0

    def test_no_params_needed(self):
        from blocks.abs_block import AbsBlock
        block = AbsBlock()
        assert block.params == {}

    def test_has_one_input_one_output(self):
        from blocks.abs_block import AbsBlock
        block = AbsBlock()
        assert len(block.inputs) == 1
        assert len(block.outputs) == 1
