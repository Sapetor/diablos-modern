import pytest
import numpy as np


@pytest.mark.unit
class TestExternalBlock:
    """Tests for External block (not fully implemented)."""

    def test_block_properties(self):
        from blocks.external import ExternalBlock
        block = ExternalBlock()
        assert block.block_name == 'External'
        assert 'filename' in block.params
        assert 'function' in block.params

    def test_default_filename(self):
        from blocks.external import ExternalBlock
        block = ExternalBlock()
        assert block.params['filename']['default'] == ''

    def test_execute_returns_error_when_file_not_loaded(self):
        """External block returns error when file_function not loaded."""
        from blocks.external import ExternalBlock
        block = ExternalBlock()
        params = {'filename': 'test.py'}
        result = block.execute(0.0, {0: np.array([1.0])}, params)
        # Now returns proper error dict instead of None
        assert result['E'] is True
        assert 'error' in result
        assert 'not loaded' in result['error']

    def test_execute_returns_error_when_no_filename(self):
        """External block returns error when no filename specified."""
        from blocks.external import ExternalBlock
        block = ExternalBlock()
        params = {'filename': ''}
        result = block.execute(0.0, {0: np.array([1.0])}, params)
        assert result['E'] is True
        assert 'No external file' in result['error']

    def test_has_input_output_definitions(self):
        from blocks.external import ExternalBlock
        block = ExternalBlock()
        assert len(block.inputs) == 1
        assert len(block.outputs) == 1

    def test_block_category(self):
        from blocks.external import ExternalBlock
        block = ExternalBlock()
        assert block.category == 'Other'

    def test_block_color(self):
        from blocks.external import ExternalBlock
        block = ExternalBlock()
        assert block.color == 'light_gray'

    def test_doc_string(self):
        from blocks.external import ExternalBlock
        block = ExternalBlock()
        assert 'External Function Block' in block.doc
        assert 'NOT FULLY IMPLEMENTED' in block.doc

    def test_input_port_name(self):
        from blocks.external import ExternalBlock
        block = ExternalBlock()
        assert block.inputs[0]['name'] == 'in'
        assert block.inputs[0]['type'] == 'any'

    def test_output_port_name(self):
        from blocks.external import ExternalBlock
        block = ExternalBlock()
        assert block.outputs[0]['name'] == 'out'
        assert block.outputs[0]['type'] == 'any'

    def test_execute_with_various_inputs(self):
        """Verify error behavior with different input types."""
        from blocks.external import ExternalBlock
        block = ExternalBlock()
        params = {'filename': 'script.py'}

        # Test with scalar input
        result = block.execute(0.0, {0: np.array([5.0])}, params)
        assert result['E'] is True

        # Test with vector input
        result = block.execute(0.0, {0: np.array([1.0, 2.0, 3.0])}, params)
        assert result['E'] is True

        # Test with multiple time steps
        result = block.execute(1.5, {0: np.array([42.0])}, params)
        assert result['E'] is True
