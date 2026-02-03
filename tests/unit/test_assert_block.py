import pytest
import numpy as np


@pytest.mark.unit
class TestAssertBlock:
    """Tests for Assert block."""

    def test_block_properties(self):
        """Test basic block properties."""
        from blocks.assert_block import AssertBlock
        block = AssertBlock()
        assert block.block_name == 'Assert'
        assert block.category == 'Sinks'
        assert block.color == 'red'
        assert 'condition' in block.params
        assert 'message' in block.params
        assert 'enabled' in block.params
        assert block.requires_outputs is False

    def test_greater_than_zero_pass(self):
        """Test >0 condition passes for positive value."""
        from blocks.assert_block import AssertBlock
        block = AssertBlock()
        params = {'condition': '>0', 'enabled': True}
        result = block.execute(0.0, {0: np.array([1.0])}, params)
        assert result.get('E') is not True
        assert 'error' not in result

    def test_greater_than_zero_fail(self):
        """Test >0 condition fails for negative value."""
        from blocks.assert_block import AssertBlock
        block = AssertBlock()
        params = {'condition': '>0', 'enabled': True, 'message': 'Test fail'}
        result = block.execute(0.0, {0: np.array([-1.0])}, params)
        assert result.get('E') is True
        assert 'error' in result
        assert 'Test fail' in result['error']
        assert 'value=-1.0' in result['error']

    def test_greater_than_zero_fail_at_zero(self):
        """Test >0 condition fails at exactly zero."""
        from blocks.assert_block import AssertBlock
        block = AssertBlock()
        params = {'condition': '>0', 'enabled': True}
        result = block.execute(0.0, {0: np.array([0.0])}, params)
        assert result.get('E') is True

    def test_less_than_zero_pass(self):
        """Test <0 condition passes for negative value."""
        from blocks.assert_block import AssertBlock
        block = AssertBlock()
        params = {'condition': '<0', 'enabled': True}
        result = block.execute(0.0, {0: np.array([-5.0])}, params)
        assert result.get('E') is not True

    def test_less_than_zero_fail(self):
        """Test <0 condition fails for positive value."""
        from blocks.assert_block import AssertBlock
        block = AssertBlock()
        params = {'condition': '<0', 'enabled': True}
        result = block.execute(0.0, {0: np.array([5.0])}, params)
        assert result.get('E') is True

    def test_greater_equal_zero_pass(self):
        """Test >=0 condition passes at zero."""
        from blocks.assert_block import AssertBlock
        block = AssertBlock()
        params = {'condition': '>=0', 'enabled': True}
        result = block.execute(0.0, {0: np.array([0.0])}, params)
        assert result.get('E') is not True

    def test_greater_equal_zero_fail(self):
        """Test >=0 condition fails for negative value."""
        from blocks.assert_block import AssertBlock
        block = AssertBlock()
        params = {'condition': '>=0', 'enabled': True}
        result = block.execute(0.0, {0: np.array([-0.1])}, params)
        assert result.get('E') is True

    def test_less_equal_zero_pass(self):
        """Test <=0 condition passes at zero."""
        from blocks.assert_block import AssertBlock
        block = AssertBlock()
        params = {'condition': '<=0', 'enabled': True}
        result = block.execute(0.0, {0: np.array([0.0])}, params)
        assert result.get('E') is not True

    def test_less_equal_zero_fail(self):
        """Test <=0 condition fails for positive value."""
        from blocks.assert_block import AssertBlock
        block = AssertBlock()
        params = {'condition': '<=0', 'enabled': True}
        result = block.execute(0.0, {0: np.array([0.1])}, params)
        assert result.get('E') is True

    def test_equal_zero_pass(self):
        """Test ==0 condition passes for zero (within tolerance)."""
        from blocks.assert_block import AssertBlock
        block = AssertBlock()
        params = {'condition': '==0', 'enabled': True}
        result = block.execute(0.0, {0: np.array([0.0])}, params)
        assert result.get('E') is not True

    def test_equal_zero_pass_near_zero(self):
        """Test ==0 condition passes for values near zero (1e-11 < 1e-10)."""
        from blocks.assert_block import AssertBlock
        block = AssertBlock()
        params = {'condition': '==0', 'enabled': True}
        result = block.execute(0.0, {0: np.array([1e-11])}, params)
        assert result.get('E') is not True

    def test_equal_zero_fail(self):
        """Test ==0 condition fails for non-zero value."""
        from blocks.assert_block import AssertBlock
        block = AssertBlock()
        params = {'condition': '==0', 'enabled': True}
        result = block.execute(0.0, {0: np.array([0.1])}, params)
        assert result.get('E') is True

    def test_not_equal_zero_pass(self):
        """Test !=0 condition passes for non-zero value."""
        from blocks.assert_block import AssertBlock
        block = AssertBlock()
        params = {'condition': '!=0', 'enabled': True}
        result = block.execute(0.0, {0: np.array([1.0])}, params)
        assert result.get('E') is not True

    def test_not_equal_zero_fail(self):
        """Test !=0 condition fails for zero."""
        from blocks.assert_block import AssertBlock
        block = AssertBlock()
        params = {'condition': '!=0', 'enabled': True}
        result = block.execute(0.0, {0: np.array([0.0])}, params)
        assert result.get('E') is True

    def test_not_equal_zero_fail_near_zero(self):
        """Test !=0 condition fails for values near zero."""
        from blocks.assert_block import AssertBlock
        block = AssertBlock()
        params = {'condition': '!=0', 'enabled': True}
        result = block.execute(0.0, {0: np.array([1e-11])}, params)
        assert result.get('E') is True

    def test_finite_pass(self):
        """Test 'finite' condition passes for normal values."""
        from blocks.assert_block import AssertBlock
        block = AssertBlock()
        params = {'condition': 'finite', 'enabled': True}
        result = block.execute(0.0, {0: np.array([42.0])}, params)
        assert result.get('E') is not True

    def test_finite_fail_with_nan(self):
        """Test 'finite' condition fails with NaN input."""
        from blocks.assert_block import AssertBlock
        block = AssertBlock()
        params = {'condition': 'finite', 'enabled': True, 'message': 'NaN detected'}
        result = block.execute(0.0, {0: np.array([np.nan])}, params)
        assert result.get('E') is True
        assert 'error' in result
        assert 'NaN detected' in result['error']

    def test_finite_fail_with_positive_inf(self):
        """Test 'finite' condition fails with +Inf input."""
        from blocks.assert_block import AssertBlock
        block = AssertBlock()
        params = {'condition': 'finite', 'enabled': True, 'message': 'Infinity detected'}
        result = block.execute(0.0, {0: np.array([np.inf])}, params)
        assert result.get('E') is True
        assert 'Infinity detected' in result['error']

    def test_finite_fail_with_negative_inf(self):
        """Test 'finite' condition fails with -Inf input."""
        from blocks.assert_block import AssertBlock
        block = AssertBlock()
        params = {'condition': 'finite', 'enabled': True}
        result = block.execute(0.0, {0: np.array([-np.inf])}, params)
        assert result.get('E') is True

    def test_disabled_assertion_no_check(self):
        """Test that with enabled=False, assertion doesn't fail."""
        from blocks.assert_block import AssertBlock
        block = AssertBlock()
        params = {'condition': '>0', 'enabled': False}
        # Should pass even with negative value
        result = block.execute(0.0, {0: np.array([-100.0])}, params)
        assert result.get('E') is not True
        assert 'error' not in result

    def test_disabled_assertion_nan(self):
        """Test that disabled assertion ignores NaN."""
        from blocks.assert_block import AssertBlock
        block = AssertBlock()
        params = {'condition': 'finite', 'enabled': False}
        result = block.execute(0.0, {0: np.array([np.nan])}, params)
        assert result.get('E') is not True

    def test_vector_input_all_pass(self):
        """Test vector input where all elements satisfy condition."""
        from blocks.assert_block import AssertBlock
        block = AssertBlock()
        params = {'condition': '>0', 'enabled': True}
        result = block.execute(0.0, {0: np.array([1.0, 2.0, 3.0, 4.0])}, params)
        assert result.get('E') is not True

    def test_vector_input_one_fails(self):
        """Test vector input where one element violates condition."""
        from blocks.assert_block import AssertBlock
        block = AssertBlock()
        params = {'condition': '>0', 'enabled': True, 'message': 'Vector violation'}
        result = block.execute(0.0, {0: np.array([1.0, 2.0, -0.5, 4.0])}, params)
        assert result.get('E') is True
        assert 'Vector violation' in result['error']
        assert 'value=-0.5' in result['error']

    def test_vector_input_all_fail(self):
        """Test vector input where all elements violate condition."""
        from blocks.assert_block import AssertBlock
        block = AssertBlock()
        params = {'condition': '>0', 'enabled': True}
        result = block.execute(0.0, {0: np.array([-1.0, -2.0, -3.0])}, params)
        assert result.get('E') is True

    def test_2d_array_input(self):
        """Test 2D array input is properly flattened."""
        from blocks.assert_block import AssertBlock
        block = AssertBlock()
        params = {'condition': '>=0', 'enabled': True}
        # All positive
        result = block.execute(0.0, {0: np.array([[1.0, 2.0], [3.0, 4.0]])}, params)
        assert result.get('E') is not True

        # One negative in 2D array
        params = {'condition': '>=0', 'enabled': True}
        result = block.execute(0.0, {0: np.array([[1.0, -2.0], [3.0, 4.0]])}, params)
        assert result.get('E') is True

    def test_unknown_condition_passes(self):
        """Test that unknown condition type passes without error."""
        from blocks.assert_block import AssertBlock
        block = AssertBlock()
        params = {'condition': 'unknown_condition', 'enabled': True}
        result = block.execute(0.0, {0: np.array([42.0])}, params)
        assert result.get('E') is not True

    def test_error_message_includes_time(self):
        """Test that error message includes time of failure."""
        from blocks.assert_block import AssertBlock
        block = AssertBlock()
        params = {'condition': '>0', 'enabled': True}
        result = block.execute(2.5, {0: np.array([-1.0])}, params)
        assert result.get('E') is True
        assert 'time=2.5' in result['error']

    def test_error_message_includes_condition(self):
        """Test that error message includes the condition type."""
        from blocks.assert_block import AssertBlock
        block = AssertBlock()
        params = {'condition': '>=0', 'enabled': True}
        result = block.execute(0.0, {0: np.array([-1.0])}, params)
        assert result.get('E') is True
        assert 'condition=>=0' in result['error']

    def test_scalar_input(self):
        """Test scalar input (not array)."""
        from blocks.assert_block import AssertBlock
        block = AssertBlock()
        params = {'condition': '>0', 'enabled': True}
        result = block.execute(0.0, {0: 5.0}, params)
        assert result.get('E') is not True

    def test_missing_input_defaults_to_zero(self):
        """Test missing input defaults to 0."""
        from blocks.assert_block import AssertBlock
        block = AssertBlock()
        params = {'condition': '==0', 'enabled': True}
        result = block.execute(0.0, {}, params)
        assert result.get('E') is not True

    def test_default_params(self):
        """Test with default parameters."""
        from blocks.assert_block import AssertBlock
        block = AssertBlock()
        # Default condition is '>0', enabled is True
        result = block.execute(0.0, {0: np.array([1.0])}, {})
        assert result.get('E') is not True

        # Should fail with negative value using defaults
        result = block.execute(0.0, {0: np.array([-1.0])}, {})
        assert result.get('E') is True
        assert 'Assertion failed' in result['error']  # Default message
