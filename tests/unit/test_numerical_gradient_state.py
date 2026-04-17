"""
Regression test for NumericalGradient block instance-state bug.

Per CLAUDE.md's critical rule, all persistent block state must live in `params`,
never on `self`. The previously-defined `self._cached_dimension = None` was
unused; this file verifies removal does not break the block's basic contract.
"""

import numpy as np
import pytest


@pytest.mark.unit
class TestNumericalGradientState:
    """Verify NumericalGradient works without the removed self._cached_dimension."""

    def test_no_instance_cache_attribute(self):
        """After the fix, the block should not carry `_cached_dimension` on self."""
        from blocks.optimization_primitives.numerical_gradient import (
            NumericalGradientBlock,
        )
        block = NumericalGradientBlock()
        assert not hasattr(block, '_cached_dimension'), (
            "Block must not hold mutable runtime state on self; use params."
        )

    def test_forward_difference_basic(self):
        """Forward-difference gradient matches (f_plus - f_center)/epsilon per dim."""
        from blocks.optimization_primitives.numerical_gradient import (
            NumericalGradientBlock,
        )
        block = NumericalGradientBlock()
        epsilon = 1e-3
        params = {'dimension': 2, 'epsilon': epsilon, 'method': 'forward'}
        # Simulate f(x)=1.0, f(x+eps*e0)=1.0+0.5*eps, f(x+eps*e1)=1.0-2.0*eps
        inputs = {
            0: np.array([1.0]),
            1: np.array([1.0 + 0.5 * epsilon]),
            2: np.array([1.0 - 2.0 * epsilon]),
        }
        result = block.execute(time=0.0, inputs=inputs, params=params)
        assert result['E'] is False
        np.testing.assert_allclose(result[0], np.array([0.5, -2.0]), rtol=1e-6)

    def test_central_difference_basic(self):
        """Central-difference gradient matches (f_plus - f_minus)/(2*epsilon)."""
        from blocks.optimization_primitives.numerical_gradient import (
            NumericalGradientBlock,
        )
        block = NumericalGradientBlock()
        epsilon = 1e-3
        params = {'dimension': 2, 'epsilon': epsilon, 'method': 'central'}
        inputs = {
            0: np.array([1.0]),
            1: np.array([1.0 + 0.5 * epsilon]),   # f_plus_0
            2: np.array([1.0 - 2.0 * epsilon]),   # f_plus_1
            3: np.array([1.0 - 0.5 * epsilon]),   # f_minus_0
            4: np.array([1.0 + 2.0 * epsilon]),   # f_minus_1
        }
        result = block.execute(time=0.0, inputs=inputs, params=params)
        assert result['E'] is False
        np.testing.assert_allclose(result[0], np.array([0.5, -2.0]), rtol=1e-6)

    def test_dimension_change_across_calls(self):
        """Block must correctly handle a dimension change between calls
        (the removed cache would have held a stale value across calls)."""
        from blocks.optimization_primitives.numerical_gradient import (
            NumericalGradientBlock,
        )
        block = NumericalGradientBlock()
        epsilon = 1e-3

        # First call: dimension=2
        params1 = {'dimension': 2, 'epsilon': epsilon, 'method': 'forward'}
        inputs1 = {
            0: np.array([0.0]),
            1: np.array([1.0 * epsilon]),
            2: np.array([2.0 * epsilon]),
        }
        result1 = block.execute(time=0.0, inputs=inputs1, params=params1)
        assert result1[0].shape == (2,)
        np.testing.assert_allclose(result1[0], [1.0, 2.0], rtol=1e-6)

        # Second call with a different dimension: block must adapt, not cache.
        params2 = {'dimension': 3, 'epsilon': epsilon, 'method': 'forward'}
        inputs2 = {
            0: np.array([0.0]),
            1: np.array([3.0 * epsilon]),
            2: np.array([4.0 * epsilon]),
            3: np.array([5.0 * epsilon]),
        }
        result2 = block.execute(time=0.01, inputs=inputs2, params=params2)
        assert result2[0].shape == (3,)
        np.testing.assert_allclose(result2[0], [3.0, 4.0, 5.0], rtol=1e-6)

    def test_execute_returns_error_dict_on_failure(self):
        """Malformed inputs still produce a proper error dict, not a crash."""
        from blocks.optimization_primitives.numerical_gradient import (
            NumericalGradientBlock,
        )
        block = NumericalGradientBlock()
        # epsilon='not a number' forces a float() conversion error inside execute.
        params = {'dimension': 2, 'epsilon': 'bad', 'method': 'forward'}
        result = block.execute(time=0.0, inputs={0: np.array([1.0])}, params=params)
        assert result['E'] is True
        assert result[0].shape == (2,)
