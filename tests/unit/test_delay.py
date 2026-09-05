"""Unit tests for the Delay block (z^-N).

The buffer used to be a plain list shifted with ``pop(0)`` on every time step --
O(N) per step for a block whose whole job is O(1). It is now a fixed-size ring
(``_buffer_`` plus a ``_head_`` index); these tests pin the FIFO semantics that
the rewrite has to preserve.
"""

import numpy as np
import pytest

from blocks.delay import DelayBlock


def _params(**ov):
    p = {"delay_steps": 1, "initial_value": 0.0, "_buffer_": [], "_init_start_": True}
    p.update(ov)
    return p


@pytest.mark.unit
class TestDelayBlock:
    def test_single_step_delay(self):
        block = DelayBlock()
        params = _params(delay_steps=1)

        assert np.isclose(block.execute(0.0, {0: 1.0}, params)[0][0], 0.0)
        assert np.isclose(block.execute(0.1, {0: 2.0}, params)[0][0], 1.0)
        assert np.isclose(block.execute(0.2, {0: 3.0}, params)[0][0], 2.0)

    def test_n_step_delay_reproduces_the_input_sequence(self):
        block = DelayBlock()
        n = 3
        params = _params(delay_steps=n, initial_value=-1.0)

        seq = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        out = [float(block.execute(i * 0.1, {0: v}, params)[0][0]) for i, v in enumerate(seq)]

        assert out[:n] == [-1.0] * n
        assert out[n:] == seq[: len(seq) - n]

    def test_buffer_length_stays_constant(self):
        """The ring must not grow: pop(0)/append kept it constant too."""
        block = DelayBlock()
        params = _params(delay_steps=4)

        for i in range(50):
            block.execute(i * 0.1, {0: float(i)}, params)

        assert len(params["_buffer_"]) == 4

    def test_head_wraps_around(self):
        block = DelayBlock()
        params = _params(delay_steps=3)

        for i in range(7):
            block.execute(i * 0.1, {0: float(i)}, params)

        assert params["_head_"] == 7 % 3

    def test_initial_values_are_distinct_objects(self):
        """A shared initial array would alias every slot of the ring."""
        block = DelayBlock()
        params = _params(delay_steps=3, initial_value=0.0)
        block.execute(0.0, {0: np.array([1.0])}, params)

        ids = {id(v) for v in params["_buffer_"]}
        assert len(ids) == 3

    def test_output_only_does_not_consume_the_input(self):
        block = DelayBlock()
        params = _params(delay_steps=2, initial_value=5.0)

        held = block.execute(0.0, {0: 1.0}, params, output_only=True)
        assert np.isclose(held[0][0], 5.0)
        # No sample was consumed, so the first real step still sees the fill value.
        assert np.isclose(block.execute(0.0, {0: 1.0}, params)[0][0], 5.0)

    def test_vector_signals_are_delayed_elementwise(self):
        block = DelayBlock()
        params = _params(delay_steps=1)

        block.execute(0.0, {0: np.array([1.0, 2.0])}, params)
        out = block.execute(0.1, {0: np.array([3.0, 4.0])}, params)
        np.testing.assert_allclose(out[0], [1.0, 2.0])

    def test_reinitializes_when_the_engine_re_arms_the_block(self):
        block = DelayBlock()
        params = _params(delay_steps=2, initial_value=0.0)

        block.execute(0.0, {0: 1.0}, params)
        block.execute(0.1, {0: 2.0}, params)

        params["_init_start_"] = True  # what reset_memblocks() does between runs
        assert np.isclose(block.execute(0.0, {0: 9.0}, params)[0][0], 0.0)

    def test_empty_restored_buffer_is_rebuilt_instead_of_raising(self):
        """A `_buffer_: []` restored from a save used to make pop(0) raise."""
        block = DelayBlock()
        params = _params(delay_steps=2, initial_value=7.0, _init_start_=False)

        result = block.execute(0.0, {0: 1.0}, params)
        assert np.isclose(result[0][0], 7.0)
        assert len(params["_buffer_"]) == 2
