"""Unit tests for AgentScopeBlock.

AgentScope is a recorder-style sink: each ``execute()`` appends the flattened
input positions vector and the current time to per-run history buffers stored in
``params``, for post-simulation playback. It never mutates instance state, so
the accumulation must live entirely in the passed-in ``params`` dict.
"""

import numpy as np
import pytest

from blocks.agent_scope import AgentScopeBlock


@pytest.mark.unit
class TestAgentScopeBlock:
    def test_metadata(self):
        block = AgentScopeBlock()
        assert block.block_name == "AgentScope"
        assert block.category == "Sinks"
        assert block.outputs == []
        assert block.requires_outputs is False

    def test_init_start_creates_history(self):
        block = AgentScopeBlock()
        params = {"_init_start_": True}
        result = block.execute(0.0, {0: np.array([1.0, 2.0, 3.0, 4.0])}, params)

        assert result == {"E": False}
        # Init flag cleared and history buffers created on first call.
        assert params["_init_start_"] is False
        assert len(params["_pos_history_"]) == 1
        assert len(params["_time_history_"]) == 1
        assert np.allclose(params["_pos_history_"][0], [1.0, 2.0, 3.0, 4.0])
        assert params["_time_history_"][0] == 0.0

    def test_accumulates_across_calls(self):
        block = AgentScopeBlock()
        params = {"_init_start_": True}

        block.execute(0.0, {0: np.array([0.0, 0.0])}, params)
        block.execute(0.1, {0: np.array([1.0, 2.0])}, params)
        block.execute(0.2, {0: np.array([3.0, 4.0])}, params)

        assert params["_time_history_"] == [0.0, 0.1, 0.2]
        assert len(params["_pos_history_"]) == 3
        assert np.allclose(params["_pos_history_"][1], [1.0, 2.0])
        assert np.allclose(params["_pos_history_"][2], [3.0, 4.0])

    def test_stored_samples_are_copies(self):
        """Reusing the same input array between steps must not alias history."""
        block = AgentScopeBlock()
        params = {"_init_start_": True}
        buf = np.array([1.0, 1.0])

        block.execute(0.0, {0: buf}, params)
        buf[:] = [9.0, 9.0]  # mutate original after recording
        block.execute(0.1, {0: buf}, params)

        assert np.allclose(params["_pos_history_"][0], [1.0, 1.0])
        assert np.allclose(params["_pos_history_"][1], [9.0, 9.0])

    def test_missing_input_defaults_to_scalar_zero(self):
        block = AgentScopeBlock()
        params = {"_init_start_": True}
        result = block.execute(0.0, {}, params)

        assert result == {"E": False}
        assert np.allclose(params["_pos_history_"][0], [0.0])

    def test_multidimensional_input_is_flattened(self):
        block = AgentScopeBlock()
        params = {"_init_start_": True}
        block.execute(0.0, {0: np.array([[1.0, 2.0], [3.0, 4.0]])}, params)

        assert np.allclose(params["_pos_history_"][0], [1.0, 2.0, 3.0, 4.0])

    def test_state_isolated_between_blocks_via_params(self):
        """Two runs with independent params must not share history (no self state)."""
        block = AgentScopeBlock()
        params_a = {"_init_start_": True}
        params_b = {"_init_start_": True}

        block.execute(0.0, {0: np.array([1.0, 1.0])}, params_a)
        block.execute(0.0, {0: np.array([2.0, 2.0])}, params_b)

        assert len(params_a["_pos_history_"]) == 1
        assert len(params_b["_pos_history_"]) == 1
        assert np.allclose(params_a["_pos_history_"][0], [1.0, 1.0])
        assert np.allclose(params_b["_pos_history_"][0], [2.0, 2.0])
