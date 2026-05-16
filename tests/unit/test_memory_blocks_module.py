"""
Unit tests for lib/engine/memory_blocks.py — single source of truth for
memory-block classification used by both `SimulationEngine.identify_memory_blocks`
and `ValidationHelper.detect_algebraic_loops`.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from lib.engine.memory_blocks import (
    OUTPUT_ONLY_SAFE_BLOCK_FNS,
    is_memory_block,
    is_strictly_proper_tf,
    is_zero_D_statespace,
)


def _stub(block_fn: str, params: dict = None, exec_params: dict = None):
    return SimpleNamespace(
        name=f"{block_fn.lower()}_test",
        block_fn=block_fn,
        params=params or {},
        exec_params=exec_params if exec_params is not None else {},
    )


@pytest.mark.unit
class TestUnconditionalMemoryBlocks:

    @pytest.mark.parametrize("block_fn", sorted(OUTPUT_ONLY_SAFE_BLOCK_FNS))
    def test_all_listed_blocks_classify(self, block_fn):
        assert is_memory_block(_stub(block_fn)), (
            f"{block_fn} appears in OUTPUT_ONLY_SAFE_BLOCK_FNS but "
            f"is_memory_block() did not classify it."
        )

    def test_canonical_block_fns_present(self):
        """Sanity-check that the obvious memory blocks are in the set."""
        for name in ('Integrator', 'StateVariable', 'PID',
                     'TransportDelay', 'ZeroOrderHold'):
            assert name in OUTPUT_ONLY_SAFE_BLOCK_FNS

    def test_known_non_memory_blocks_excluded(self):
        for name in ('Gain', 'Sum', 'Constant', 'Saturation', 'Step'):
            assert name not in OUTPUT_ONLY_SAFE_BLOCK_FNS
            assert not is_memory_block(_stub(name)), (
                f"{name} should NOT be classified as a memory block."
            )


@pytest.mark.unit
class TestStrictlyProperTF:

    def test_strictly_proper_tranfn_classified(self):
        block = _stub("TranFn", exec_params={
            "numerator": [1.0], "denominator": [1.0, 1.0],
        })
        assert is_strictly_proper_tf(block)
        assert is_memory_block(block)

    def test_proper_tranfn_not_classified(self):
        block = _stub("TranFn", exec_params={
            "numerator": [1.0, 2.0], "denominator": [1.0, 1.0],
        })
        assert not is_strictly_proper_tf(block)
        assert not is_memory_block(block)

    def test_strictly_proper_discrete_tranfn_classified(self):
        block = _stub("DiscreteTranFn", exec_params={
            "numerator": [1.0], "denominator": [1.0, 1.0],
        })
        assert is_strictly_proper_tf(block)
        assert is_memory_block(block)

    def test_proper_discrete_tranfn_not_classified(self):
        block = _stub("DiscreteTranFn", exec_params={
            "numerator": [1.0, 0.5], "denominator": [1.0, 0.2],
        })
        assert not is_strictly_proper_tf(block)
        assert not is_memory_block(block)

    def test_falls_back_to_params_when_exec_params_empty(self):
        """Legacy callers (ValidationHelper) may pass raw params only."""
        block = _stub("TranFn", params={
            "numerator": [1.0], "denominator": [1.0, 1.0],
        }, exec_params={})
        assert is_strictly_proper_tf(block)


@pytest.mark.unit
class TestZeroDStateSpace:

    def test_zero_D_statespace_classified(self):
        block = _stub("StateSpace", exec_params={"D": np.array([[0.0]])})
        assert is_zero_D_statespace(block)
        assert is_memory_block(block)

    def test_nonzero_D_statespace_not_classified(self):
        block = _stub("StateSpace", exec_params={"D": np.array([[1.0]])})
        assert not is_zero_D_statespace(block)
        assert not is_memory_block(block)

    def test_zero_D_discrete_statespace_classified(self):
        block = _stub("DiscreteStateSpace", exec_params={"D": np.array([[0.0]])})
        assert is_zero_D_statespace(block)
        assert is_memory_block(block)

    def test_unresolved_D_string_returns_false_safely(self):
        """A workspace-variable string for D must not raise."""
        block = _stub("StateSpace", params={"D": "D_matrix"}, exec_params={})
        # Should not raise; conservative answer is False (treat as non-memory)
        assert is_zero_D_statespace(block) is False


@pytest.mark.unit
class TestSharedTaxonomyConsistency:
    """
    Make sure the unified module stays the single source of truth — both
    call sites import from lib.engine.memory_blocks and produce the same
    classification.
    """

    def test_engine_uses_shared_module(self):
        """SimulationEngine.identify_memory_blocks imports the shared helper."""
        import inspect
        from lib.engine import simulation_engine
        src = inspect.getsource(simulation_engine.SimulationEngine.identify_memory_blocks)
        assert "from lib.engine.memory_blocks import" in src or \
               "memory_blocks import" in src, (
            "identify_memory_blocks no longer references the shared "
            "memory_blocks module — the taxonomy has drifted again."
        )

    def test_improvements_uses_shared_module(self):
        """ValidationHelper.detect_algebraic_loops imports the shared helper."""
        import inspect
        from lib.improvements import ValidationHelper
        src = inspect.getsource(ValidationHelper.detect_algebraic_loops)
        assert "memory_blocks import" in src, (
            "detect_algebraic_loops no longer references the shared "
            "memory_blocks module — the taxonomy has drifted again."
        )

    def test_no_inline_MEMORY_BLOCK_TYPES_set_in_improvements(self):
        """The duplicate MEMORY_BLOCK_TYPES literal must be gone."""
        with open("lib/improvements.py") as f:
            content = f.read()
        assert "MEMORY_BLOCK_TYPES = {" not in content, (
            "lib/improvements.py still defines a local MEMORY_BLOCK_TYPES "
            "set — this duplicates the shared OUTPUT_ONLY_SAFE_BLOCK_FNS "
            "and is exactly the drift this refactor was meant to remove."
        )
