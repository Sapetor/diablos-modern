"""Regression: Selector comma-list indices survive workspace resolution.

WorkspaceManager.resolve_params runs every string param through safe_expr, which
evaluates a comma list like "0,2" into a Python tuple (0, 2) before the block or
the compiled kernel ever sees it. Before the fix that broke both paths: the
interpreter crashed in _parse_indices ('tuple' object has no attribute 'split')
and the compiled kernel did str((0, 2)) == "(0, 2)", failed to int-parse, and
silently selected index 0. Both paths now normalize the resolved value back to a
comma string via blocks.selector.normalize_indices_str.
"""

import numpy as np
import pytest

from blocks.selector import SelectorBlock, normalize_indices_str
from lib.engine.compiler_kernels import BuildContext, get_kernel_builder
from lib.workspace import WorkspaceManager


@pytest.mark.unit
class TestNormalizeIndicesStr:
    def test_string_passes_through(self):
        assert normalize_indices_str("0,2") == "0,2"
        assert normalize_indices_str("1:3") == "1:3"

    def test_tuple_and_list_become_comma_string(self):
        assert normalize_indices_str((0, 2)) == "0,2"
        assert normalize_indices_str([1, 3, 5]) == "1,3,5"
        assert normalize_indices_str(np.array([2, 4])) == "2,4"

    def test_scalar_int_becomes_string(self):
        assert normalize_indices_str(0) == "0"
        assert normalize_indices_str(3) == "3"


@pytest.mark.unit
class TestSelectorCommaListAfterResolve:
    def test_resolve_params_evaluates_comma_list_to_tuple(self):
        # Documents the trigger: this is why the block must normalize.
        resolved = WorkspaceManager().resolve_params({"indices": "0,2"})
        assert resolved["indices"] == (0, 2)

    def test_interpreter_handles_resolved_tuple(self):
        resolved = WorkspaceManager().resolve_params({"indices": "0,2"})
        block = SelectorBlock()
        result = block.execute(
            time=0.0, inputs={0: np.array([10.0, 11.0, 12.0, 13.0])}, params=resolved
        )
        assert np.allclose(result[0], [10.0, 12.0])

    def test_compiled_kernel_handles_resolved_tuple(self):
        resolved = WorkspaceManager().resolve_params({"indices": "0,2"})
        ctx = BuildContext(
            block=None,
            b_name="se0",
            fn="Selector",
            params=resolved,
            input_sources=["x"],
            deps={},
            state_map={},
            block_matrices={},
        )
        ex = get_kernel_builder("Selector")(ctx)
        sig = {"x": np.array([10.0, 11.0, 12.0, 13.0])}
        ex(0.0, None, None, sig)
        assert np.allclose(sig["se0"], [10.0, 12.0])
