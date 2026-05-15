"""
Regression tests for WorkspaceManager.resolve_params.

Commit 3e58bfb (safe_eval rollout) changed workspace.py line ~80 from
    val = safe_expr(value, variables=self.variables)
to
    val = safe_expr(value, variables=self.variables, allow_numpy=False)

Without allow_numpy=False, bare names like "sin" / "cos" were silently
rewritten to the numpy ufunc objects (np.sin, np.cos, ...) before blocks
ever saw the parameter.  Downstream effects:
  - WaveEquation1D: np.array(np.sin, dtype=float) -> TypeError
  - SystemCompiler: compile_expr("<ufunc 'cos'>") -> SafeEvalError
  - MathFunction: swallowed exception, returned 0.0 silently
"""

import pytest


@pytest.fixture(autouse=True)
def isolated_workspace():
    """Reset WorkspaceManager singleton around each test."""
    from lib.workspace import WorkspaceManager
    prev = WorkspaceManager._instance
    WorkspaceManager._instance = None
    yield
    WorkspaceManager._instance = prev


@pytest.mark.unit
class TestResolveParamsPreservesPatternStrings:
    """Regression: safe_expr(allow_numpy=True) silently rewrote block-parameter
    strings like 'sin', 'cos', 'sine' into numpy ufuncs, breaking PDE init
    conditions and MathFunction blocks."""

    def setup_method(self):
        from lib.workspace import WorkspaceManager
        self.wm = WorkspaceManager()
        self.wm.variables = {}

    @pytest.mark.parametrize("name", [
        # numpy ufuncs that must stay as strings (not ufunc objects)
        "sin", "cos", "tan", "exp", "log", "sqrt", "sign",
        # user-facing function-name strings used by MathFunction / PDE blocks
        "sine", "cosine", "gaussian", "square", "pi", "e",
        # Note: "abs" is intentionally NOT listed — safe_eval always resolves
        # "abs" to the Python builtin, which is correct pre-regression behaviour.
    ])
    def test_numpy_name_preserved_as_string(self, name):
        result = self.wm.resolve_params({"function": name})
        assert result["function"] == name, (
            f"Expected '{name}' preserved as string, got {result['function']!r}"
        )

    def test_workspace_arithmetic_still_resolves(self):
        self.wm.variables = {"K": 5.0}
        result = self.wm.resolve_params({"gain": "2*K"})
        assert result["gain"] == 10.0

    def test_workspace_list_expression_resolves(self):
        self.wm.variables = {"K": 3.0}
        result = self.wm.resolve_params({"numerator": "[K, K]"})
        assert result["numerator"] == [3.0, 3.0]

    def test_math_dot_prefix_still_works(self):
        self.wm.variables = {"K": 0.0}
        result = self.wm.resolve_params({"gain": "math.cos(K)"})
        assert result["gain"] == 1.0

    def test_direct_variable_name_still_resolves(self):
        """A param whose value IS a variable name should resolve to the variable's value."""
        self.wm.variables = {"K": 42.0}
        result = self.wm.resolve_params({"gain": "K"})
        assert result["gain"] == 42.0

    def test_unknown_bare_name_kept_as_string(self):
        """Non-numpy, non-variable bare name must stay as-is."""
        result = self.wm.resolve_params({"mode": "ramp"})
        assert result["mode"] == "ramp"

    def test_non_string_params_untouched(self):
        """Numeric and list params must pass through unchanged."""
        result = self.wm.resolve_params({"gain": 3.14, "coeff": [1, 2, 3]})
        assert result["gain"] == 3.14
        assert result["coeff"] == [1, 2, 3]
