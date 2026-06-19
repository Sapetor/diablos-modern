"""
safe_eval.py — Hardened expression evaluator for DiaBloS parameter strings.

Replaces raw eval() calls with an AST-walking interpreter that enforces a
strict whitelist of allowed node types. No eval(), no exec() anywhere.
"""

import ast
import logging
import math
import types
from typing import Any, Dict, Optional

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------

class SafeEvalError(ValueError):
    """Raised when safe_eval refuses to evaluate an expression."""


# ---------------------------------------------------------------------------
# Allowlist construction
# ---------------------------------------------------------------------------

_MAX_LEN = 4096

# Maximum number of elements an allocating numpy constructor may produce.
# Guards against memory-exhaustion DoS via user-controlled parameter strings
# (e.g. np.zeros(10**9) or np.ones((100000, 100000))), which would otherwise
# bypass the Pow exponent cap since the large value is a Call argument.
_MAX_ALLOC_ELEMENTS = 10_000_000


def _shape_element_count(shape: Any) -> Optional[int]:
    """
    Return the total element count implied by a numpy ``shape`` argument
    (an int or a tuple/list of ints), or None if it can't be determined.
    """
    try:
        if isinstance(shape, (int, np.integer)) and not isinstance(shape, bool):
            return int(shape)
        if isinstance(shape, (tuple, list)):
            total = 1
            for dim in shape:
                if not isinstance(dim, (int, np.integer)) or isinstance(dim, bool):
                    return None
                total *= int(dim)
            return total
    except Exception:
        return None
    return None


def _guard_alloc(count: Optional[int]) -> None:
    """Raise SafeEvalError if a requested element count exceeds the cap."""
    if count is not None and count > _MAX_ALLOC_ELEMENTS:
        raise SafeEvalError(
            f"Array of {count} elements exceeds the maximum allowed size "
            f"({_MAX_ALLOC_ELEMENTS}) to prevent memory exhaustion"
        )


def _make_np_namespace() -> types.SimpleNamespace:
    """Build a frozen namespace with only the allowed numpy names."""
    if not _HAS_NUMPY:
        return types.SimpleNamespace()

    # Bounded wrappers around allocating constructors. They accept the same
    # positional signatures numpy exposes (safe_eval forbids keyword args), and
    # raise SafeEvalError before delegating if the request is too large.
    def _bounded_shape_ctor(fn):
        def wrapper(shape, *rest):
            _guard_alloc(_shape_element_count(shape))
            return fn(shape, *rest)
        return wrapper

    def _bounded_eye(N, *rest):
        M = rest[0] if rest else N
        count = None
        if (isinstance(N, (int, np.integer)) and not isinstance(N, bool)
                and isinstance(M, (int, np.integer)) and not isinstance(M, bool)):
            count = int(N) * int(M)
        _guard_alloc(count)
        return np.eye(N, *rest)

    def _bounded_identity(n, *rest):
        count = None
        if isinstance(n, (int, np.integer)) and not isinstance(n, bool):
            count = int(n) * int(n)
        _guard_alloc(count)
        return np.identity(n, *rest)

    def _bounded_arange(*args):
        # arange([start,] stop[, step])
        count = None
        try:
            if len(args) == 1:
                start, stop, step = 0, args[0], 1
            elif len(args) == 2:
                start, stop, step = args[0], args[1], 1
            elif len(args) >= 3:
                start, stop, step = args[0], args[1], args[2]
            else:
                start = stop = step = None
            if step not in (None, 0) and all(
                isinstance(v, (int, float, np.integer, np.floating))
                for v in (start, stop, step)
            ):
                count = max(int(math.ceil((float(stop) - float(start)) / float(step))), 0)
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            count = None
        # Guard outside the try so SafeEvalError (a ValueError subclass) is not
        # swallowed by the except clause above.
        _guard_alloc(count)
        return np.arange(*args)

    def _bounded_linspace(start, stop, *rest):
        num = rest[0] if rest else 50
        if isinstance(num, (int, np.integer)) and not isinstance(num, bool):
            _guard_alloc(int(num))
        return np.linspace(start, stop, *rest)

    _np_names = {
        # Constants
        "pi": np.pi,
        "e": np.e,
        "inf": np.inf,
        "nan": np.nan,
        # Unary
        "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
        "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
        "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
        "exp": np.exp, "expm1": np.expm1,
        "log": np.log, "log2": np.log2, "log10": np.log10, "log1p": np.log1p,
        "sqrt": np.sqrt, "square": np.square,
        "abs": np.abs, "sign": np.sign,
        "floor": np.floor, "ceil": np.ceil, "round": np.round,
        "trunc": np.trunc,
        "real": np.real, "imag": np.imag, "conjugate": np.conjugate,
        "isnan": np.isnan, "isfinite": np.isfinite, "isinf": np.isinf,
        # Binary / variadic
        "arctan2": np.arctan2, "atan2": np.arctan2,
        "power": np.power, "mod": np.mod, "fmod": np.fmod,
        "minimum": np.minimum, "maximum": np.maximum,
        "min": np.min, "max": np.max,
        "sum": np.sum, "prod": np.prod, "mean": np.mean,
        "std": np.std, "var": np.var,
        "dot": np.dot, "cross": np.cross,
        "where": np.where, "clip": np.clip, "hypot": np.hypot,
        # Array constructors (allocating ones wrapped to cap element count)
        "array": np.array, "asarray": np.asarray,
        "atleast_1d": np.atleast_1d, "atleast_2d": np.atleast_2d,
        "zeros": _bounded_shape_ctor(np.zeros),
        "ones": _bounded_shape_ctor(np.ones),
        "eye": _bounded_eye,
        "identity": _bounded_identity,
        "full": _bounded_shape_ctor(np.full),
        "linspace": _bounded_linspace, "arange": _bounded_arange,
        "diag": np.diag, "vstack": np.vstack, "hstack": np.hstack,
        "concatenate": np.concatenate, "reshape": np.reshape,
        "transpose": np.transpose,
    }
    return types.SimpleNamespace(**_np_names)


def _make_math_namespace() -> types.SimpleNamespace:
    """Build a frozen namespace with allowed math/numpy names for `math.` prefix."""
    m: Dict[str, Any] = {}
    _math_names = [
        "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
        "sinh", "cosh", "tanh",
        "exp", "expm1", "log", "log2", "log10", "log1p",
        "sqrt", "floor", "ceil", "trunc", "fmod", "hypot",
        "pi", "e", "inf", "nan",
        "isnan", "isfinite", "isinf",
        "pow", "fabs",
    ]
    for name in _math_names:
        if hasattr(math, name):
            m[name] = getattr(math, name)
        elif _HAS_NUMPY and hasattr(np, name):
            m[name] = getattr(np, name)
    return types.SimpleNamespace(**m)


_NP_NS = _make_np_namespace()
_MATH_NS = _make_math_namespace()

# Set of allowed attr names on np / math namespaces
_NP_ALLOWLIST = frozenset(vars(_NP_NS).keys())
_MATH_ALLOWLIST = frozenset(vars(_MATH_NS).keys())

# Always-bound builtins (present even when allow_numpy=False)
_BUILTINS: Dict[str, Any] = {
    "abs": abs, "min": min, "max": max, "sum": sum,
    "len": len, "round": round,
    "int": int, "float": float, "bool": bool, "complex": complex,
    "list": list, "tuple": tuple, "dict": dict, "range": range,
    "True": True, "False": False, "None": None,
}


def _build_env(variables: Optional[Dict[str, Any]], allow_numpy: bool) -> Dict[str, Any]:
    """Merge builtins, optional numpy, and user variables into one lookup dict."""
    env: Dict[str, Any] = dict(_BUILTINS)
    if allow_numpy and _HAS_NUMPY:
        # Expose all np namespace names as bare names too
        env.update(vars(_NP_NS))
        env["np"] = _NP_NS
        env["math"] = _MATH_NS
    else:
        # Do NOT bind 'np' — so np.sin raises "Name 'np' not defined"
        env["math"] = _MATH_NS
    if variables:
        env.update(variables)
    return env


# ---------------------------------------------------------------------------
# AST walker
# ---------------------------------------------------------------------------

# Operator maps
_UNARY_OPS = {
    ast.UAdd: lambda x: +x,
    ast.USub: lambda x: -x,
    ast.Not:  lambda x: not x,
    ast.Invert: lambda x: ~x,
}

_BIN_OPS = {
    ast.Add:      lambda a, b: a + b,
    ast.Sub:      lambda a, b: a - b,
    ast.Mult:     lambda a, b: a * b,
    ast.Div:      lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod:      lambda a, b: a % b,
    ast.Pow:      lambda a, b: a ** b,
    ast.MatMult:  lambda a, b: a @ b,
}

_BOOL_OPS = {
    ast.And: all,
    ast.Or:  any,
}

_CMP_OPS = {
    ast.Eq:    lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
    ast.Lt:    lambda a, b: a < b,
    ast.LtE:   lambda a, b: a <= b,
    ast.Gt:    lambda a, b: a > b,
    ast.GtE:   lambda a, b: a >= b,
    ast.Is:    lambda a, b: a is b,
    ast.IsNot: lambda a, b: a is not b,
    ast.In:    lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}

# Nodes that are always forbidden
_FORBIDDEN_NODE_NAMES = frozenset({
    "Lambda", "ListComp", "SetComp", "DictComp", "GeneratorExp",
    "Yield", "YieldFrom", "Await",
    "Import", "ImportFrom",
    "FunctionDef", "AsyncFunctionDef", "ClassDef",
    "Assign", "AugAssign", "AnnAssign",
    "Global", "Nonlocal",
    "Raise", "Try", "TryStar", "Delete", "With", "AsyncWith",
    "For", "AsyncFor", "While",
    "If",                    # statement If (not IfExp)
    "JoinedStr", "FormattedValue",
    "NamedExpr",             # walrus :=
    "Starred",
    "AsyncFunctionDef",
    "ExceptHandler",
})


class _Walker:
    """Recursive AST evaluator — no eval/exec used."""

    def __init__(self, env: Dict[str, Any]):
        self._env = env

    def visit(self, node: ast.AST) -> Any:
        name = type(node).__name__

        # Blanket-forbid entire node classes
        if name in _FORBIDDEN_NODE_NAMES:
            raise SafeEvalError(f"Forbidden AST node: {name}")

        method = getattr(self, f"_visit_{name}", None)
        if method is None:
            raise SafeEvalError(f"Unsupported AST node: {name}")
        return method(node)

    # --- Literals ----------------------------------------------------------

    def _visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def _visit_Constant(self, node: ast.Constant) -> Any:
        return node.value

    # Python 3.7 compat (Num/Str/NameConstant were merged into Constant in 3.8)
    def _visit_Num(self, node) -> Any:  # type: ignore[override]
        return node.n

    def _visit_Str(self, node) -> Any:  # type: ignore[override]
        return node.s

    def _visit_NameConstant(self, node) -> Any:  # type: ignore[override]
        return node.value

    # --- Collections -------------------------------------------------------

    def _visit_Tuple(self, node: ast.Tuple) -> Any:
        return tuple(self.visit(e) for e in node.elts)

    def _visit_List(self, node: ast.List) -> Any:
        return [self.visit(e) for e in node.elts]

    def _visit_Dict(self, node: ast.Dict) -> Any:
        return {self.visit(k): self.visit(v) for k, v in zip(node.keys, node.values)}

    def _visit_Set(self, node: ast.Set) -> Any:
        return {self.visit(e) for e in node.elts}

    # --- Names -------------------------------------------------------------

    def _visit_Name(self, node: ast.Name) -> Any:
        if not isinstance(node.ctx, ast.Load):
            raise SafeEvalError(f"Non-load Name context for '{node.id}'")
        try:
            return self._env[node.id]
        except KeyError:
            raise SafeEvalError(f"Name '{node.id}' is not defined")

    def _visit_Load(self, node: ast.Load) -> Any:  # pragma: no cover
        return None  # context node, never visited directly

    # --- Attribute ---------------------------------------------------------

    def _visit_Attribute(self, node: ast.Attribute) -> Any:
        # ONLY allow <Name>.attr where Name is 'np' or 'math'
        if not isinstance(node.value, ast.Name):
            raise SafeEvalError(
                "Attribute access only allowed on bare 'np' or 'math' names"
            )
        ns_name = node.value.id
        attr = node.attr

        if ns_name == "np":
            # np must be in env (i.e. allow_numpy=True)
            if "np" not in self._env:
                raise SafeEvalError("Name 'np' is not defined (numpy not enabled)")
            if attr not in _NP_ALLOWLIST:
                raise SafeEvalError(f"np.{attr} is not in the numpy allowlist")
            return getattr(_NP_NS, attr)
        elif ns_name == "math":
            if attr not in _MATH_ALLOWLIST:
                raise SafeEvalError(f"math.{attr} is not in the math allowlist")
            return getattr(_MATH_NS, attr)
        else:
            raise SafeEvalError(
                f"Attribute access not allowed on '{ns_name}' (only np/math)"
            )

    # --- Operators ---------------------------------------------------------

    def _visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        op_type = type(node.op)
        fn = _UNARY_OPS.get(op_type)
        if fn is None:
            raise SafeEvalError(f"Unsupported unary operator: {op_type.__name__}")
        return fn(self.visit(node.operand))

    def _visit_BinOp(self, node: ast.BinOp) -> Any:
        op_type = type(node.op)
        fn = _BIN_OPS.get(op_type)
        if fn is None:
            raise SafeEvalError(f"Unsupported binary operator: {op_type.__name__}")
        left = self.visit(node.left)
        right = self.visit(node.right)
        # Guard against astronomically large integer exponents (DoS / memory bomb)
        if isinstance(node.op, ast.Pow):
            if isinstance(right, int) and not isinstance(right, bool) and abs(right) > 10000:
                raise SafeEvalError(
                    f"Exponent {right} is too large (max 10000) to prevent memory exhaustion"
                )
        try:
            return fn(left, right)
        except (ZeroDivisionError, OverflowError, ValueError) as exc:
            raise SafeEvalError(str(exc)) from exc

    def _visit_BoolOp(self, node: ast.BoolOp) -> Any:
        op_type = type(node.op)
        if op_type == ast.And:
            result = True
            for v in node.values:
                result = self.visit(v)
                if not result:
                    return result
            return result
        elif op_type == ast.Or:
            result = False
            for v in node.values:
                result = self.visit(v)
                if result:
                    return result
            return result
        raise SafeEvalError(f"Unsupported BoolOp: {op_type.__name__}")

    def _visit_Compare(self, node: ast.Compare) -> Any:
        # Support chained comparisons: a < b < c
        left = self.visit(node.left)
        for op, comparator_node in zip(node.ops, node.comparators):
            right = self.visit(comparator_node)
            fn = _CMP_OPS.get(type(op))
            if fn is None:
                raise SafeEvalError(f"Unsupported comparator: {type(op).__name__}")
            if not fn(left, right):
                return False
            left = right
        return True

    def _visit_IfExp(self, node: ast.IfExp) -> Any:
        test = self.visit(node.test)
        if test:
            return self.visit(node.body)
        return self.visit(node.orelse)

    # --- Call --------------------------------------------------------------

    def _visit_Call(self, node: ast.Call) -> Any:
        # Reject **kwargs and *args
        if node.keywords:
            raise SafeEvalError("Keyword arguments are not allowed in safe_eval")

        # Callee must be a Name or a whitelisted Attribute (np.sin, math.cos)
        func_node = node.func
        if isinstance(func_node, ast.Name):
            callee = self.visit(func_node)
        elif isinstance(func_node, ast.Attribute):
            # _visit_Attribute already enforces the np/math restriction
            callee = self._visit_Attribute(func_node)
        else:
            raise SafeEvalError(
                f"Call callee must be a Name or np./math. attribute, "
                f"got {type(func_node).__name__}"
            )

        args = [self.visit(a) for a in node.args]
        try:
            return callee(*args)
        except SafeEvalError:
            raise
        except (ZeroDivisionError, OverflowError, ValueError) as exc:
            raise SafeEvalError(str(exc)) from exc
        except Exception as exc:
            # An unexpected error from an allowlisted callable may be a genuine
            # defect rather than user-input error. Log it before reclassifying
            # as a SafeEvalError so it isn't silently masked.
            _logger.debug("Unexpected error calling %r in safe_eval: %s",
                          getattr(callee, "__name__", callee), exc, exc_info=True)
            raise SafeEvalError(f"Call error: {exc}") from exc

    # --- Subscript ---------------------------------------------------------

    def _visit_Subscript(self, node: ast.Subscript) -> Any:
        base = self.visit(node.value)
        # In Python 3.9+ the slice is a direct node; in 3.8- it's an Index wrapper
        index = self.visit(node.slice)
        try:
            return base[index]
        except (IndexError, KeyError, TypeError) as exc:
            raise SafeEvalError(str(exc)) from exc

    def _visit_Index(self, node) -> Any:  # type: ignore[override]
        # Python 3.8 compat — Index wraps the actual value
        return self.visit(node.value)  # type: ignore[attr-defined]

    def _visit_Slice(self, node: ast.Slice) -> Any:
        lower = self.visit(node.lower) if node.lower else None
        upper = self.visit(node.upper) if node.upper else None
        step  = self.visit(node.step)  if node.step  else None
        return slice(lower, upper, step)

    def _visit_ExtSlice(self, node) -> Any:  # type: ignore[override]
        # Python 3.8 compat for multi-dim: ExtSlice wraps multiple dims
        return tuple(self.visit(d) for d in node.dims)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def safe_literal(s: Any) -> Any:
    """
    Wrapper around ast.literal_eval.

    Already-parsed values (int, float, list, tuple, dict, np.ndarray) are
    returned unchanged. Empty/whitespace strings raise SafeEvalError.
    """
    if not isinstance(s, str):
        return s
    if not s.strip():
        raise SafeEvalError("Empty expression")
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError) as exc:
        raise SafeEvalError(str(exc)) from exc


def safe_expr(s: Any, variables: Optional[Dict[str, Any]] = None,
              allow_numpy: bool = True) -> Any:
    """
    Evaluate a numeric expression using a hardened AST walker.

    Non-str values are returned as-is. Empty/whitespace strings raise.
    """
    if not isinstance(s, str):
        return s
    if not s.strip():
        raise SafeEvalError("Empty expression")
    if len(s) > _MAX_LEN:
        raise SafeEvalError("expression too long")

    env = _build_env(variables, allow_numpy)
    try:
        tree = ast.parse(s, mode="eval")
    except SyntaxError as exc:
        raise SafeEvalError(str(exc)) from exc

    walker = _Walker(env)
    try:
        result = walker.visit(tree)
    except SafeEvalError:
        raise
    except OverflowError as exc:
        raise SafeEvalError(str(exc)) from exc

    return result


# ---------------------------------------------------------------------------
# CompiledExpr — parse once, evaluate many
# ---------------------------------------------------------------------------

class CompiledExpr:
    """
    Pre-parsed, validated expression. Call with variables dict to evaluate.

    Use for per-step hot-loop evaluation where the expression string is fixed
    but variables change each iteration.
    """

    def __init__(self, tree: ast.Expression, allow_numpy: bool):
        self._tree = tree
        self._allow_numpy = allow_numpy

    def __call__(self, variables: Optional[Dict[str, Any]] = None) -> Any:
        env = _build_env(variables, self._allow_numpy)
        walker = _Walker(env)
        try:
            return walker.visit(self._tree)
        except SafeEvalError:
            raise
        except OverflowError as exc:
            raise SafeEvalError(str(exc)) from exc


def compile_expr(s: str, allow_numpy: bool = True) -> CompiledExpr:
    """
    Parse and validate s once. Returns a CompiledExpr callable.

    Raises SafeEvalError at compile time for any disallowed construct.
    """
    if not isinstance(s, str):
        raise SafeEvalError("compile_expr requires a str")
    if not s.strip():
        raise SafeEvalError("Empty expression")
    if len(s) > _MAX_LEN:
        raise SafeEvalError("expression too long")

    try:
        tree = ast.parse(s, mode="eval")
    except SyntaxError as exc:
        raise SafeEvalError(str(exc)) from exc

    # Validate at compile time by doing a dry-run walk with an empty env.
    # We use a permissive env that accepts any Name so structural checks pass
    # even if variable names aren't known yet.
    env = _build_env({}, allow_numpy)
    _validate_tree(tree, env)

    return CompiledExpr(tree, allow_numpy)


# Marker/structural node base classes that _Walker.visit never receives
# directly (operators are dispatched via type(node.op) lookups, contexts are
# read off Name/Subscript nodes, etc.). These must be skipped when checking
# for a corresponding _visit_<name> handler.
_NON_VISITED_NODE_BASES = (
    ast.expr_context,   # Load / Store / Del
    ast.operator,       # Add / Sub / Mult / ...
    ast.unaryop,        # UAdd / USub / Not / Invert
    ast.boolop,         # And / Or
    ast.cmpop,          # Eq / Lt / ...
    ast.comprehension,  # only inside (already-forbidden) comprehensions
    ast.arguments,
    ast.arg,
    ast.keyword,        # Call keywords are rejected explicitly below
)


def _validate_tree(tree: ast.Expression, env: Dict[str, Any]) -> None:
    """
    Walk the AST for structural validity without resolving Name values.
    Raises SafeEvalError if any forbidden node is encountered.
    """
    for node in ast.walk(tree):
        name = type(node).__name__
        if name in _FORBIDDEN_NODE_NAMES:
            raise SafeEvalError(f"Forbidden AST node: {name}")

        # Mirror _Walker.visit: any node that the walker would actually visit
        # must have a _visit_<name> handler. Reject unsupported-but-not-forbidden
        # nodes at compile time so compile_expr fully validates structure upfront
        # (matching its docstring) instead of only failing at first __call__.
        if (node is not tree
                and not isinstance(node, _NON_VISITED_NODE_BASES)
                and not hasattr(_Walker, f"_visit_{name}")):
            raise SafeEvalError(f"Unsupported AST node: {name}")

        # Check Attribute nodes at compile time
        if isinstance(node, ast.Attribute):
            if not isinstance(node.value, ast.Name):
                raise SafeEvalError(
                    "Attribute access only allowed on bare 'np' or 'math' names"
                )
            ns_name = node.value.id
            attr = node.attr
            if ns_name == "np" and attr not in _NP_ALLOWLIST:
                raise SafeEvalError(f"np.{attr} is not in the numpy allowlist")
            elif ns_name == "math" and attr not in _MATH_ALLOWLIST:
                raise SafeEvalError(f"math.{attr} is not in the math allowlist")
            elif ns_name not in ("np", "math"):
                raise SafeEvalError(
                    f"Attribute access not allowed on '{ns_name}' (only np/math)"
                )

        # Check Call callee type
        if isinstance(node, ast.Call):
            if node.keywords:
                raise SafeEvalError("Keyword arguments are not allowed in safe_eval")
            func_node = node.func
            if not isinstance(func_node, (ast.Name, ast.Attribute)):
                raise SafeEvalError(
                    "Call callee must be a Name or np./math. attribute"
                )
