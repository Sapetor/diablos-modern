
import logging
import numpy as np
from blocks.base_block import BaseBlock
from lib.safe_eval import safe_expr, SafeEvalError

logger = logging.getLogger(__name__)


class FunctionBlock(BaseBlock):
    """
    Evaluate a user-supplied Python expression of the block's inputs and time.

    Inputs are exposed two ways:
      - ``u`` is the 0-indexed list of input signals: ``u[0]``, ``u[1]``, ...
      - ``u1``, ``u2``, ... are 1-indexed aliases (MATLAB/Simulink ``Fcn`` style).
    The current simulation time is bound to ``t``.

    The number of input ports is editable from the property panel
    (``io_editable == 'input'``). Expressions are evaluated through the hardened
    ``safe_expr`` AST walker, so numpy math (``sin``, ``cos``, ``np.tanh``, ...)
    is available but imports, attribute escapes, and statements are rejected.
    """

    @property
    def block_name(self):
        return "Function"

    @property
    def category(self):
        return "Math"

    @property
    def color(self):
        return "green"

    @property
    def io_editable(self):
        # Users add/remove input ports via the property-editor port spinner.
        return 'input'

    @property
    def doc(self):
        return (
            "Evaluate a Python expression of the inputs and time."
            "\n\nInputs:"
            "\n- u[0], u[1], ...  (0-indexed list of input ports)"
            "\n- u1, u2, ...      (1-indexed aliases, MATLAB/Simulink Fcn style)"
            "\n- t                (current simulation time)"
            "\n\nMath:"
            "\n- Bare numpy functions: sin, cos, tan, exp, log, sqrt, sign, abs, ..."
            "\n- Prefixed: np.tanh(u[0]), math.atan2(u[1], u[0])"
            "\n- Constants: pi, e"
            "\n\nParameters:"
            "\n- expression: e.g. 'sin(u[0])', 'u1*u2 + t', '[u[0], u[0]**2]'"
            "\n- Input ports: set the port count in the property panel."
            "\n\nUsage:"
            "\nModel arbitrary nonlinearities (pendulum sin(theta), friction"
            "\ncurves, switching laws) without writing a new block class."
            "\nReturn a list to produce a vector output, e.g. '[u[0], u[1]]'."
        )

    @property
    def params(self):
        return {
            "expression": {
                "type": "string",
                "default": "u[0]",
                "doc": "Python expression using u[i]/u1.. and t. e.g. 'sin(u[0]) + t'.",
            }
        }

    @property
    def inputs(self):
        return [{"name": "u1", "type": "any"}]

    @property
    def outputs(self):
        return [{"name": "y", "type": "any"}]

    def draw_icon(self, block_rect):
        """Render with the fallback block-name label."""
        return None

    def execute(self, time, inputs, params, **kwargs):
        expr = str(params.get("expression", "u[0]"))

        # Port count is engine-managed via '_inputs_'; for direct unit calls that
        # omit it, infer the count from the supplied input indices.
        if "_inputs_" in params:
            n = int(params["_inputs_"])
        elif inputs:
            n = max(inputs.keys()) + 1
        else:
            n = 1

        signals = [inputs.get(i, 0.0) for i in range(n)]

        variables = {"t": time, "u": signals}
        for i, sig in enumerate(signals):
            variables[f"u{i + 1}"] = sig  # 1-indexed aliases: u1, u2, ...

        try:
            result = safe_expr(expr, variables=variables, allow_numpy=True)
            return {0: np.atleast_1d(np.asarray(result, dtype=float))}
        except (SafeEvalError, ValueError, TypeError) as e:
            name = params.get("_name_", "Function")
            logger.warning(f"{name} expression error in '{expr}': {e}")
            return {"E": True, "error": f"Function expression error: {e}"}
