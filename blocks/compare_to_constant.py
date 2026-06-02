
import logging
import numpy as np
from blocks.base_block import BaseBlock

logger = logging.getLogger(__name__)

_REL_OPS = {
    ">": np.greater,
    ">=": np.greater_equal,
    "<": np.less,
    "<=": np.less_equal,
    "==": np.equal,
    "!=": np.not_equal,
}


class CompareToConstantBlock(BaseBlock):
    """
    Compare an input signal against a fixed constant.

    Output is 1.0 where ``input OP constant`` holds and 0.0 otherwise
    (element-wise). A one-input convenience form of RelationalOperator.
    """

    @property
    def block_name(self):
        return "CompareToConstant"

    @property
    def category(self):
        return "Logic"

    @property
    def color(self):
        return "orange"

    @property
    def doc(self):
        return (
            "Compare the input against a constant: y = (in OP constant)."
            "\n\nParameters:"
            "\n- operator: one of >, >=, <, <=, ==, !="
            "\n- constant: the value to compare against."
            "\n\nOutput:"
            "\n- 1.0 where the comparison holds, 0.0 otherwise (element-wise)."
            "\n\nUsage:"
            "\nThreshold detection, e.g. trigger when a signal exceeds a setpoint."
        )

    @property
    def params(self):
        return {
            "operator": {
                "type": "choice",
                "default": ">",
                "options": [">", ">=", "<", "<=", "==", "!="],
                "doc": "Comparison applied as input OP constant.",
            },
            "constant": {
                "type": "float",
                "default": 0.0,
                "doc": "Constant value compared against the input.",
            },
        }

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return [{"name": "y", "type": "any"}]

    def execute(self, time, inputs, params, **kwargs):
        op = str(params.get("operator", ">"))
        fn = _REL_OPS.get(op)
        if fn is None:
            return {"E": True, "error": f"CompareToConstant: unknown operator '{op}'"}
        try:
            u = np.atleast_1d(inputs.get(0, 0.0)).astype(float)
            c = float(params.get("constant", 0.0))
            return {0: fn(u, c).astype(float)}
        except (ValueError, TypeError) as e:
            logger.error(f"CompareToConstant error: {e}")
            return {"E": True, "error": str(e)}
