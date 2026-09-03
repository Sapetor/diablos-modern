import logging
import numpy as np
from blocks.base_block import BaseBlock

logger = logging.getLogger(__name__)

# Operator symbol -> elementwise numpy comparison.
_REL_OPS = {
    ">": np.greater,
    ">=": np.greater_equal,
    "<": np.less,
    "<=": np.less_equal,
    "==": np.equal,
    "!=": np.not_equal,
}


class RelationalOperatorBlock(BaseBlock):
    """
    Compare two input signals with a relational operator.

    Output is 1.0 where the comparison is true and 0.0 where it is false
    (element-wise for vector signals). Useful for switching laws, guards, and
    threshold logic in hybrid-system teaching examples.
    """

    @property
    def block_name(self):
        return "RelationalOperator"

    @property
    def category(self):
        return "Logic"

    @property
    def color(self):
        return "orange"

    @property
    def doc(self):
        return (
            "Compare two inputs: y = (in1 OP in2)."
            "\n\nParameters:"
            "\n- operator: one of >, >=, <, <=, ==, !="
            "\n\nOutput:"
            "\n- 1.0 where the comparison holds, 0.0 otherwise (element-wise)."
            "\n\nUsage:"
            "\nDrive a Switch control port, or build bang-bang / threshold logic."
        )

    @property
    def params(self):
        return {
            "operator": {
                "type": "choice",
                "default": ">",
                "options": [">", ">=", "<", "<=", "==", "!="],
                "doc": "Relational comparison applied as in1 OP in2.",
            }
        }

    @property
    def inputs(self):
        return [{"name": "in1", "type": "any"}, {"name": "in2", "type": "any"}]

    @property
    def outputs(self):
        return [{"name": "y", "type": "any"}]

    def draw_icon(self, block_rect):
        """Draw a comparison chevron fed by two input stubs."""
        from PyQt5.QtGui import QPainterPath

        path = QPainterPath()
        # Input stubs
        path.moveTo(0.04, 0.24)
        path.lineTo(0.22, 0.24)
        path.moveTo(0.04, 0.76)
        path.lineTo(0.22, 0.76)
        # ">" chevron
        path.moveTo(0.30, 0.10)
        path.lineTo(0.70, 0.50)
        path.lineTo(0.30, 0.90)
        # Output stub
        path.moveTo(0.72, 0.50)
        path.lineTo(0.96, 0.50)
        return path

    def execute(self, time, inputs, params, **kwargs):
        op = str(params.get("operator", ">"))
        fn = _REL_OPS.get(op)
        if fn is None:
            return {"E": True, "error": f"RelationalOperator: unknown operator '{op}'"}
        try:
            a = np.atleast_1d(inputs.get(0, 0.0)).astype(float)
            b = np.atleast_1d(inputs.get(1, 0.0)).astype(float)
            return {0: fn(a, b).astype(float)}
        except (ValueError, TypeError) as e:
            logger.error(f"RelationalOperator error: {e}")
            return {"E": True, "error": str(e)}
