
import logging
import numpy as np
from blocks.base_block import BaseBlock

logger = logging.getLogger(__name__)


class LogicalOperatorBlock(BaseBlock):
    """
    Combine input signals with a boolean logical operator.

    Inputs are treated as boolean: any nonzero value is True. Output is 1.0
    (True) or 0.0 (False), element-wise. The number of input ports is editable
    from the property panel; ``NOT`` uses only the first input.
    """

    @property
    def block_name(self):
        return "LogicalOperator"

    @property
    def category(self):
        return "Logic"

    @property
    def color(self):
        return "orange"

    @property
    def io_editable(self):
        # Variable number of inputs; NOT operates on the first input only.
        return 'input'

    @property
    def doc(self):
        return (
            "Boolean logic over the inputs (nonzero = True)."
            "\n\nParameters:"
            "\n- operator: AND, OR, NAND, NOR, XOR, NOT"
            "\n- Input ports: set the port count in the property panel."
            "\n\nOutput:"
            "\n- 1.0 (True) or 0.0 (False), element-wise."
            "\n\nUsage:"
            "\nGate events, combine threshold detectors, build switching logic."
            "\nNOT uses only the first input."
        )

    @property
    def params(self):
        return {
            "operator": {
                "type": "choice",
                "default": "AND",
                "options": ["AND", "OR", "NAND", "NOR", "XOR", "NOT"],
                "doc": "Boolean operator applied across the inputs.",
            }
        }

    @property
    def inputs(self):
        return [{"name": "in1", "type": "any"}, {"name": "in2", "type": "any"}]

    @property
    def outputs(self):
        return [{"name": "y", "type": "any"}]

    def execute(self, time, inputs, params, **kwargs):
        op = str(params.get("operator", "AND")).upper()

        if "_inputs_" in params:
            n = int(params["_inputs_"])
        elif inputs:
            n = max(inputs.keys()) + 1
        else:
            n = 1
        n = max(n, 1)

        try:
            bvals = [np.atleast_1d(inputs.get(i, 0.0)).astype(float) != 0 for i in range(n)]

            if op == "NOT":
                result = np.logical_not(bvals[0])
            elif op in ("AND", "NAND"):
                result = bvals[0]
                for b in bvals[1:]:
                    result = np.logical_and(result, b)
                if op == "NAND":
                    result = np.logical_not(result)
            elif op in ("OR", "NOR"):
                result = bvals[0]
                for b in bvals[1:]:
                    result = np.logical_or(result, b)
                if op == "NOR":
                    result = np.logical_not(result)
            elif op == "XOR":
                result = bvals[0]
                for b in bvals[1:]:
                    result = np.logical_xor(result, b)
            else:
                return {"E": True, "error": f"LogicalOperator: unknown operator '{op}'"}

            return {0: np.atleast_1d(result).astype(float)}
        except (ValueError, TypeError, IndexError) as e:
            logger.error(f"LogicalOperator error: {e}")
            return {"E": True, "error": str(e)}
