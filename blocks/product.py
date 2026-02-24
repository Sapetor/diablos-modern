import logging
import numpy as np
from blocks.base_block import BaseBlock

logger = logging.getLogger(__name__)


class ProductBlock(BaseBlock):
    """
    A block that multiplies or divides multiple input signals.
    Similar to Sum block but for multiplication/division operations.
    """

    def __init__(self):
        super().__init__()

    @property
    def block_name(self):
        return "Product"

    @property
    def category(self):
        return "Math"

    @property
    def doc(self):
        return (
            "Multiplies or divides multiple input signals."
            "\n\nParameters:"
            "\n- Ops: A string of '*' and '/' characters defining the operation for each input port."
            "\n  Example: '*/' creates 2 ports: (in1 / in2)."
            "\n  Example: '**' creates 2 ports: (in1 * in2)."
            "\n\nUsage:"
            "\nUsed for signal modulation, ratio calculations, or Newton's method (f/f')."
        )

    @property
    def params(self):
        return {
            "ops": {"default": "**", "type": "string"},
        }

    @property
    def inputs(self):
        """Default inputs (used when block is first created)."""
        return [{"name": "in1", "type": "any"}, {"name": "in2", "type": "any"}]

    def get_inputs(self, params=None):
        """
        Get inputs dynamically based on the 'ops' parameter.

        The 'ops' parameter is a string where each character represents
        an input: '*' for multiply, '/' for divide.
        Example: "**" creates 2 inputs, both multiply
        Example: "*/" creates 2 inputs, first multiply, second divide
        """
        if params is None:
            params = self.params

        # Get the ops string
        ops_string = params.get('ops', '**')

        # Handle case where ops might still be wrapped in a dict (initialization)
        if isinstance(ops_string, dict):
            ops_string = ops_string.get('default', '**')

        # Create an input for each character in the ops string
        num_inputs = len(ops_string)
        return [{"name": f"in{i+1}", "type": "any"} for i in range(num_inputs)]

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    def draw_icon(self, block_rect):
        """Draw multiplication X symbol in normalized 0-1 coordinates."""
        from PyQt5.QtGui import QPainterPath
        path = QPainterPath()
        path.moveTo(0.2, 0.2)
        path.lineTo(0.8, 0.8)
        path.moveTo(0.2, 0.8)
        path.lineTo(0.8, 0.2)
        return path

    def symbolic_execute(self, inputs, params):
        """
        Symbolic execution for equation extraction.

        Returns symbolic expression: y = u1 * u2 / u3 ...

        Args:
            inputs: Dict of symbolic input expressions {port_idx: sympy_expr}
            params: Dict of block parameters

        Returns:
            Dict of symbolic output expressions {0: product_expr}
        """
        try:
            from sympy import Symbol, Integer
        except ImportError:
            return None

        ops_string = params.get('ops', '**')
        result = Integer(1)

        for i in range(len(ops_string)):
            op = ops_string[i] if i < len(ops_string) else '*'
            u = inputs.get(i, Symbol(f'u{i}'))

            if op == '*':
                result = result * u
            else:  # '/'
                result = result / u

        return {0: result}

    def execute(self, time, inputs, params, **kwargs):
        try:
            result = 1.0
            ops_string = params.get('ops', '**')

            for i in sorted(inputs.keys()):
                op = '*'  # Default operation
                if i < len(ops_string):
                    op = ops_string[i]

                input_value = np.atleast_1d(inputs[i])

                if op == '*':
                    result = result * input_value
                elif op == '/':
                    # Handle division by zero gracefully
                    with np.errstate(divide='ignore', invalid='ignore'):
                        result = result / input_value
                        has_inf = np.any(np.isinf(result))
                        has_nan = np.any(np.isnan(result))
                        if has_inf or has_nan:
                            logger.warning(f"Product block '{params.get('_name_', '?')}': division by zero (inf={has_inf}, nan={has_nan})")
                        # Replace inf/-inf with large finite numbers
                        result = np.where(np.isinf(result), np.sign(result) * 1e308, result)
                        # Replace NaN with 0
                        result = np.where(np.isnan(result), 0.0, result)

            return {0: result}
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid input type in product block. Expected numeric. Error: {str(e)}")
            return {'E': True, 'error': f'Invalid input type in product block: {str(e)}'}
