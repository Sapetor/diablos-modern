import logging
import numpy as np
from blocks.base_block import BaseBlock

logger = logging.getLogger(__name__)


class SumBlock(BaseBlock):
    def __init__(self):
        super().__init__()

    @property
    def block_name(self):
        return "Sum"

    @property
    def category(self):
        return "Math"

    @property
    def color(self):
        return "lime_green"

    @property
    def params(self):
        return {
            "sign": {"default": "++", "type": "string"},
        }

    @property
    def inputs(self):
        """Default inputs (used when block is first created)."""
        return [{"name": "in1", "type": "any"}, {"name": "in2", "type": "any"}]

    def get_inputs(self, params=None):
        """
        Get inputs dynamically based on the 'sign' parameter.

        The 'sign' parameter is a string where each character represents
        an input: '+' for positive, '-' for negative.
        Example: "+++" creates 3 inputs, all positive
        Example: "++-" creates 3 inputs, first two positive, third negative
        """
        if params is None:
            params = self.params

        # Get the sign string - params is a flat dict like {'sign': '++'}
        sign_string = params.get('sign', '++')

        # Handle case where sign might still be wrapped in a dict (initialization)
        if isinstance(sign_string, dict):
            sign_string = sign_string.get('default', '++')

        # Create an input for each character in the sign string
        num_inputs = len(sign_string)
        return [{"name": f"in{i+1}", "type": "any"} for i in range(num_inputs)]

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    def draw_icon(self, block_rect):
        """Sum block icon is the sign text - handled specially in draw_Block."""
        # Sum uses text rendering, so we return None and let the switch handle it
        return None

    def execute(self, time, inputs, params):
        try:
            suma = 0.0
            sign_string = params.get('sign', '++')

            for i in sorted(inputs.keys()):
                sign = '+' # Default sign
                if i < len(sign_string):
                    sign = sign_string[i]

                input_value = np.atleast_1d(inputs[i])

                if sign == '+':
                    suma += input_value
                elif sign == '-':
                    suma -= input_value

            return {0: suma}
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid input type in sum block. Expected numeric. Error: {str(e)}")
            return {'error': True}

