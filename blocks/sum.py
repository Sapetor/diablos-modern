import numpy as np
from blocks.base_block import BaseBlock

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
        return [{"name": "in1", "type": "any"}, {"name": "in2", "type": "any"}]

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    def execute(self, time, inputs, params):
        try:
            suma = 0.0
            for i in sorted(inputs.keys()):
                sign = '+' # Default sign
                if i < len(params['sign']):
                    sign = params['sign'][i]
                
                input_value = np.atleast_1d(inputs[i])

                if sign == '+':
                    suma += input_value
                elif sign == '-':
                    suma -= input_value
            
            return {0: suma}
        except (ValueError, TypeError) as e:
            print(f"ERROR: Invalid input type in sum block. Expected numeric. Error: {str(e)}")
            return {'E': True}
