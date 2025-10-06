
import numpy as np
from blocks.base_block import BaseBlock

class DerivativeBlock(BaseBlock):
    """
    A block that calculates the derivative of a signal.
    """

    def __init__(self):
        super().__init__()
        self.t_old = 0
        self.i_old = 0
        self.didt_old = 0

    @property
    def block_name(self):
        return "Deriv"

    @property
    def category(self):
        return "Math"

    @property
    def color(self):
        return "lime_green"

    @property
    def doc(self):
        return "Calculates the derivative of a signal."

    @property
    def params(self):
        return {}

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    def execute(self, time, inputs, params):
        if params.get('_init_start_', True):
            self.t_old = time
            self.i_old = np.array(inputs[0], dtype=float)
            self.didt_old = np.zeros_like(self.i_old)
            params['_init_start_'] = False
            return {0: self.didt_old}
        
        if time == self.t_old:
            return {0: np.array(self.didt_old)}
        
        dt = time - self.t_old
        di = np.array(inputs[0], dtype=float) - self.i_old
        didt = di/dt
        
        self.t_old = time
        self.i_old = np.array(inputs[0], dtype=float)
        self.didt_old = didt
        
        return {0: np.array(didt)}
