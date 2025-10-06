
import numpy as np
from blocks.base_block import BaseBlock

class StepBlock(BaseBlock):
    """
    A block that generates a step signal.
    """

    def __init__(self):
        super().__init__()
        self.step_old = 0
        self.change_old = False

    @property
    def block_name(self):
        return "Step"

    @property
    def category(self):
        return "Sources"

    @property
    def color(self):
        return "blue"

    @property
    def doc(self):
        return "Generates a step signal."

    @property
    def params(self):
        return {
            "value": {"type": "float", "default": 1.0, "doc": "The value of the step."},
            "delay": {"type": "float", "default": 0.0, "doc": "The delay of the step."},
            "type": {"type": "string", "default": "up", "doc": "up, down, pulse, constant"},
            "pulse_start_up": {"type": "bool", "default": True, "doc": "If type is pulse, defines if it starts up or down."}
        }

    @property
    def inputs(self):
        return []

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    def execute(self, time, inputs, params):
        if params.get('_init_start_', True):
            self.step_old = time
            self.change_old = not params['pulse_start_up']
            params['_init_start_'] = False
        
        delay = float(params['delay'])
        
        if params['type'] == 'up':
            change = True if time < delay else False
        elif params['type'] == 'down':
            change = True if time > delay else False
        elif params['type'] == 'pulse':
            if time - self.step_old >= delay:
                self.step_old += delay
                change = not self.change_old
            else:
                change = self.change_old
        elif params['type'] == 'constant':
            change = False
        else:
            print("ERROR: 'type' not correctly defined in", params['_name_'])
            return {'E': True}

        if change:
            self.change_old = True
            return {0: np.atleast_1d(np.zeros_like(np.array(params['value'], dtype=float)))}
        else:
            self.change_old = False
            return {0: np.atleast_1d(np.array(params['value'], dtype=float))}
