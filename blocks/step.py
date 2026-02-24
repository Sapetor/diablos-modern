
import numpy as np
from blocks.base_block import BaseBlock

class StepBlock(BaseBlock):
    """
    A block that generates a step signal.
    """

    def __init__(self):
        super().__init__()

    @property
    def block_name(self):
        return "Step"

    @property
    def category(self):
        return "Sources"

    @property
    def b_type(self):
        """Source block - generates output without requiring input."""
        return 0

    @property
    def color(self):
        return "blue"

    @property
    def doc(self):
        return (
            "Generates a Step function."
            "\n\nOutput is 0 before 'Delay' time, and 'Final Value' afterwards."
            "\n\nParameters:"
            "\n- Final Value: The height of the step."
            "\n- Step Time: Time (seconds) when the step occurs."
            "\n\nUsage:"
            "\nCommonly used to test step response of control systems."
        )

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

    def draw_icon(self, block_rect):
        """Draw step signal icon in normalized 0-1 coordinates."""
        from PyQt5.QtGui import QPainterPath
        path = QPainterPath()
        # Step signal: low -> high
        path.moveTo(0.1, 0.7)
        path.lineTo(0.5, 0.7)
        path.lineTo(0.5, 0.3)
        path.lineTo(0.9, 0.3)
        return path

    def execute(self, time, inputs, params, **kwargs):
        if params.get('_init_start_', True):
            params['_step_old'] = time
            params['_change_old'] = not params.get('pulse_start_up', True)
            params['_init_start_'] = False

        delay = float(params.get('delay', 0.0))
        step_type = params.get('type', 'up')

        if step_type == 'up':
            change = True if time < delay else False
        elif step_type == 'down':
            change = True if time > delay else False
        elif step_type == 'pulse':
            if time - params['_step_old'] >= delay:
                params['_step_old'] += delay
                change = not params['_change_old']
            else:
                change = params['_change_old']
        elif step_type == 'constant':
            change = False
        else:
            print("ERROR: 'type' not correctly defined in", params.get('_name_', 'unknown'))
            return {0: 0.0, 'E': True}

        value = params.get('value', 1.0)
        if change:
            params['_change_old'] = True
            return {0: np.atleast_1d(np.zeros_like(np.array(value, dtype=float))), 'E': False}
        else:
            params['_change_old'] = False
            return {0: np.atleast_1d(np.array(value, dtype=float)), 'E': False}
