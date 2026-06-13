
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
            "\n\nThe 'type' parameter selects the waveform:"
            "\n- up: 0 before Delay, Value afterwards (default)."
            "\n- down: Value before Delay, 0 afterwards."
            "\n- pulse: square wave with Delay as the half-period."
            "\n- constant: always outputs Value."
            "\n- impulse: Dirac-delta approximation (Value/dt for one step at"
            " Delay, 0 elsewhere); equivalent to the standalone Impulse block."
            "\n\nUsage:"
            "\nCommonly used to test step response of control systems."
        )

    @property
    def params(self):
        return {
            "value": {"type": "float", "default": 1.0, "doc": "The value of the step (or impulse strength/area when type is 'impulse')."},
            "delay": {"type": "float", "default": 0.0, "doc": "The delay of the step (half-period when type is 'pulse'; fire time when type is 'impulse')."},
            "type": {"type": "string", "default": "up", "doc": "Waveform: up, down, pulse, constant, or impulse (Dirac-delta approximation, same as the Impulse block)."},
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

    @staticmethod
    def _impulse_output(time, params, dt):
        """Shared Dirac-delta approximation for the type=='impulse' subtype.

        Mirrors the standalone ImpulseBlock: emit value/dt for the single first
        sample at or after ``delay`` (recorded via the ``_impulse_fired`` flag in
        params), and 0 elsewhere. Kept here as a same-file helper so the impulse
        shape is defined once for the Step block's 'impulse' branch.
        """
        delay = float(params.get('delay', 0.0))
        value = params.get('value', 1.0)
        if not params.get('_impulse_fired', False) and time >= delay:
            params['_impulse_fired'] = True
            return {0: np.atleast_1d(np.array(value / dt, dtype=float)), 'E': False}
        return {0: np.atleast_1d(np.zeros_like(np.array(value, dtype=float))), 'E': False}

    def execute(self, time, inputs, params, **kwargs):
        if params.get('_init_start_', True):
            params['_step_old'] = time
            params['_change_old'] = not params.get('pulse_start_up', True)
            params['_impulse_fired'] = False
            params['_init_start_'] = False

        delay = float(params.get('delay', 0.0))
        step_type = params.get('type', 'up')

        if step_type == 'up':
            change = True if time < delay else False
        elif step_type == 'down':
            change = True if time > delay else False
        elif step_type == 'pulse':
            if delay <= 0:
                return {'E': True, 'error': "pulse 'delay' (half-period) must be positive"}
            # Advance across every half-period boundary crossed since last call so
            # that large/variable timesteps do not skip toggles (mirrors PRBS).
            while time - params['_step_old'] >= delay:
                params['_step_old'] += delay
                params['_change_old'] = not params['_change_old']
            change = params['_change_old']
        elif step_type == 'impulse':
            dt = kwargs.get('dtime', params.get('dtime', 0.01))
            return self._impulse_output(time, params, dt)
        elif step_type == 'constant':
            change = False
        else:
            return {'E': True, 'error': f"unknown step type {step_type}"}

        value = params.get('value', 1.0)
        if change:
            params['_change_old'] = True
            return {0: np.atleast_1d(np.zeros_like(np.array(value, dtype=float))), 'E': False}
        else:
            params['_change_old'] = False
            return {0: np.atleast_1d(np.array(value, dtype=float)), 'E': False}
