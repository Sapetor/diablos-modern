
import numpy as np
from blocks.base_block import BaseBlock


class ImpulseBlock(BaseBlock):
    """
    A block that generates a discrete impulse (Dirac delta approximation).

    Outputs value/dt for one simulation time step at the specified delay,
    and 0 elsewhere. The integral over all time equals value.
    """

    @property
    def block_name(self):
        return "Impulse"

    @property
    def category(self):
        return "Sources"

    @property
    def b_type(self):
        return 0

    @property
    def color(self):
        return "blue"

    @property
    def doc(self):
        return (
            "Generates a discrete impulse (Dirac delta approximation)."
            "\n\nOutputs Value/dt for one time step at the Delay time, 0 elsewhere."
            "\nThe integral of the output equals Value."
            "\n\nParameters:"
            "\n- Value: The impulse strength (area under the pulse)."
            "\n- Delay: Time (seconds) when the impulse fires."
            "\n\nUsage:"
            "\nUsed to obtain impulse responses of transfer functions and systems."
        )

    @property
    def params(self):
        return {
            "value": {"type": "float", "default": 1.0, "doc": "Impulse strength (area)."},
            "delay": {"type": "float", "default": 0.0, "doc": "Time when the impulse fires."},
        }

    @property
    def inputs(self):
        return []

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    def draw_icon(self, block_rect):
        """Draw impulse icon: vertical spike with arrow."""
        from PyQt5.QtGui import QPainterPath
        path = QPainterPath()
        # Baseline
        path.moveTo(0.1, 0.8)
        path.lineTo(0.9, 0.8)
        # Spike
        path.moveTo(0.35, 0.8)
        path.lineTo(0.35, 0.15)
        # Arrowhead
        path.moveTo(0.25, 0.3)
        path.lineTo(0.35, 0.15)
        path.lineTo(0.45, 0.3)
        return path

    def execute(self, time, inputs, params, **kwargs):
        # The impulse area equals value only if value/dt uses the *actual*
        # solver step. Guessing a fallback (e.g. 0.01) silently rescales the
        # Dirac approximation to value*(dt_real/0.01), so require a real dt from
        # the engine and fail loudly if it is missing instead of fabricating one.
        dt = kwargs.get('dtime', params.get('dtime', None))
        if dt is None:
            return {0: np.atleast_1d(np.array(0.0, dtype=float)), 'E': True,
                    'error': 'Impulse requires the simulation step (dtime) to '
                             'scale value/dt; none was supplied by the engine.'}
        dt = float(dt)
        delay = float(params.get('delay', 0.0))
        value = float(params.get('value', 1.0))

        if params.get('_init_start_', True):
            params['_impulse_fired'] = False
            params['_init_start_'] = False

        # Half-open right-edge convention: the impulse fires on the first sample
        # at or after `delay` (time >= delay). When `delay` falls between grid
        # points the spike therefore lands up to one dt late. This matches the
        # Step block's 'impulse' branch (step.py) so the two stay consistent.
        if not params.get('_impulse_fired', False) and time >= delay:
            params['_impulse_fired'] = True
            return {0: np.atleast_1d(np.array(value / dt, dtype=float)), 'E': False}

        return {0: np.atleast_1d(np.array(0.0, dtype=float)), 'E': False}
