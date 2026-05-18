from blocks.base_block import BaseBlock
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ScopeBlock(BaseBlock):
    def __init__(self):
        super().__init__()

    @property
    def block_name(self):
        return "Scope"

    @property
    def category(self):
        return "Sinks"

    @property
    def b_type(self):
        """Sink block - consumes output without producing further output."""
        return 3

    @property
    def color(self):
        return "red"

    @property
    def io_editable(self):
        return 'input'

    @property
    def params(self):
        from blocks.param_templates import verification_mode_param
        return {
            "labels": {"default": "default", "type": "string"},
            **verification_mode_param(),
            "_init_start_": {"default": True, "type": "bool"},
        }

    @property
    def doc(self):
        return (
            "Oscilloscope / Plotter."
            "\n\nDisplays time-domain signals during or after simulation."
            "\n\nParameters:"
            "\n- Title: Plot window title."
            "\n- Labels: Comma-separated legend labels."
            "\n\nUsage:"
            "\nThe primary way to visualize simulation results."
            "\nDouble-click after simulation to re-open the plot."
        )

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return []

    def draw_icon(self, block_rect):
        """Draw oscilloscope icon in normalized 0-1 coordinates."""
        from PyQt5.QtGui import QPainterPath
        path = QPainterPath()
        path.moveTo(0.1, 0.9)
        path.lineTo(0.9, 0.9)  # x-axis
        path.moveTo(0.1, 0.9)
        path.lineTo(0.1, 0.1)  # y-axis
        path.moveTo(0.1, 0.6)
        path.quadTo(0.3, 0.2, 0.5, 0.6)
        path.quadTo(0.7, 1.0, 0.9, 0.6)
        return path

    def execute(self, time, inputs, params, **kwargs):
        """
        Collect input signals for plotting.
        """
        logger.debug(f"SCOPE EXECUTE: {self.block_name} time={time} inputs={inputs}")
        # To prevent saving data in wrong iterations (RK45 integration)
        if '_skip_' in params.keys() and params['_skip_']:
            params['_skip_'] = False
            return {0: np.array([0.0]), 'E': False}

        def _coerce(val):
            """Coerce any port value to a 1-d float array.

            Handles None (returns [0.0]) and 0-d numpy arrays (which np.concatenate
            rejects). Without this, an upstream block returning None or a stale
            params['vector'] from a prior fast-path run (which writes a Python
            list, not an ndarray) crashes scope.execute() with
            'zero-dimensional arrays cannot be concatenated'.
            """
            if val is None:
                return np.array([0.0])
            arr = np.atleast_1d(val).flatten()
            return arr if arr.size > 0 else np.array([0.0])

        # The fast-path replay loop writes a Python list to exec_params['vector']
        # after each simulation. If we land here on a later sim with a stale
        # non-ndarray vector, we cannot safely concatenate. Force re-init.
        prev_vec = params.get('vector', None)
        if not isinstance(prev_vec, np.ndarray) or prev_vec.ndim != 1:
            params['_init_start_'] = True

        # Initialization of the saving vector
        # Handle multiple input ports by concatenating all inputs
        if params.get('_init_start_', True):
            logger.debug(f"Scope {params.get('_name_', 'unknown')} initializing, inputs: {inputs}")
            combined_input = [_coerce(inputs[p]) for p in sorted(inputs.keys())]
            aux_vector = np.concatenate(combined_input) if combined_input else np.array([0.0])
            params['vec_dim'] = len(aux_vector)

            labels = params.get('labels', 'default')
            if labels == 'default':
                labels = params.get('_name_', 'scope') + '-0'
            labels = labels.replace(' ', '').split(',')
            if len(labels) >= params['vec_dim']:
                labels = labels[:params['vec_dim']]
            else:
                for i in range(len(labels), params['vec_dim']):
                    labels.append(params['_name_'] + '-' + str(i))
            params['vec_labels'] = labels
            params['_init_start_'] = False
            logger.debug(f"Scope {params.get('_name_', 'unknown')} initialized, vec_labels: {params['vec_labels']}")
        else:
            aux_vector = params['vector']
            combined_input = [_coerce(inputs[p]) for p in sorted(inputs.keys())]
            new_sample = np.concatenate(combined_input) if combined_input else np.array([0.0])
            # If per-sample dimension changed since last init (e.g. user added
            # a port between sims), re-initialize instead of producing a
            # ragged buffer.
            expected_dim = params.get('vec_dim', new_sample.size)
            if new_sample.size != expected_dim:
                aux_vector = new_sample
                params['vec_dim'] = new_sample.size
            else:
                aux_vector = np.concatenate((aux_vector, new_sample))
        params['vector'] = aux_vector
        return {0: np.array([0.0]), 'E': False}
