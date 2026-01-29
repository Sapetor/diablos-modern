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
    def color(self):
        return "red"

    @property
    def params(self):
        return {
            "labels": {"default": "default", "type": "string"},
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

    def execute(self, time, inputs, params):
        """
        Collect input signals for plotting.
        """
        logger.debug(f"SCOPE EXECUTE: {self.block_name} time={time} inputs={inputs}")
        # To prevent saving data in wrong iterations (RK45 integration)
        if '_skip_' in params.keys() and params['_skip_']:
            params['_skip_'] = False
            return {0: np.array([0.0]), 'E': False}

        # Initialization of the saving vector
        if params.get('_init_start_', True):
            logger.debug(f"Scope {params.get('_name_', 'unknown')} initializing, inputs: {inputs}")
            aux_vector = np.atleast_1d(inputs[0])
            try:
                params['vec_dim'] = len(inputs[0])
            except TypeError:
                params['vec_dim'] = 1

            labels = params['labels']
            if labels == 'default':
                labels = params['_name_'] + '-0'
            labels = labels.replace(' ', '').split(',')
            if len(labels) - params['vec_dim'] >= 0:
                labels = labels[:params['vec_dim']]
            elif len(labels) - params['vec_dim'] < 0:
                for i in range(len(labels), params['vec_dim']):
                    labels.append(params['_name_'] + '-' + str(i))
            elif len(labels) == params['vec_dim'] == 1:
                labels = labels[0]
            params['vec_labels'] = labels
            params['_init_start_'] = False
            logger.debug(f"Scope {params.get('_name_', 'unknown')} initialized, vec_labels: {params['vec_labels']}")
        else:
            aux_vector = params['vector']
            aux_vector = np.concatenate((aux_vector, np.atleast_1d(inputs[0])))
        params['vector'] = aux_vector
        return {0: np.array([0.0]), 'E': False}
