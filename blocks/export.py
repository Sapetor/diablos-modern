from blocks.base_block import BaseBlock
import numpy as np


class ExportBlock(BaseBlock):
    def __init__(self):
        super().__init__()

    @property
    def block_name(self):
        return "Export"

    @property
    def category(self):
        return "Sinks"

    @property
    def color(self):
        return "red"

    @property
    def params(self):
        return {
            "str_name": {"default": "default", "type": "string"},
            "_init_start_": {"default": True, "type": "bool"},
        }

    @property
    def doc(self):
        return "Exports input signals to a file."

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return []

    def draw_icon(self, block_rect):
        """Draw export/file icon in normalized 0-1 coordinates."""
        from PyQt5.QtGui import QPainterPath
        path = QPainterPath()
        # Box
        path.moveTo(0.2, 0.2)
        path.lineTo(0.8, 0.2)
        path.lineTo(0.8, 0.8)
        path.lineTo(0.2, 0.8)
        path.lineTo(0.2, 0.2)
        # Arrow out
        path.moveTo(0.5, 0.5)
        path.lineTo(1.0, 0.5)
        path.moveTo(0.8, 0.3)
        path.lineTo(1.0, 0.5)
        path.lineTo(0.8, 0.7)
        return path

    def execute(self, time, inputs, params):
        """
        Save and export block signals.
        """
        # To prevent saving data in wrong iterations (RK45 integration)
        if '_skip_' in params.keys() and params['_skip_']:
            params['_skip_'] = False
            return {0: np.array([0.0]), 'E': False}

        # Initialization of the saving vector
        if params.get('_init_start_', True):
            aux_vector = np.array([inputs[0]])
            try:
                params['vec_dim'] = len(inputs[0])
            except TypeError:
                params['vec_dim'] = 1

            labels = params['str_name']
            if labels == 'default':
                labels = params['_name_'] + '-0'
            labels = labels.replace(' ', '').split(',')
            if len(labels) < params['vec_dim']:
                for i in range(params['vec_dim'] - len(labels)):
                    labels.append(params['_name_'] + '-' + str(params['vec_dim'] + i - 1))
            elif len(labels) > params['vec_dim']:
                labels = labels[:params['vec_dim']]
            if len(labels) == params['vec_dim'] == 1:
                labels = labels[0]
            params['vec_labels'] = labels
            params['_init_start_'] = False
        else:
            aux_vector = params['vector']
            aux_vector = np.concatenate((aux_vector, [inputs[0]]))
        params['vector'] = aux_vector
        return {0: np.array([0.0]), 'E': False}
