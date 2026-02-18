import logging
import numpy as np
from blocks.base_block import BaseBlock

logger = logging.getLogger(__name__)

class DemuxBlock(BaseBlock):
    """
    A block that splits an input vector into multiple outputs.
    """

    @property
    def block_name(self):
        return "Demux"

    @property
    def category(self):
        return "Routing"

    @property
    def color(self):
        return "orange"

    @property
    def doc(self):
        return (
            "Demultiplexer (Demux)."
            "\n\nSplits a vector input signal into individual scalar/vector components."
            "\n\nParameters:"
            "\n- Outputs: Number of output ports."
            "\n\nUsage:"
            "\nUse to extract signals from a bus or Mux."
        )

    @property
    def io_editable(self):
        return 'output'

    @property
    def params(self):
        return {
            "output_shape": {"type": "int", "default": 1, "doc": "The size of each output vector."}
        }

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return [{"name": "out1", "type": "any"}, {"name": "out2", "type": "any"}]

    def draw_icon(self, block_rect):
        """Draw demultiplexer icon in normalized 0-1 coordinates."""
        from PyQt5.QtGui import QPainterPath
        path = QPainterPath()
        # Input
        path.moveTo(0.2, 0.5); path.lineTo(0.4, 0.5)
        # Main body (rectangle)
        path.moveTo(0.4, 0.2)
        path.lineTo(0.4, 0.8)
        path.lineTo(0.8, 0.8)
        path.lineTo(0.8, 0.2)
        path.lineTo(0.4, 0.2)
        # Output lines
        path.moveTo(0.8, 0.3); path.lineTo(1.0, 0.3)
        path.moveTo(0.8, 0.7); path.lineTo(1.0, 0.7)
        return path

    def execute(self, time, inputs, params, **kwargs):
        # Check input dimensions first
        if len(inputs[0]) / params['output_shape'] < len(self.outputs):
            return {'E': True, 'error': f"Not enough inputs or wrong output shape in {params.get('_name_', 'Demux')}"}

        elif len(inputs[0]) / params['output_shape'] > len(self.outputs):
            logger.warning(f"More elements in vector than expected outputs. Truncating. Block: {params.get('_name_', 'Demux')}")

        try:
            input_array = np.array(inputs[0], dtype=float).flatten()
            output_shape = int(params['output_shape'])
            outputs = {}
            for i in range(len(self.outputs)):
                start = i * output_shape
                end = start + output_shape
                outputs[i] = input_array[start:end]
            return outputs
        except (ValueError, TypeError):
            return {'E': True, 'error': 'Invalid input type in demux block. Expected numeric.'}

