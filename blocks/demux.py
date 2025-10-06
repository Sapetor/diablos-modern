
import numpy as np
from blocks.base_block import BaseBlock

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
        return "Splits an input vector into multiple outputs."

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

    def execute(self, time, inputs, params):
        # Check input dimensions first
        if len(inputs[0]) / params['output_shape'] < len(self.outputs):
            print("ERROR: Not enough inputs or wrong output shape in", params['_name_'])
            return {'E': True}

        elif len(inputs[0]) / params['output_shape'] > len(self.outputs):
            print("WARNING: There are more elements in vector for the expected outputs. System will truncate. Block", params['_name_'])

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
            print(f"ERROR: Invalid input type in demux block. Expected numeric.")
            return {'E': True}
