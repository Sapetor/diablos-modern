from blocks.base_block import BaseBlock

class MuxBlock(BaseBlock):
    def __init__(self):
        super().__init__()

    @property
    def block_name(self):
        return "Mux"

    @property
    def category(self):
        return "Routing"

    @property
    def color(self):
        return "orange"

    @property
    def params(self):
        return {}

    @property
    def doc(self):
        return "Multiplexes multiple inputs into a single output vector."

    @property
    def inputs(self):
        return [{"name": "in1", "type": "any"}, {"name": "in2", "type": "any"}]

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    def execute(self, time, inputs, params):
        return {0: list(inputs.values())}
