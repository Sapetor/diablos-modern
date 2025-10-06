from blocks.base_block import BaseBlock
from lib import functions

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

    def execute(self, time, inputs, params):
        return functions.export(time, inputs, params)
