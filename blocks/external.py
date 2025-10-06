from blocks.base_block import BaseBlock

class ExternalBlock(BaseBlock):
    def __init__(self):
        super().__init__()

    @property
    def block_name(self):
        return "External"

    @property
    def category(self):
        return "Other"

    @property
    def color(self):
        return "light_gray"

    @property
    def params(self):
        return {
            "filename": {"default": "<no filename>", "type": "string"},
        }

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    def execute(self, time, inputs, params):
        # This block executes an external function, which is handled in the main loop
        pass
