from blocks.base_block import BaseBlock


class FromBlock(BaseBlock):
    """
    Tag-based signal receiver (pulls from matching Goto tag).
    """

    @property
    def block_name(self):
        return "From"

    @property
    def category(self):
        return "Routing"

    @property
    def color(self):
        return "orange"

    @property
    def doc(self):
        return "Receives signal from a Goto block with the same tag."

    @property
    def params(self):
        return {
            "tag": {"type": "string", "default": "A", "doc": "Tag name to link Goto/From."}
        }

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    def execute(self, time, inputs, params):
        # Pass through received value
        return {0: inputs.get(0, 0)}
