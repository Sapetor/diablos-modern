from blocks.base_block import BaseBlock


class GotoBlock(BaseBlock):
    """
    Tag-based signal jumper (sends signal to matching From with same tag).
    """

    @property
    def block_name(self):
        return "Goto"

    @property
    def category(self):
        return "Routing"

    @property
    def color(self):
        return "orange"

    @property
    def doc(self):
        return "Sends its input to any From block with the same tag."

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
        return {0: inputs.get(0, 0)}
