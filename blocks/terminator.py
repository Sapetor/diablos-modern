
from blocks.base_block import BaseBlock

class TerminatorBlock(BaseBlock):
    """
    A block that terminates a signal.
    """

    @property
    def block_name(self):
        return "Term"

    @property
    def category(self):
        return "Sinks"

    @property
    def color(self):
        return "red"

    @property
    def doc(self):
        return (
            "Signal Terminator."
            "\n\nSafely terminates an unused output signal."
            "\n\nUsage:"
            "\nPrevents 'Unconnected Output' warnings during validation."
        )

    @property
    def params(self):
        return {}

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return []

    def execute(self, time, inputs, params):
        return {}
