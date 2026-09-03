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
    def b_type(self):
        """Sink block - consumes output without producing further output."""
        return 3

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

    def draw_icon(self, block_rect):
        """Draw a ground symbol: a stub into three shrinking rails."""
        from PyQt5.QtGui import QPainterPath

        path = QPainterPath()
        path.moveTo(0.5, 0.2)
        path.lineTo(0.5, 0.6)
        path.moveTo(0.2, 0.6)
        path.lineTo(0.8, 0.6)
        path.moveTo(0.3, 0.75)
        path.lineTo(0.7, 0.75)
        path.moveTo(0.4, 0.9)
        path.lineTo(0.6, 0.9)
        return path

    def execute(self, time, inputs, params, **kwargs):
        return {}
