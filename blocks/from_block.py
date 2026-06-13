from blocks.base_block import BaseBlock


class FromBlock(BaseBlock):
    """
    Tag-based signal receiver (pulls from matching Goto tag).
    """

    @property
    def block_name(self):
        return "From"

    @property
    def fn_name(self):
        return "from_block"

    @property
    def category(self):
        return "Routing"

    @property
    def color(self):
        return "orange"

    @property
    def doc(self):
        return (
            "From Tag."
            "\n\nReceives a signal from a matching 'Goto' block."
            "\n\nParameters:"
            "\n- Tag: Identifier of the source 'Goto' block."
            "\n\nUsage:"
            "\nReduces visual clutter."
        )

    @property
    def params(self):
        return {
            "tag": {"type": "string", "default": "A", "doc": "Tag name to link Goto/From."},
            "signal_name": {"type": "string", "default": "", "doc": "Optional label; defaults to tag when empty."}
        }

    @property
    def inputs(self):
        return []

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    def draw_icon(self, block_rect):
        """From uses tag text rendering - handled in DBlock switch."""
        return None

    def execute(self, time, inputs, params, **kwargs):
        # Value will be fed via hidden virtual line into input_queue under key 0.
        # This is a pure pass-through routing block: the incoming signal is
        # forwarded untouched (no shape/dtype normalization) so any signal type
        # ("any" port) is preserved. The bare-int 0 default only applies when the
        # virtual routing line is absent (an unconnected/dangling From tag).
        return {0: inputs.get(0, 0)}

