from blocks.tag_base import TagRoutingBlock


class FromBlock(TagRoutingBlock):
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
    def inputs(self):
        return []

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]
