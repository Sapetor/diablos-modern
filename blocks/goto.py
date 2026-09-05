from blocks.tag_base import TagRoutingBlock


class GotoBlock(TagRoutingBlock):
    """
    Tag-based signal jumper (sends signal to matching From with same tag).
    """

    @property
    def block_name(self):
        return "Goto"

    @property
    def fn_name(self):
        return "goto_block"

    @property
    def doc(self):
        return (
            "Goto Tag."
            "\n\nSends a signal to a matching 'From' block without a visible wire."
            "\n\nParameters:"
            "\n- Tag: Unique identifier (string) to match with 'From'."
            "\n\nUsage:"
            "\nReduces visual clutter by hiding long connections."
        )

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return []
