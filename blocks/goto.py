from blocks.base_block import BaseBlock


class GotoBlock(BaseBlock):
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
    def category(self):
        return "Routing"

    @property
    def color(self):
        return "orange"

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
    def params(self):
        return {
            "tag": {"type": "string", "default": "A", "doc": "Tag name to link Goto/From."},
            "signal_name": {"type": "string", "default": "", "doc": "Optional label; defaults to tag when empty."}
        }

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return []

    def draw_icon(self, block_rect):
        """Goto uses tag text rendering - handled in DBlock switch."""
        return None

    def execute(self, time, inputs, params):
        return {0: inputs.get(0, 0)}

