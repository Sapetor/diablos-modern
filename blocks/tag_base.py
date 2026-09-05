"""Shared definition for the tag-routing block pair (Goto / From).

``Goto`` and ``From`` are the two ends of one wireless connection: they carry
the same ``tag``/``signal_name`` parameter pair, are drawn with the same
orange pentagon, and both forward whatever arrives on the routing line
untouched. Only the port direction differs, so only the port direction is
declared per block.
"""

from blocks.base_block import BaseBlock


class TagRoutingBlock(BaseBlock):
    """Abstract base for the tag-matched, wireless routing blocks."""

    @property
    def category(self):
        return "Routing"

    @property
    def color(self):
        return "orange"

    @property
    def shape(self):
        return "tag"

    @property
    def params(self):
        return {
            "tag": {"type": "string", "default": "A", "doc": "Tag name to link Goto/From."},
            "signal_name": {
                "type": "string",
                "default": "",
                "doc": "Optional label; defaults to tag when empty.",
            },
        }

    def draw_icon(self, block_rect):
        """Tag blocks render their tag text - handled in the DBlock switch."""
        return None

    def execute(self, time, inputs, params, **kwargs):
        """Forward the routed signal untouched.

        Both ends read port 0: ``Goto`` from the real wire feeding it, ``From``
        from the hidden virtual line the engine injects under the same key. The
        signal is passed through with no shape or dtype normalization so any
        signal type (an "any" port) survives the hop. The bare-int ``0`` default
        applies only when nothing is routed here - an unconnected Goto, or a
        From whose tag matches no Goto.
        """
        return {0: inputs.get(0, 0)}
