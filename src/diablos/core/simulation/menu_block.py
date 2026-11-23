"""
MenuBlocks class - represents blocks in the palette menu.
"""


class MenuBlocks:
    """Represents a block template in the block palette."""

    def __init__(
        self,
        block_fn,
        fn_name,
        io_params,
        ex_params,
        b_color,
        coords,
        external=False,
        block_class=None,
        colors=None,
    ):
        self.block_fn = block_fn
        self.fn_name = fn_name
        self.ins = io_params["inputs"]
        self.outs = io_params["outputs"]
        self.b_type = io_params["b_type"]
        self.io_edit = io_params["io_edit"]
        self.params = ex_params
        self.b_color = b_color
        self.size = coords
        self.side_length = (30, 30)
        self.external = external
        self.collision = None
        self.block_class = block_class
        self.colors = colors

    # The palette draws its own tiles and icons (diablos.gui.widgets.
    # modern_palette); templates carry metadata only. The old QPixmap/QFont
    # attributes and draw_menublock() were unused and tied the engine to Qt.

    # -- Display name (presentation only; fn_name/block_fn stay the keys) ----

    @property
    def display_name(self) -> str:
        """English human-readable label for palette/command-palette entries.

        Built-in/user blocks use their class's ``display_name`` hook; library
        blocks (no class) keep their user-chosen name. Callers translate with
        ``tr()`` at display time.
        """
        cached = getattr(self, "_display_name", None)
        if cached is None:
            cached = self._compute_display_name()
            self._display_name = cached
        return cached

    def _compute_display_name(self) -> str:
        fallback = self.fn_name if self.block_class is None else self.block_fn
        if self.block_class is not None:
            try:
                return str(self.block_class().display_name) or str(fallback)
            except Exception:
                pass
            from diablos.blocks.base_block import prettify_block_name

            return prettify_block_name(fallback)
        return str(fallback)

    def search_text(self, translated_name: str = "") -> str:
        """Lower-cased haystack for filters: display name, translation and ids."""
        parts = [self.display_name, translated_name, self.fn_name, self.block_fn]
        parts = [str(p).lower() for p in parts if p]
        return " ".join(parts + [p.replace(" ", "") for p in parts[:2]])
