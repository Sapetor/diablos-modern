"""
MenuBlocks class - represents blocks in the palette menu.
"""

from PyQt6.QtGui import QPixmap, QFont
from PyQt6.QtCore import Qt, QRect

from lib.app_paths import resource_path


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
        pixmap = QPixmap(resource_path(f"lib/icons/{self.block_fn.lower()}.png"))
        if not pixmap.isNull():
            self.image = pixmap.scaled(
                self.side_length[0],
                self.side_length[1],
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        else:
            self.image = pixmap
        self.external = external
        self.collision = None
        self.font = QFont("Arial", 10)
        self.block_class = block_class
        self.colors = colors

    def draw_menublock(self, painter, pos):
        # Lazy import to avoid circular dependency
        from lib.theming.theme_manager import theme_manager

        self.collision = QRect(40, 80 + 40 * pos, self.side_length[0], self.side_length[1])
        painter.fillRect(self.collision, self.b_color)
        if not self.image.isNull():
            painter.drawPixmap(self.collision.topLeft(), self.image)

        painter.setFont(self.font)
        painter.setPen(theme_manager.get_color("text_primary"))
        text_rect = QRect(90, 80 + 40 * pos, 100, 30)
        painter.drawText(
            text_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self.fn_name
        )

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
            from blocks.base_block import prettify_block_name

            return prettify_block_name(fallback)
        return str(fallback)

    def search_text(self, translated_name: str = "") -> str:
        """Lower-cased haystack for filters: display name, translation and ids."""
        parts = [self.display_name, translated_name, self.fn_name, self.block_fn]
        parts = [str(p).lower() for p in parts if p]
        return " ".join(parts + [p.replace(" ", "") for p in parts[:2]])
