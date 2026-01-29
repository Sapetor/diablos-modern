"""
MenuBlocks class - represents blocks in the palette menu.
"""

from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt, QRect


class MenuBlocks:
    """Represents a block template in the block palette."""

    def __init__(self, block_fn, fn_name, io_params, ex_params, b_color, coords, external=False, block_class=None, colors=None):
        self.block_fn = block_fn
        self.fn_name = fn_name
        self.ins = io_params['inputs']
        self.outs = io_params['outputs']
        self.b_type = io_params['b_type']
        self.io_edit = io_params['io_edit']
        self.params = ex_params
        self.b_color = b_color
        self.size = coords
        self.side_length = (30, 30)
        pixmap = QPixmap(f'./lib/icons/{self.block_fn.lower()}.png')
        if not pixmap.isNull():
            self.image = pixmap.scaled(self.side_length[0], self.side_length[1], Qt.KeepAspectRatio, Qt.SmoothTransformation)
        else:
            self.image = pixmap
        self.external = external
        self.collision = None
        self.font = QFont('Arial', 10)
        self.block_class = block_class
        self.colors = colors

    def draw_menublock(self, painter, pos):
        # Lazy import to avoid circular dependency
        from modern_ui.themes.theme_manager import theme_manager

        self.collision = QRect(40, 80 + 40*pos, self.side_length[0], self.side_length[1])
        painter.fillRect(self.collision, self.b_color)
        if not self.image.isNull():
            painter.drawPixmap(self.collision.topLeft(), self.image)

        painter.setFont(self.font)
        painter.setPen(theme_manager.get_color('text_primary'))
        text_rect = QRect(90, 80 + 40*pos, 100, 30)
        painter.drawText(text_rect, Qt.AlignLeft | Qt.AlignVCenter, self.fn_name)
