"""
Button UI component for DiaBloS.
"""

from PyQt5.QtGui import QColor, QFont, QPen
from PyQt5.QtCore import QRect, Qt


class Button:
    """UI Button widget for toolbar actions."""

    def __init__(self, name, coords, active=True):
        self.name = name
        self.text = name.strip('_')  # Remove underscores for display
        self.collision = QRect(*coords) if isinstance(coords, tuple) else coords
        self.pressed = False
        self.active = active
        self.font = QFont()
        self.font.setPointSize(12)  # Adjust font size as needed
        self.collision = QRect(*coords)

    def draw_button(self, painter):
        if painter is None:
            return
        if not self.active:
            bg_color = QColor(240, 240, 240)
            text_color = QColor(128, 128, 128)
        else:
            bg_color = QColor(200, 200, 200) if self.pressed else QColor(220, 220, 220)
            text_color = QColor(0, 0, 0)

        painter.setBrush(bg_color)
        painter.setPen(Qt.NoPen)
        painter.drawRect(self.collision)

        painter.setFont(self.font)
        painter.setPen(text_color)
        painter.drawText(self.collision, Qt.AlignCenter, self.text)

        if self.active:
            painter.setPen(QPen(QColor(100, 100, 100), 1))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(self.collision)

    def contains(self, point):
        return self.collision.contains(point)
