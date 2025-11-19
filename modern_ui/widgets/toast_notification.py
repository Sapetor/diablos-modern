"""Toast Notification Widget
Brief visual feedback for keyboard shortcuts and actions.
"""

from PyQt5.QtWidgets import QLabel, QGraphicsOpacityEffect
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QFont
from modern_ui.themes.theme_manager import theme_manager


class ToastNotification(QLabel):
    """A brief notification message that fades in and out."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Tool | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)

        # Styling
        self.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumWidth(200)
        self.setMaximumWidth(400)
        self.setWordWrap(True)

        # Opacity effect
        self.opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity_effect)

        # Animation
        self.fade_animation = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.fade_animation.setEasingCurve(QEasingCurve.InOutQuad)

        # Auto-hide timer
        self.hide_timer = QTimer(self)
        self.hide_timer.setSingleShot(True)
        self.hide_timer.timeout.connect(self._start_fade_out)

        self._apply_styling()

        # Connect to theme changes
        theme_manager.theme_changed.connect(self._apply_styling)

    def _apply_styling(self):
        """Apply theme-aware styling."""
        bg_color = theme_manager.get_color('surface_elevated')
        text_color = theme_manager.get_color('text_primary')
        border_color = theme_manager.get_color('accent_primary')

        self.setStyleSheet(f"""
            QLabel {{
                background-color: {bg_color.name()};
                color: {text_color.name()};
                border: 2px solid {border_color.name()};
                border-radius: 8px;
                padding: 12px 20px;
            }}
        """)

    def show_message(self, message: str, duration: int = 2000):
        """
        Show a toast notification with a message.

        Args:
            message: The message to display
            duration: How long to show the message in milliseconds (default 2000ms)
        """
        self.setText(message)
        self.adjustSize()

        # Position in bottom-right of parent
        if self.parent():
            parent_rect = self.parent().rect()
            x = parent_rect.width() - self.width() - 20
            y = parent_rect.height() - self.height() - 20
            self.move(x, y)

        # Fade in
        self.opacity_effect.setOpacity(0.0)
        self.show()

        self.fade_animation.stop()
        self.fade_animation.setDuration(300)
        self.fade_animation.setStartValue(0.0)
        self.fade_animation.setEndValue(0.95)
        self.fade_animation.start()

        # Schedule fade out
        self.hide_timer.start(duration)

    def _start_fade_out(self):
        """Start the fade out animation."""
        self.fade_animation.stop()
        self.fade_animation.setDuration(300)
        self.fade_animation.setStartValue(0.95)
        self.fade_animation.setEndValue(0.0)
        self.fade_animation.finished.connect(self.hide)
        self.fade_animation.start()
