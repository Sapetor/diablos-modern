"""Welcome / empty-state overlay for the diagram canvas.

A child widget of the canvas that is shown only while the diagram being edited
has no blocks. At the top level it offers a short welcome, the basic gestures
and a few curated example diagrams; inside an empty subsystem it shows a lighter
hint instead. Everything outside the real buttons is transparent to the mouse,
so double-click-to-add and palette drops on the canvas keep working.
"""

import logging
import os

from PyQt6.QtCore import QEvent, Qt, QTimer
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from lib.app_paths import resource_path
from lib.i18n import tr
from modern_ui.themes.theme_manager import RADIUS, SPACE, TYPE, theme_manager

logger = logging.getLogger(__name__)

MODE_HIDDEN = "hidden"
MODE_WELCOME = "welcome"
MODE_SUBSYSTEM = "subsystem"


def curated_examples():
    """(file name, title, description) for the example cards, in display order."""
    return [
        (
            "first_order_step_response.diablos",
            tr("First-order step response"),
            tr("A step input driving a first-order lag"),
        ),
        (
            "pid_second_order.diablos",
            tr("PID control loop"),
            tr("Feedback control of a second-order plant"),
        ),
        (
            "pendulum_nonlinear_vs_linear.diablos",
            tr("Nonlinear pendulum"),
            tr("Nonlinear dynamics vs. their linearization"),
        ),
    ]


class WelcomeOverlay(QWidget):
    """Theme-aware empty-state overlay, parented to (and sized to) the canvas."""

    def __init__(self, canvas, window):
        super().__init__(canvas)
        self.canvas = canvas
        self.window_ref = window
        self.mode = MODE_HIDDEN
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setObjectName("welcomeOverlay")

        outer = QVBoxLayout(self)
        outer.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.panel = QFrame()
        self.panel.setObjectName("welcomePanel")
        self.panel.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        outer.addWidget(self.panel, 0, Qt.AlignmentFlag.AlignCenter)
        lay = QVBoxLayout(self.panel)
        lay.setContentsMargins(SPACE["2xl"], SPACE["2xl"], SPACE["2xl"], SPACE["2xl"])
        lay.setSpacing(SPACE["md"])

        self.title_label = QLabel()
        self.title_label.setObjectName("welcomeTitle")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hint_label = QLabel()
        self.hint_label.setObjectName("welcomeHint")
        self.hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hint_label.setWordWrap(True)
        lay.addWidget(self.title_label)
        lay.addWidget(self.hint_label)

        # Example cards + actions live in one container so the subsystem hint
        # can hide them together.
        self.actions_box = QWidget()
        self.actions_box.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        box = QVBoxLayout(self.actions_box)
        box.setContentsMargins(0, SPACE["lg"], 0, 0)
        box.setSpacing(SPACE["md"])
        self.examples_label = QLabel(tr("Start from an example"))
        self.examples_label.setObjectName("welcomeSection")
        self.examples_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        box.addWidget(self.examples_label)

        cards = QHBoxLayout()
        cards.setSpacing(SPACE["lg"])
        self.example_buttons = []
        for fname, title, desc in curated_examples():
            btn = self._make_card(title, desc)
            btn.setProperty("example_file", fname)
            btn.clicked.connect(lambda _=False, f=fname: self.open_example(f))
            cards.addWidget(btn)
            self.example_buttons.append(btn)
        box.addLayout(cards)

        row = QHBoxLayout()
        row.setSpacing(SPACE["md"])
        row.addStretch(1)
        self.open_button = QPushButton(tr("Open…"))
        self.open_button.setObjectName("welcomeAction")
        self.open_button.clicked.connect(self._open_file)
        self.browse_button = QPushButton(tr("Browse examples"))
        self.browse_button.setObjectName("welcomeAction")
        self.browse_button.clicked.connect(self._browse_examples)
        row.addWidget(self.open_button)
        row.addWidget(self.browse_button)
        row.addStretch(1)
        box.addLayout(row)
        lay.addWidget(self.actions_box)

        self._apply_styling()
        theme_manager.theme_changed.connect(self._apply_styling)
        canvas.installEventFilter(self)
        self.hide()
        self.refresh()

    # -- construction helpers ------------------------------------------------

    @staticmethod
    def _make_card(title, desc):
        btn = QPushButton()
        btn.setObjectName("welcomeCard")
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setMinimumWidth(170)
        btn.setMaximumWidth(210)
        lay = QVBoxLayout(btn)
        lay.setContentsMargins(SPACE["lg"], SPACE["lg"], SPACE["lg"], SPACE["lg"])
        lay.setSpacing(SPACE["xs"])
        t = QLabel(title)
        t.setObjectName("cardTitle")
        t.setWordWrap(True)
        d = QLabel(desc)
        d.setObjectName("cardDesc")
        d.setWordWrap(True)
        for lbl in (t, d):
            lbl.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        lay.addWidget(t)
        lay.addWidget(d)
        lay.addStretch(1)
        # Height follows the font (so it scales with UI scale), not a pixel literal.
        line = t.fontMetrics().lineSpacing()
        btn.setMinimumHeight(line * 5 + SPACE["lg"] * 2)
        return btn

    def _apply_styling(self, *_):
        c = theme_manager.get_color
        pt = TYPE
        self.setStyleSheet(f"""
            #welcomePanel {{
                background-color: {c("surface_elevated").name()};
                border: 1px solid {c("border_primary").name()};
                border-radius: {RADIUS["lg"] * 2}px;
            }}
            QLabel {{ background: transparent; }}
            #welcomeTitle {{
                color: {c("text_primary").name()};
                font-size: {pt["heading"] + 4}pt; font-weight: 600;
            }}
            #welcomeHint {{ color: {c("text_secondary").name()}; font-size: {pt["subtitle"]}pt; }}
            #welcomeSection {{
                color: {c("text_secondary").name()};
                font-size: {pt["body"]}pt; font-weight: 600;
            }}
            #welcomeCard, #welcomeAction {{
                background-color: {c("surface_primary").name()};
                border: 1px solid {c("border_primary").name()};
                border-radius: {RADIUS["lg"]}px;
                text-align: left;
            }}
            #welcomeAction {{
                color: {c("text_primary").name()};
                font-size: {pt["body_strong"]}pt;
                padding: {SPACE["md"]}px {SPACE["xl"]}px;
            }}
            #welcomeCard:hover, #welcomeAction:hover, #welcomeCard:focus, #welcomeAction:focus {{
                border: 1px solid {c("accent_primary").name()};
            }}
            #cardTitle {{
                color: {c("text_primary").name()};
                font-size: {pt["body_strong"]}pt; font-weight: 600;
            }}
            #cardDesc {{ color: {c("text_secondary").name()}; font-size: {pt["body"]}pt; }}
        """)

    # -- state ----------------------------------------------------------------

    def desired_mode(self):
        dsim = getattr(self.canvas, "dsim", None)
        if dsim is None or getattr(dsim, "blocks_list", None):
            return MODE_HIDDEN
        if getattr(dsim, "current_subsystem", None):
            return MODE_SUBSYSTEM
        return MODE_WELCOME

    def schedule_refresh(self):
        """Cheap check for the canvas paint path; defers the real work."""
        if self.desired_mode() != self.mode:
            QTimer.singleShot(0, self.refresh)

    def refresh(self):
        mode = self.desired_mode()
        self.mode = mode
        if mode == MODE_HIDDEN:
            self.hide()
            return
        welcome = mode == MODE_WELCOME
        self.title_label.setVisible(welcome)
        self.actions_box.setVisible(welcome)
        self.title_label.setText(tr("Welcome to DiaBloS"))
        if welcome:
            self.hint_label.setText(
                tr("Double-click the canvas to add a block, or drag one from the palette.")
            )
        else:
            self.hint_label.setText(
                tr(
                    "This subsystem is empty. Double-click to add a block, or press Esc to go back up."
                )
            )
        self.setGeometry(self.canvas.rect())
        self.show()
        self.raise_()

    def eventFilter(self, obj, event):
        if obj is self.canvas and event.type() == QEvent.Type.Resize:
            self.setGeometry(self.canvas.rect())
        return False

    # -- actions --------------------------------------------------------------

    def open_example(self, fname):
        path = os.path.join(resource_path("examples"), fname)
        self.window_ref.open_example(path)

    def _open_file(self):
        self.window_ref.open_diagram()

    def _browse_examples(self):
        menu = QMenu(self)
        self.window_ref.menu_builder._populate_examples_menu(menu)
        menu.exec(self.browse_button.mapToGlobal(self.browse_button.rect().bottomLeft()))
