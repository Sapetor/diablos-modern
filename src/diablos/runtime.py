"""Process-level runtime switches that sit between the engine and the GUI.

``ensure_headless_qapp`` serves the ``DIABLOS_HEADLESS_QAPP=1`` opt-in: an
offscreen QApplication for third-party user blocks that construct a
QPixmap/QFont/QWidget outside ``draw_icon`` (Qt refuses, and aborts, without
one). The engine calls it before loading blocks; the Qt part lives in
``diablos.gui.headless_qapp`` and is imported only when the opt-in is set,
so ``diablos.core`` never imports the GUI.
"""

import os


def ensure_headless_qapp():
    """The opt-in offscreen QApplication, or None when not requested."""
    if os.environ.get("DIABLOS_HEADLESS_QAPP") != "1":
        return None
    from diablos.gui.headless_qapp import offscreen_qapp

    return offscreen_qapp()
