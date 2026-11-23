"""The self-test's window check (``diablos self-test`` without ``--no-gui``).

Opens the main window offscreen, loads a bundled example and paints the
canvas. The engine's runner (``diablos.core.self_test``) receives this from
the launcher; it has no window of its own.
"""

import os
import sys


def check_main_window(example_path, example_name):
    from diablos.core.qt_setup import configure_qt_env

    if not os.environ.get("QT_QPA_PLATFORM"):
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
    configure_qt_env()
    from PyQt6.QtWidgets import QApplication

    from diablos.gui.main_window import ModernDiaBloSWindow

    app = QApplication.instance() or QApplication([sys.argv[0]])
    window = ModernDiaBloSWindow()
    window.show()
    app.processEvents()
    # The Examples-menu route: no recent-files entry, no trust prompt.
    if window.open_example(example_path) is None:
        raise AssertionError(f"could not open {example_name}")
    app.processEvents()
    n_blocks = len(window.dsim.blocks_list)
    image = window.canvas.grab()
    if n_blocks == 0 or image.isNull():
        raise AssertionError(f"{n_blocks} blocks, canvas image null={image.isNull()}")
    # hide(), not close(): closeEvent deletes the user's autosave file.
    window.dsim.dirty = False
    window.hide()
    app.processEvents()
    return f"main window opened {example_name} ({n_blocks} blocks) and painted the canvas"
