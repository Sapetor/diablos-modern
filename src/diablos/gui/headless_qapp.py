"""Offscreen QApplication for headless runs that opt in (see diablos.runtime)."""

import os

# Strong process-lifetime reference: without it the Python wrapper can be
# collected while Qt objects still exist, and PyQt crashes tearing them down
# at shutdown.
_QAPP = None


def offscreen_qapp():
    global _QAPP
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    _QAPP = QApplication.instance() or QApplication(["diablos-run"])
    return _QAPP
