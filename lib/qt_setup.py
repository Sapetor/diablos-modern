"""Process-wide Qt environment, set before the first Qt import.

Every entry point that builds a QApplication (the GUI in ``diablos_modern.py``,
the headless CLI in ``lib.cli`` and the test suite in ``tests/conftest.py``)
calls :func:`configure_qt_env` first. It deliberately imports nothing from Qt,
pyqtgraph or matplotlib: the variables it sets are only read when those
libraries are imported.
"""

import os


def configure_qt_env(headless: bool = False) -> None:
    """Pin the Qt binding and, optionally, force headless backends.

    ``PYQTGRAPH_QT_LIB``: pyqtgraph picks a binding the first time it is
    imported. A developer environment can have PyQt5 installed next to PyQt6,
    and two bindings in one process segfault, so always name PyQt6.

    ``headless``: select Qt's ``offscreen`` platform plugin and matplotlib's
    ``Agg`` backend so no window can appear. Both are ``setdefault`` so an
    explicit value in the shell still wins.
    """
    os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt6")
    if headless:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        os.environ.setdefault("MPLBACKEND", "Agg")
