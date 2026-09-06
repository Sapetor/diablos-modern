"""Fixtures for the validation suite.

Every case builds one or more ``DSim`` objects, each owning pyqtgraph-backed
plotting state. Left to CPython's own timing that state is freed at whatever
arbitrary later allocation triggers a collection -- and a pyqtgraph item deleted
from under a widget that is mid-construction segfaults the interpreter, so a
module that leaks a few dozen of them shows up as a crash somewhere else
entirely. Collecting on the way out of each test keeps the deletions here, where
nothing is being built.
"""

import gc

import pytest


@pytest.fixture(autouse=True)
def _collect_dsims():
    yield
    gc.collect()


@pytest.fixture(autouse=True)
def _qt_ready(qapp):
    """Bind the session ``QApplication`` before any block or DSim is built."""
    return qapp
