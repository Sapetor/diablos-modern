"""Run DiaBloS from a source checkout.

    python diablos_modern.py                         # the application
    python diablos_modern.py run diagram.diablos     # headless (see diablos.main)

An installed DiaBloS (``pip install 'diablos-modern[gui]'``) provides the same
entry point as the ``diablos`` command and ``python -m diablos``. This file is
also the PyInstaller entry script (``diablos.spec``).
"""

import multiprocessing
import os
import sys

# Before anything heavy: a frozen build's worker processes re-enter here.
multiprocessing.freeze_support()

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from diablos.main import entry  # noqa: E402

if __name__ == "__main__":
    sys.exit(entry())
