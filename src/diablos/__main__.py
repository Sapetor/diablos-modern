"""``python -m diablos``: the same entry point as the ``diablos`` command."""

import sys

from diablos.main import entry

if __name__ == "__main__":
    sys.exit(entry())
