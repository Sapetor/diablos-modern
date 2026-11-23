"""DiaBloS: block-diagram modelling and simulation.

Subpackages:

* ``diablos.core``   -- the engine (numpy + scipy only)
* ``diablos.blocks`` -- the built-in block library
* ``diablos.gui``    -- the PyQt6 application (``pip install 'diablos-modern[gui]'``)

Scripts use :mod:`diablos.api` (``from diablos.api import load``).
"""

from diablos.compat import install_moved_module_aliases as _install_moved_aliases

_install_moved_aliases()


def __getattr__(name):
    if name == "__version__":
        from diablos.gui import __version__  # version logic only; imports no Qt

        return __version__
    raise AttributeError(f"module 'diablos' has no attribute {name!r}")
