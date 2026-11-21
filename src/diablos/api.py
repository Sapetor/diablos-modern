"""DiaBloS scripting API: load a diagram, run it, read and analyse the results.

    from diablos.api import load

    sim = load("examples/first_order_step_response.diablos")
    result = sim.run(t=5.0)
    result["output"]          # a Scope channel as a numpy array

Needs only numpy and scipy (no PyQt6). The implementation lives in
:mod:`diablos.core.api`; this module is the stable public name. See
docs/user-guide/python-api.md for the full reference.
"""

from diablos.core.api import Diagram, SimResult, SimulationError, from_builder, load

__all__ = ["Diagram", "SimResult", "SimulationError", "from_builder", "load"]
