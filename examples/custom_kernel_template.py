"""Template for a custom block that also carries a compiled-path *kernel*.

Two execution paths exist (see ``docs/FAST_SOLVER.md``):

* the **interpreted** path calls ``execute()`` once per block per time step.
  Every block works there; that is the only path a user block needs;
* the **compiled** path (fast solver) turns the whole diagram into one ODE
  right-hand side for ``scipy.integrate.solve_ivp``.  A block joins it by
  registering a *kernel builder* -- a function that bakes the block's
  parameters into a small closure ``f(t, y, dy_vec, signals)``.

Registering a kernel is what this file shows.  Read the caveat in
``docs/BLOCK_API.md`` first: the compiler additionally gates a diagram on
``SystemCompiler.COMPILABLE_BLOCKS``, an allowlist that a user module cannot
extend, so today a diagram containing a user block always runs on the
interpreted path.  ``execute()`` therefore stays the source of truth, and the
kernel below is what a block contributed to the engine (or a vendored build)
would look like.

Nothing here is required for a normal custom block -- start from
``custom_block_template.py`` instead.
"""

import numpy as np

from blocks.base_block import BaseBlock
from lib.engine.compiler_kernels import EventSpec, events, kernel, signal_scalar

BLOCK_API_VERSION = 1


class SoftClipBlock(BaseBlock):
    """``y = clip(u, -limit, +limit)`` -- a saturation with a single limit."""

    @property
    def block_name(self):
        return "SoftClip"

    @property
    def category(self):
        return "Math"

    @property
    def doc(self):
        return "Symmetric saturation: y = clip(u, -limit, +limit)."

    @property
    def params(self):
        return {
            "limit": {"type": "float", "default": 1.0, "doc": "Symmetric clip level"},
        }

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    def execute(self, time, inputs, params, **kwargs):
        limit = abs(float(params.get("limit", 1.0)))
        u = np.atleast_1d(inputs.get(0, 0.0))
        return {0: np.clip(u, -limit, limit)}


# --------------------------------------------------------------------------- #
# Compiled path
#
# The name passed to @kernel is the *canonical* fn-name:
# ``lib.engine.block_names.canonical_fn(block_name)``, which is essentially
# ``block_name.title()`` plus a few historical overrides.  "SoftClip" -> in
# doubt, print canonical_fn("SoftClip") once rather than guessing.
# --------------------------------------------------------------------------- #


@kernel("Softclip")
def build_soft_clip(ctx):
    """Return the closure the compiled solver calls for one SoftClip block.

    ``ctx`` is a ``BuildContext``: ``b_name`` is this block's signal key,
    ``params`` its parameters, ``input_sources[i]`` the signal key feeding
    input port ``i`` (None when unconnected).  Do every lookup and conversion
    *here*, at build time -- the returned closure runs thousands of times per
    solve and must stay allocation-free.
    """
    b_name = ctx.b_name
    src = ctx.input_sources[0] if ctx.input_sources else None
    limit = abs(float(ctx.params.get("limit", 1.0)))

    def exec_soft_clip(t, y, dy_vec, signals):
        # signals is the shared per-step dict of block outputs; write this
        # block's output under its own name. `y`/`dy_vec` are the state vector
        # and its derivative -- only stateful kernels touch them.
        value = signals.get(src, 0.0) if src else 0.0
        signals[b_name] = min(max(value, -limit), limit)

    return exec_soft_clip


@events("Softclip")
def events_soft_clip(ctx):
    """Declare the kinks so the solver lands exactly on each clip instant.

    An event is a smooth scalar ``g(t, y, signals)`` whose sign change marks
    the discontinuity.  Returning ``[]`` is fine -- the block then simply runs
    without located events, which for a hard nonlinearity means the adaptive
    solver smears the corner over a step or two.
    """
    src = ctx.input_sources[0] if ctx.input_sources else None
    if not src:
        return []
    limit = abs(float(ctx.params.get("limit", 1.0)))
    if not np.isfinite(limit):
        return []

    specs = []
    for label, level in (("lower_limit", -limit), ("upper_limit", limit)):

        def _g(t, y, signals, _src=src, _level=level):
            return signal_scalar(signals, _src) - _level

        specs.append(EventSpec(block=ctx.b_name, label=label, func=_g))
    return specs
