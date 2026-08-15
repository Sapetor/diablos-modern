"""The seam between a block's configured ``params`` and its runtime ``exec_params``.

Every block carries two parameter dicts:

``params``
    What the user configured, plus structural stamps written by the UI, the
    file loader and the subsystem manager.  Workspace variables are *not*
    resolved here -- a gain may still be the string ``"Kp"``.

``exec_params``
    What :meth:`SimulationEngine.execute_block` actually hands to
    ``execute()``.  Built by ``_resolve_block_params`` as a resolved *copy* of
    ``params``, and then owned by the block: all runtime state ('mem', '_int',
    '_prev', ...) accumulates here.

The copy is one-directional and lossy, which is the hazard this module exists
to contain:

* ``params`` -> ``exec_params`` happens only inside ``_resolve_block_params``,
  which early-returns when the cached dict is still valid for this step.  The
  cache fingerprint (``compile_cache.compile_param_items``) deliberately
  *excludes* '_'-prefixed keys, so a '_' key written to ``params`` mid-run is
  invisible twice over: the copy does not run, and the write cannot force it
  to.
* ``exec_params`` -> ``params`` never happens at all.

Two shipped bugs had exactly this shape, one per direction: the RK45 ``_skip_``
flag (written to ``params``, read from ``exec_params``, so sinks recorded every
sub-step) and the optimizer's ``_accumulated_cost_`` readback (written to
``exec_params``, read from ``params``, so the objective was a constant 0.0).

The two rules below close one direction each.  Read with
:func:`runtime_params`; push down with :data:`PUSH_DOWN_KEYS`.
"""

from __future__ import annotations

from typing import Any, Dict

# Keys the engine or UI legitimately writes to `params` *during* a run and
# that the block must observe on its next execution.
#
# Keep this set tiny and justified.  Anything here is copied over the block's
# own value on every step, so a key the *block* maintains must never appear --
# '_init_start_' is the cautionary example: reset_memblocks sets it True in
# both dicts, the block clears its own copy to False once initialised, and
# pushing the stale True down again would re-initialise the block on every
# single step.
#
# '_inputs_'/'_outputs_' already round-trip by accident: they are the two
# entries in compile_cache._COMPILE_CACHE_ALLOWED_INTERNAL_KEYS, so changing
# them alters the fingerprint and forces a full re-resolve.  They are listed
# here so the behaviour is stated rather than inherited from a cache decision.
PUSH_DOWN_KEYS = frozenset(
    {
        "_skip_",  # RK45 sub-step gate; set per sub-step by the interpreter
        "_name_",  # block rename must reach the block's error messages
        "_inputs_",
        "_outputs_",
    }
)


def runtime_params(block: Any) -> Dict[str, Any]:
    """Return the dict the block actually reads and writes at runtime.

    Prefer ``exec_params``: that is what ``execute()`` receives, so it holds
    every value the run produced.  Fall back to ``params`` for callers that
    run before (or without) a simulation -- canvas validation, and result
    readbacks on a diagram that was never initialised.

    Always read block state through this helper rather than touching either
    dict directly, so a reader cannot pick the one the writer did not use.
    """
    exec_params = getattr(block, "exec_params", None)
    if exec_params:
        return exec_params
    return getattr(block, "params", None) or {}


def push_down_internal_params(block: Any) -> None:
    """Refresh :data:`PUSH_DOWN_KEYS` in ``exec_params`` from ``params``.

    Called on the ``_resolve_block_params`` cache-hit path, where the full
    '_'-prefixed copy is skipped.  Only the narrow set above is refreshed --
    a blanket copy would clobber the runtime state the block owns.
    """
    exec_params = getattr(block, "exec_params", None)
    if not exec_params:
        return
    params = getattr(block, "params", None) or {}
    for key in PUSH_DOWN_KEYS:
        if key in params:
            exec_params[key] = params[key]
