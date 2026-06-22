"""Canonical block-function name normalization.

The compiled solver and the post-solve replay loop both need to map a block's
raw ``block_fn`` (as stored in the diagram, e.g. ``"HeatEquation1D"`` or
``"TranFn"``) to the single canonical spelling used by their dispatch ladders
(e.g. ``"Heatequation1D"``, ``"TransferFcn"``).

Historically this normalization was hand-copied in four places
(``system_compiler._create_block_executor``, ``system_compiler.compile_system``
state allocation, and two sites in ``simulation_engine.run_compiled_simulation``)
and the copies drifted. ``canonical_fn`` is the single source of truth so the
compiler and the replay loop can never disagree on a block's identity.
"""


def canonical_fn(block_fn):
    """Return the canonical dispatch name for a raw ``block_fn``.

    The base transform is ``str.title()`` (which the dispatch ladders assume);
    the explicit overrides below repair the handful of names ``title()`` maps to
    the wrong CamelCase form.

    Note: the ``...1d -> ...1D`` PDE overrides are belt-and-suspenders. ``title()``
    already uppercases the letter following a digit (``"heatequation1d".title()``
    is ``"Heatequation1D"``), so they are no-ops for every real ``block_fn``; they
    are kept to document the canonical PDE names and guard against future callers.

    Args:
        block_fn: The block's raw ``block_fn`` string (may be ``None``/empty).

    Returns:
        The canonical name, or ``""`` for a falsy input.
    """
    if not block_fn:
        return ''

    fn = block_fn.title()

    if fn == 'Statespace':
        fn = 'StateSpace'
    if fn in ('Transferfcn', 'Tranfn'):
        fn = 'TransferFcn'
    if block_fn == 'PID':  # 'PID'.title() == 'Pid'; keep the upstream spelling
        fn = 'PID'
    if fn == 'Ratelimiter':
        fn = 'RateLimiter'

    # PDE 1D blocks (no-ops in practice; see module/function docstring).
    if fn == 'Heatequation1d':
        fn = 'Heatequation1D'
    if fn == 'Waveequation1d':
        fn = 'Waveequation1D'
    if fn == 'Advectionequation1d':
        fn = 'Advectionequation1D'
    if fn == 'Diffusionreaction1d':
        fn = 'Diffusionreaction1D'

    return fn
