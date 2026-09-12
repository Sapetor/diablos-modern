"""Outline shape resolution for blocks, free of any GUI toolkit.

The canvas painter and the LaTeX exporters must agree on which outline a block
has: a Gain is a triangle on screen and an ``isosceles triangle`` in TikZ, a
Goto is a tag in both.  The rule used to live only in
``modern_ui.renderers.block_renderer``, which imports Qt, so ``lib/`` -- where
the exporters live -- could not reach it and quietly exported everything as a
rectangle.  The pure logic lives here and ``block_renderer.resolve_block_shape``
delegates, so the two cannot diverge.
"""

import logging

logger = logging.getLogger(__name__)

#: Outline shapes every drawing routine knows how to produce (see BaseBlock.shape).
BLOCK_SHAPES = ("rect", "triangle", "circle", "tag")

#: Shape used when a block carries no ``block_instance`` (legacy / stub blocks
#: in tests), keyed by ``block_fn``.  Everything else falls back to a rectangle.
SHAPE_FALLBACK_BY_FN = {
    "Gain": "triangle",
    "MatrixGain": "triangle",
    "Sum": "circle",
    "Product": "circle",
    "Goto": "tag",
    "From": "tag",
}

#: A circle only reads well with a handful of ports on its curved edge; past
#: this the block is drawn as a rounded rectangle instead.
CIRCLE_MAX_INPUTS = 3


def _block_mask(block):
    """Return the block's mask definition, or None (never raises into a paint)."""
    try:
        from lib.masks import get_mask

        return get_mask(block)
    except Exception:  # pragma: no cover - defensive: painting must not fail
        logger.debug("Mask lookup failed while resolving a block shape", exc_info=True)
        return None


def resolve_block_shape(block) -> str:
    """Return the outline shape token to draw for ``block``.

    Reads ``block.block_instance.shape`` (BaseBlock hook) and applies the
    port-count fallback for circles, so every drawing routine (body, shadow,
    hover, export) agrees on the same outline.
    """
    instance = getattr(block, "block_instance", None)
    shape = None
    if instance is not None:
        try:
            shape = getattr(instance, "shape", None)
        except Exception:  # a broken property must never break painting
            shape = None
    if not shape:
        shape = SHAPE_FALLBACK_BY_FN.get(getattr(block, "block_fn", ""), "rect")
    # A masked subsystem may pick one of the outlines for its block; it wins
    # over the class hook so shadow, body, hover and export all agree.
    mask = _block_mask(block)
    if mask is not None and mask.get("shape"):
        shape = mask.get("shape")
    if shape not in BLOCK_SHAPES:
        shape = "rect"
    if shape == "circle":
        if getattr(block, "in_ports", 0) > CIRCLE_MAX_INPUTS or getattr(block, "out_ports", 0) > 1:
            shape = "rect"
    return shape
