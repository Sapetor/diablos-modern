"""Block category -> theme colour key, shared by the canvas and the palette."""

# First matching fragment wins, so order matters. A category matching none of
# them (Logic, Other, a user block's own category) uses block_other.
_CATEGORY_KEYS = (
    ("source", "block_source"),
    ("math", "block_process"),
    ("control", "block_control"),
    ("sink", "block_sink"),
    ("routing", "block_routing"),
    ("analysis", "block_analysis"),
    ("pde", "block_pde"),
    ("optim", "block_optimization"),
)


def category_theme_key(category) -> str:
    """Theme key for a block category's fill, e.g. ``"Routing"`` -> ``"block_routing"``.

    The matching accent (palette chips, category dots) is the same key plus
    ``"_accent"``.
    """
    c = str(category or "").lower()
    for fragment, key in _CATEGORY_KEYS:
        if fragment in c:
            return key
    return "block_other"
