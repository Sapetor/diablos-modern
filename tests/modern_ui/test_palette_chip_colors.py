"""
Palette chips take the same category colour family as the block on the canvas.

``_category_chip_colors`` used to fall back to ``text_secondary`` for every
category it did not name, so Analysis, Logic, Optimization, Optimization
Primitives and PDE all drew as identical grey chips, and Routing was mapped to
``block_other_accent`` although the canvas paints Routing blocks with
``block_routing``. The chip now mirrors ``SimulationModel._get_category_color``:
category ``X`` on the canvas uses ``block_X``, its chip uses ``block_X_accent``.
"""

import pytest

from lib.theming.theme_manager import ThemeType, theme_manager
from modern_ui.widgets.modern_palette import _category_chip_colors

# Every category the shipped blocks declare, with the canvas colour key that
# SimulationModel._get_category_color resolves it to.
CANVAS_KEY = {
    "Sources": "block_source",
    "Math": "block_process",
    "Control": "block_control",
    "Routing": "block_routing",
    "Sinks": "block_sink",
    "Analysis": "block_analysis",
    "PDE": "block_pde",
    "Optimization": "block_optimization",
    "Optimization Primitives": "block_optimization",
    "Logic": "block_other",
    "Other": "block_other",
}


@pytest.fixture(params=[ThemeType.LIGHT, ThemeType.DARK], ids=["light", "dark"])
def theme(request, qapp):
    previous = theme_manager.current_theme
    theme_manager.set_theme(request.param)
    yield request.param
    theme_manager.set_theme(previous)


@pytest.mark.parametrize("category", sorted(CANVAS_KEY))
def test_chip_accent_matches_the_canvas_colour_family(theme, category):
    _bg, accent, _fg = _category_chip_colors(category)
    expected = theme_manager.get_color(CANVAS_KEY[category] + "_accent")
    assert accent.name() == expected.name(), "%s chip uses %s, canvas family %s_accent is %s" % (
        category,
        accent.name(),
        CANVAS_KEY[category],
        expected.name(),
    )


def test_categories_with_their_own_canvas_colour_get_distinct_chips(theme):
    """The regression: five categories used to collapse onto one grey."""
    own_colour = [c for c, key in CANVAS_KEY.items() if key != "block_other"]
    accents = {c: _category_chip_colors(c)[1].name() for c in own_colour}
    grey = theme_manager.get_color("block_other_accent").name()
    assert grey not in accents.values(), "a coloured category fell back to grey: %r" % accents
    # The two Optimization families share one canvas colour; everything else differs.
    assert len(set(accents.values())) == len(set(CANVAS_KEY[c] for c in own_colour))


def test_unknown_category_falls_back_like_the_canvas(theme):
    _bg, accent, _fg = _category_chip_colors("Something New")
    assert accent.name() == theme_manager.get_color("block_other_accent").name()
