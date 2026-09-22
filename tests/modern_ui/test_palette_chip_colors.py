"""
Palette chips and category dots take the same colour family as the block on the canvas.

The palette used to carry its own category->colour chains. The chip fell back to
``text_secondary``, so Analysis, Logic, Optimization, Optimization Primitives and
PDE drew as identical grey chips, and Routing disagreed with the canvas; the dot
beside it had drifted further still. All three now resolve through
``lib.theming.categories.category_theme_key``: the canvas fills with the key, the
chip and dot use ``key + "_accent"``.
"""

import pytest

from lib.theming.categories import category_theme_key
from lib.theming.theme_manager import ThemeType, theme_manager
from modern_ui.widgets.modern_palette import _category_accent, _category_chip_colors

# Every category the shipped blocks declare, plus one they don't.
EXPECTED_KEY = {
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
    "Something New": "block_other",
}


@pytest.mark.parametrize("category", sorted(EXPECTED_KEY))
def test_category_theme_key(category):
    assert category_theme_key(category) == EXPECTED_KEY[category]


@pytest.fixture(params=[ThemeType.LIGHT, ThemeType.DARK], ids=["light", "dark"])
def theme(request, qapp):
    previous = theme_manager.current_theme
    theme_manager.set_theme(request.param)
    yield request.param
    theme_manager.set_theme(previous)


@pytest.mark.parametrize("category", sorted(EXPECTED_KEY))
def test_chip_and_dot_use_the_category_accent(theme, category):
    expected = theme_manager.get_color(EXPECTED_KEY[category] + "_accent").name()
    assert _category_chip_colors(category)[1].name() == expected
    assert _category_accent(category).name() == expected
