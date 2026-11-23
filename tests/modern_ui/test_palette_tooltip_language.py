"""Palette tooltips show the block description in the UI language.

The tooltip used to insert the block's ``doc`` text verbatim, so with the
UI in Spanish every palette tooltip still read in English, although the
catalog has every description (the Properties panel already translated it).
"""

import pytest

from diablos.core import i18n
from diablos.gui.widgets import modern_palette

pytestmark = pytest.mark.qt


@pytest.fixture
def spanish():
    previous = i18n.current_language()
    i18n.set_language("es")
    yield
    i18n.set_language(previous)


def test_block_description_in_the_tooltip_is_translated(qapp, spanish):
    from diablos.core.lib import DSim

    step = next(b for b in DSim().menu_blocks if b.fn_name == "step")
    row = modern_palette.CompactBlockRow(step, "Sources", None)
    doc = str(step.block_class().doc)
    tip = row.toolTip()
    assert i18n.tr(doc) != doc, "the catalog should have this description"
    assert i18n.tr(doc).splitlines()[0] in tip
    assert doc.splitlines()[0] not in tip
