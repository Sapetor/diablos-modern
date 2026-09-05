"""Regression test for File > New.

``ProjectManager.new_diagram`` -> ``DiagramService.new_diagram`` ->
``DSim.new_diagram()``, a method ``lib/lib.py`` never defined -- so File > New
raised ``AttributeError`` and did nothing.
"""

import pytest
from PyQt5.QtCore import QPoint


@pytest.mark.unit
class TestNewDiagram:
    def test_resets_the_diagram(self, qapp):
        from lib.lib import DSim

        dsim = DSim()
        menu = {b.fn_name: b for b in dsim.menu_blocks}
        step = dsim.add_block(menu["step"], QPoint(100, 100))
        scope = dsim.add_block(menu["scope"], QPoint(300, 100))
        dsim.add_line((step.name, 0, step.out_coords[0]), (scope.name, 0, scope.in_coords[0]))
        assert dsim.blocks_list and dsim.line_list

        dsim.new_diagram()

        assert dsim.blocks_list == []
        assert dsim.line_list == []
        # An empty diagram has nothing unsaved...
        assert dsim.dirty is False
        # ...and is untitled, so the next save cannot overwrite the old file.
        assert dsim.filename == "data.diablos"
        assert dsim.file_service.filename == "data.diablos"
