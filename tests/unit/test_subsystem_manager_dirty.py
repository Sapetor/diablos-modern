"""Navigating into a subsystem must not discard the unsaved-changes flag.

BUG
---
``SubsystemManager.enter_subsystem`` reset ``self.dsim.dirty = False`` together
with the view state (``ss_count``). That was harmless while ``DSim.dirty`` was a
private copy nobody read, but ``DSim.dirty`` is now a property over
``model.dirty`` -- the one flag the status bar and the close prompt read. So
editing a diagram and then double-clicking into a subsystem silently marked the
edits as saved, and the window closed without warning.

FIX
---
``enter_subsystem`` no longer touches ``dirty``: drilling into a subsystem is a
view change, not a save.
"""

import pytest
from PyQt5.QtCore import QPoint, QRect


@pytest.fixture
def dsim(qapp):
    from lib.lib import DSim

    return DSim()


def _subsystem_with_contents(dsim):
    """A Subsystem block registered in the model, holding one child block."""
    from blocks.subsystem import Subsystem

    sub = Subsystem(block_name="Sub1", sid=1, coords=QRect(0, 0, 100, 80))
    sub.sub_blocks = []
    sub.sub_lines = []
    dsim.model.blocks_list.append(sub)
    dsim.blocks_list = dsim.model.blocks_list
    return sub


@pytest.mark.unit
class TestEnterSubsystemKeepsDirty:
    def test_edit_then_enter_subsystem_stays_dirty(self, dsim):
        """edit -> enter subsystem -> the diagram is still unsaved."""
        sub = _subsystem_with_contents(dsim)

        # An edit the user has not saved.
        menu = {b.fn_name: b for b in dsim.menu_blocks}
        dsim.add_block(menu["step"], QPoint(100, 100))
        assert dsim.dirty is True

        dsim.subsystem_manager.enter_subsystem(sub)

        assert dsim.dirty is True, "entering a subsystem must not clear the unsaved flag"
        assert dsim.model.dirty is True

    def test_enter_subsystem_does_not_invent_a_dirty_flag(self, dsim):
        """A clean diagram stays clean -- the fix must not flip the flag on."""
        sub = _subsystem_with_contents(dsim)
        dsim.dirty = False

        dsim.subsystem_manager.enter_subsystem(sub)

        assert dsim.dirty is False

    def test_navigation_still_switches_scope(self, dsim):
        """The behaviour the reset was bundled with is unchanged."""
        sub = _subsystem_with_contents(dsim)
        dsim.subsystem_manager.enter_subsystem(sub)

        assert dsim.subsystem_manager.current_subsystem == sub.name
        assert dsim.blocks_list is sub.sub_blocks
        assert dsim.line_list is sub.sub_lines
        assert dsim.ss_count == 0
