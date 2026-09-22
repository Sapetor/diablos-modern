"""
A masked subsystem shows its mask name unless the user has typed their own label.

``apply_mask_appearance`` decides whether ``username`` is still app-generated,
and used to accept only ``username == name``. The ways a Subsystem is created
each leave a different default, and none matched:

    create from selection  name="Subsystem3"  username="Subsystem"
    paste                  name="subsystem3"  username="Subsystem3"

so masking a freshly created subsystem, or pasting a masked one, kept the
generic label. Renaming the mask also left the old mask name behind, because
nothing counted a previous mask's name as generated.
"""

from types import SimpleNamespace

import pytest
from PyQt6.QtCore import QRect

from blocks.subsystem import Subsystem
from lib.masks import _has_default_username, apply_mask_appearance, set_mask
from modern_ui.managers.clipboard_manager import ClipboardManager

MOTOR = {"name": "Motor", "parameters": []}
PUMP = {"name": "Pump", "parameters": []}


@pytest.fixture(autouse=True)
def _qt(qapp):
    """DBlock builds a QPixmap, which needs a live QApplication."""
    return qapp


def _created_from_selection(sid=3):
    """The naming SubsystemManager._new_subsystem_block leaves behind."""
    subsys = Subsystem()
    subsys.sid = sid
    subsys.name = f"Subsystem{sid}"
    return subsys


def _paste_copy_of(subsys):
    """Copy ``subsys`` and paste it through the real ClipboardManager."""
    subsys.selected = True
    subsys.ports = {"in": [], "out": []}
    dsim = SimpleNamespace(
        blocks_list=[subsys], line_list=[], menu_blocks=[], colors=None, dirty=False
    )
    canvas = SimpleNamespace(
        dsim=dsim,
        update=lambda: None,
        block_selected=SimpleNamespace(emit=lambda *_: None),
        simulation_status_changed=SimpleNamespace(emit=lambda *_: None),
        history_manager=SimpleNamespace(
            capture_snapshot=lambda: {"pre": True}, push_snapshot=lambda *_: None
        ),
    )
    manager = ClipboardManager(canvas)
    manager.copy_selected_blocks()
    manager.paste_blocks(None)
    (pasted,) = dsim.blocks_list[1:]
    return pasted


@pytest.mark.unit
class TestHasDefaultUsername:
    @pytest.mark.parametrize(
        "block_fn, name, username",
        [
            ("Gain", "gain3", "gain3"),  # an ordinary block: username defaults to name
            ("Gain", "gain3", ""),
            ("Subsystem", "Subsystem3", "Subsystem"),  # created from a selection
            ("Subsystem", "subsystem3", "Subsystem3"),  # pasted
        ],
    )
    def test_generated_labels_count_as_default(self, block_fn, name, username):
        block = SimpleNamespace(name=name, username=username, block_fn=block_fn, sid=3)
        assert _has_default_username(block)

    @pytest.mark.parametrize("username", ["MyPlant", "Motor", "Subsystem4"])
    def test_a_typed_label_is_kept(self, username):
        block = SimpleNamespace(name="subsystem3", username=username, block_fn="Subsystem", sid=3)
        assert not _has_default_username(block)


@pytest.mark.unit
class TestMaskNameIsShown:
    def test_masking_a_subsystem_created_from_a_selection(self):
        subsys = _created_from_selection()
        set_mask(subsys, MOTOR)
        assert subsys.username == "Motor"

    def test_pasting_a_masked_subsystem(self):
        source = Subsystem(block_name="Subsystem2", sid=2, coords=QRect(100, 100, 100, 80))
        set_mask(source, MOTOR)
        pasted = _paste_copy_of(source)
        assert pasted.name != source.name
        assert pasted.username == "Motor"

    def test_renaming_the_mask_carries_the_label_along(self):
        subsys = _created_from_selection()
        set_mask(subsys, MOTOR)
        set_mask(subsys, PUMP)
        assert subsys.username == "Pump"


@pytest.mark.unit
class TestUserLabelIsRespected:
    def test_masking_keeps_a_typed_label(self):
        subsys = Subsystem(block_name="MyPlant", sid=6, coords=QRect(0, 0, 100, 80))
        set_mask(subsys, MOTOR)
        assert subsys.username == "MyPlant"

    def test_renaming_the_mask_keeps_a_typed_label(self):
        subsys = Subsystem(block_name="MyPlant", sid=6, coords=QRect(0, 0, 100, 80))
        set_mask(subsys, MOTOR)
        set_mask(subsys, PUMP)
        assert subsys.username == "MyPlant"

    def test_force_still_overrides(self):
        subsys = Subsystem(block_name="MyPlant", sid=6, coords=QRect(0, 0, 100, 80))
        set_mask(subsys, MOTOR)
        apply_mask_appearance(subsys, force=True)
        assert subsys.username == "Motor"
