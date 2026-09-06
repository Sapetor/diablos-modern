"""User block modules: discovery, isolation, registration and end-to-end use.

``lib/user_blocks.py`` lets a third party add a block by dropping a ``.py``
file into a folder -- no repository edit, no rebuild.  The properties that make
that safe are all tested here:

* **precedence** -- ``DIABLOS_BLOCKS_PATH`` beats a diagram-local ``blocks/``
  folder beats the per-user folder, and the repository's own ``blocks/``
  package is never scanned as a user folder (in a dev checkout it *is* the
  per-user path);
* **isolation** -- a module that raises on import, a class that breaks the
  contract, or a name that collides with a built-in is skipped with a message;
  everything else still loads;
* **frozen builds** -- where ``blocks/`` inside the bundle is not scanned, user
  blocks must still load (they are imported by path);
* **registration** -- a loaded class reaches the palette like a built-in, with
  a marker identifying it;
* **round trip** -- a diagram saved with a user block loads and runs.
"""

import json
import os
import sys
import types

import pytest

from lib import user_blocks as ub

# --------------------------------------------------------------------------- #
# Module sources written into tmp_path
# --------------------------------------------------------------------------- #

GOOD_BLOCK = """
import numpy as np
from blocks.base_block import BaseBlock

BLOCK_API_VERSION = 1


class {cls}(BaseBlock):
    @property
    def block_name(self):
        return "{name}"

    @property
    def category(self):
        return "{category}"

    @property
    def params(self):
        return {{"gain": {{"type": "float", "default": {gain}, "doc": "Scale factor"}}}}

    @property
    def inputs(self):
        return [{{"name": "in", "type": "any"}}]

    @property
    def outputs(self):
        return [{{"name": "out", "type": "any"}}]

    def execute(self, time, inputs, params, **kwargs):
        return {{0: np.atleast_1d(inputs.get(0, 0.0)) * params["gain"]}}
"""

BROKEN_IMPORT = """
raise RuntimeError("this module is broken on purpose")
"""

INVALID_BLOCK = """
from blocks.base_block import BaseBlock


class InvalidBlock(BaseBlock):
    @property
    def block_name(self):
        return "InvalidUserBlock"

    @property
    def params(self):
        return {"gain": {"type": "float"}}          # no 'default'

    @property
    def inputs(self):
        return [{"type": "any"}]                    # no 'name'

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    def execute(self, time, inputs, params):        # no **kwargs
        return {0: 0.0}
"""

FUTURE_API = """
from blocks.base_block import BaseBlock

BLOCK_API_VERSION = 9999


class FutureBlock(BaseBlock):
    @property
    def block_name(self):
        return "FutureUserBlock"

    @property
    def params(self):
        return {}

    @property
    def inputs(self):
        return []

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    def execute(self, time, inputs, params, **kwargs):
        return {0: 0.0}
"""

COLLIDING_BLOCK = GOOD_BLOCK.format(cls="ImpostorGainBlock", name="Gain", category="Math", gain=1.0)


def _write(folder, file_name, source):
    os.makedirs(str(folder), exist_ok=True)
    path = os.path.join(str(folder), file_name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(source)
    return path


def _good(cls="AcmeScaleBlock", name="AcmeScale", category="Math", gain=2.0):
    return GOOD_BLOCK.format(cls=cls, name=name, category=category, gain=gain)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _isolated_user_blocks(tmp_path, monkeypatch):
    """No env var, and a private (empty) per-user folder, unless a test says so."""
    monkeypatch.delenv(ub.BLOCKS_ENV_VAR, raising=False)
    home = tmp_path / "userdata"
    home.mkdir()
    monkeypatch.setattr(ub, "get_user_data_dir", lambda: str(home))
    ub.purge_user_modules()
    yield home
    ub.purge_user_modules()


# --------------------------------------------------------------------------- #
# Discovery and precedence
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestSearchPaths:
    def test_default_order_is_env_then_project_then_user(
        self, tmp_path, monkeypatch, _isolated_user_blocks
    ):
        env_a = tmp_path / "env_a"
        env_b = tmp_path / "env_b"
        monkeypatch.setenv(ub.BLOCKS_ENV_VAR, os.pathsep.join([str(env_a), str(env_b)]))
        diagram = tmp_path / "project" / "model.diablos"

        paths = ub.user_block_search_paths(str(diagram))

        assert paths == [
            str(env_a),
            str(env_b),
            str(tmp_path / "project" / "blocks"),
            str(_isolated_user_blocks / "blocks"),
        ]

    def test_no_diagram_means_no_project_folder(self, _isolated_user_blocks):
        assert ub.user_block_search_paths() == [str(_isolated_user_blocks / "blocks")]

    def test_duplicates_are_collapsed_keeping_priority(self, tmp_path, monkeypatch):
        folder = tmp_path / "shared"
        monkeypatch.setenv(ub.BLOCKS_ENV_VAR, os.pathsep.join([str(folder), str(folder)]))
        assert ub.user_block_search_paths().count(str(folder)) == 1

    def test_the_builtin_blocks_package_is_never_a_user_folder(self, monkeypatch):
        """In a dev checkout ``<user data>/blocks`` IS the built-in package."""
        import blocks.base_block

        builtin = os.path.dirname(os.path.abspath(blocks.base_block.__file__))
        monkeypatch.setenv(ub.BLOCKS_ENV_VAR, builtin)
        monkeypatch.setattr(ub, "get_user_data_dir", lambda: os.path.dirname(builtin))

        assert ub.user_block_search_paths() == []

    def test_project_folder_helper(self):
        assert ub.project_blocks_dir(None) is None
        got = ub.project_blocks_dir(os.path.join("some", "where", "m.diablos"))
        assert got.endswith(os.path.join("where", "blocks"))


@pytest.mark.unit
class TestFileDiscovery:
    def test_only_public_top_level_modules_are_picked_up(self, tmp_path, monkeypatch):
        folder = tmp_path / "blocks"
        _write(folder, "acme.py", _good())
        _write(folder, "_helper.py", "VALUE = 1\n")
        _write(folder, "__init__.py", "")
        _write(folder, "notes.txt", "not python")
        os.makedirs(str(folder / "subpackage"))
        _write(folder / "subpackage", "deep.py", _good(cls="DeepBlock", name="Deep"))
        monkeypatch.setenv(ub.BLOCKS_ENV_VAR, str(folder))

        found = ub.discover_user_block_files()

        assert [os.path.basename(p) for p in found] == ["acme.py"]

    def test_higher_priority_folder_shadows_the_same_file_name(self, tmp_path, monkeypatch):
        high = tmp_path / "high"
        low = tmp_path / "low"
        _write(high, "acme.py", _good(name="HighPriority"))
        _write(low, "acme.py", _good(name="LowPriority"))
        monkeypatch.setenv(ub.BLOCKS_ENV_VAR, os.pathsep.join([str(high), str(low)]))

        report = ub.load_user_blocks()

        assert [c().block_name for c in report.classes] == ["HighPriority"]

    def test_missing_folders_are_skipped_silently(self, tmp_path, monkeypatch):
        monkeypatch.setenv(ub.BLOCKS_ENV_VAR, str(tmp_path / "does-not-exist"))
        assert ub.discover_user_block_files() == []


# --------------------------------------------------------------------------- #
# Loading and isolation
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestLoading:
    def test_a_valid_module_loads_and_is_marked(self, tmp_path, monkeypatch):
        folder = tmp_path / "blocks"
        path = _write(folder, "acme.py", _good())
        monkeypatch.setenv(ub.BLOCKS_ENV_VAR, str(folder))

        report = ub.load_user_blocks()

        assert len(report.classes) == 1
        cls = report.classes[0]
        assert cls().block_name == "AcmeScale"
        assert ub.is_user_block(cls) is True
        assert ub.user_block_source(cls) == path
        assert report.problems == []

    def test_a_broken_module_is_skipped_and_the_others_still_load(self, tmp_path, monkeypatch):
        folder = tmp_path / "blocks"
        _write(folder, "aa_broken.py", BROKEN_IMPORT)
        _write(folder, "bb_good.py", _good())
        monkeypatch.setenv(ub.BLOCKS_ENV_VAR, str(folder))

        report = ub.load_user_blocks()

        assert [c().block_name for c in report.classes] == ["AcmeScale"]
        assert len(report.problems) == 1
        assert "broken on purpose" in report.problems[0].message
        assert "aa_broken.py" in report.problems[0].describe()

    def test_a_broken_module_leaves_nothing_in_sys_modules(self, tmp_path, monkeypatch):
        folder = tmp_path / "blocks"
        _write(folder, "wreck.py", BROKEN_IMPORT)
        monkeypatch.setenv(ub.BLOCKS_ENV_VAR, str(folder))

        ub.load_user_blocks()

        assert ub.USER_MODULE_PREFIX + ".wreck" not in sys.modules

    def test_an_invalid_block_is_skipped_with_an_actionable_message(self, tmp_path, monkeypatch):
        folder = tmp_path / "blocks"
        _write(folder, "invalid.py", INVALID_BLOCK)
        monkeypatch.setenv(ub.BLOCKS_ENV_VAR, str(folder))

        report = ub.load_user_blocks()

        assert report.classes == []
        assert len(report.problems) == 1
        message = report.problems[0].message
        assert "default" in message and "**kwargs" in message

    def test_a_name_that_collides_with_a_builtin_is_skipped(self, tmp_path, monkeypatch):
        folder = tmp_path / "blocks"
        _write(folder, "impostor.py", COLLIDING_BLOCK)
        monkeypatch.setenv(ub.BLOCKS_ENV_VAR, str(folder))

        report = ub.load_user_blocks(builtin_names=["Gain", "Sum"])

        assert report.classes == []
        assert "already taken" in report.problems[0].message

    def test_a_newer_api_version_is_refused(self, tmp_path, monkeypatch):
        folder = tmp_path / "blocks"
        _write(folder, "future.py", FUTURE_API)
        monkeypatch.setenv(ub.BLOCKS_ENV_VAR, str(folder))

        report = ub.load_user_blocks()

        assert report.classes == []
        assert "BLOCK_API_VERSION" in report.problems[0].message

    def test_imported_builtin_classes_are_not_re_registered(self, tmp_path, monkeypatch):
        folder = tmp_path / "blocks"
        _write(folder, "reexport.py", "from blocks.gain import GainBlock  # noqa: F401\n")
        monkeypatch.setenv(ub.BLOCKS_ENV_VAR, str(folder))

        report = ub.load_user_blocks()

        assert report.classes == []
        assert report.problems == []

    def test_reload_picks_up_an_edited_file(self, tmp_path, monkeypatch):
        folder = tmp_path / "blocks"
        _write(folder, "acme.py", _good(gain=2.0))
        monkeypatch.setenv(ub.BLOCKS_ENV_VAR, str(folder))

        first = ub.load_user_blocks().classes[0]
        assert first().params["gain"]["default"] == 2.0

        _write(folder, "acme.py", _good(gain=5.0))
        again = ub.load_user_blocks(reload=True).classes[0]

        assert again().params["gain"]["default"] == 5.0

    def test_load_never_raises_on_an_unreadable_folder(self, tmp_path, monkeypatch):
        # A file where a folder is expected: os.path.isdir() is False, skipped.
        bogus = _write(tmp_path, "not_a_folder", "")
        monkeypatch.setenv(ub.BLOCKS_ENV_VAR, bogus)
        assert ub.load_user_blocks().classes == []


# --------------------------------------------------------------------------- #
# Frozen (PyInstaller) mode
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestFrozenMode:
    def test_user_blocks_load_when_frozen(self, tmp_path, monkeypatch):
        """The bundled blocks/ package is not scanned, but user blocks are."""
        from lib.block_loader import load_blocks

        folder = tmp_path / "userdata" / "blocks"
        _write(folder, "acme.py", _good())

        monkeypatch.setattr(sys, "frozen", True, raising=False)
        monkeypatch.setattr(ub, "get_user_data_dir", lambda: str(tmp_path / "userdata"))

        names = [cls().block_name for cls in load_blocks(reload_user=True)]

        assert "AcmeScale" in names
        assert "Gain" in names, "the frozen static registry must still provide built-ins"

    def test_frozen_user_folder_follows_app_paths(self, monkeypatch):
        from lib import app_paths

        monkeypatch.setattr(sys, "frozen", True, raising=False)
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(os, "makedirs", lambda *a, **k: None)
        monkeypatch.setattr(ub, "get_user_data_dir", app_paths.get_user_data_dir)

        folder = ub.user_blocks_dir()

        assert folder.endswith(os.path.join("DiaBloS", "blocks"))
        assert "Application Support" in folder or ".local/share" in folder


# --------------------------------------------------------------------------- #
# Registration in the model / palette
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestBlockLoaderIntegration:
    def test_load_blocks_appends_user_classes_after_builtins(self, tmp_path, monkeypatch):
        from lib.block_loader import load_blocks, load_builtin_blocks

        folder = tmp_path / "blocks"
        _write(folder, "acme.py", _good())
        monkeypatch.setenv(ub.BLOCKS_ENV_VAR, str(folder))

        builtins_only = load_builtin_blocks()
        everything = load_blocks(reload_user=True)

        assert len(everything) == len(builtins_only) + 1
        assert everything[: len(builtins_only)] == builtins_only
        assert everything[-1]().block_name == "AcmeScale"

    def test_include_user_false_skips_third_party_code(self, tmp_path, monkeypatch):
        from lib.block_loader import load_blocks

        folder = tmp_path / "blocks"
        _write(folder, "acme.py", _good())
        monkeypatch.setenv(ub.BLOCKS_ENV_VAR, str(folder))

        names = [cls().block_name for cls in load_blocks(include_user=False)]

        assert "AcmeScale" not in names


@pytest.mark.qt
class TestPaletteRegistration:
    def test_a_user_block_reaches_the_palette_with_a_marker(self, qapp, tmp_path, monkeypatch):
        from lib.lib import DSim
        from modern_ui.widgets.modern_palette import (
            CompactBlockRow,
            ModernBlockPalette,
            is_user_menu_block,
        )

        folder = tmp_path / "blocks"
        _write(folder, "acme.py", _good(category="Control"))
        monkeypatch.setenv(ub.BLOCKS_ENV_VAR, str(folder))
        ub.purge_user_modules()

        dsim = DSim()
        entries = [mb for mb in dsim.menu_blocks if mb.block_fn == "AcmeScale"]
        assert len(entries) == 1, "the user block was not registered in menu_blocks"
        entry = entries[0]
        assert entry.category == "Control"
        assert is_user_menu_block(entry) is True

        palette = ModernBlockPalette(dsim)
        try:
            rows = {
                row.menu_block.block_fn: row
                for section in palette._sections
                for row in section.findChildren(CompactBlockRow)
            }
            assert "AcmeScale" in rows, "the user block is missing from the palette"
            assert rows["AcmeScale"].user_badge is not None
            assert rows["Gain"].user_badge is None
            assert "acme.py" in rows["AcmeScale"].toolTip()
        finally:
            palette.deleteLater()

    def test_reload_blocks_picks_up_a_new_file(self, qapp, tmp_path, monkeypatch):
        from lib.lib import DSim

        folder = tmp_path / "blocks"
        monkeypatch.setenv(ub.BLOCKS_ENV_VAR, str(folder))
        os.makedirs(str(folder))

        dsim = DSim()
        assert not any(mb.block_fn == "AcmeScale" for mb in dsim.menu_blocks)

        _write(folder, "acme.py", _good())
        count = dsim.model.reload_blocks()

        assert count == 1
        # DSim aliases the model's list, so an in-place rebuild must be visible
        # through the alias the palette reads.
        assert any(mb.block_fn == "AcmeScale" for mb in dsim.menu_blocks)


# --------------------------------------------------------------------------- #
# Diagrams that use a user block
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestMissingBlockReporting:
    def test_missing_names_walk_subsystems(self):
        data = {
            "blocks_data": [
                {"block_fn": "Gain"},
                {"block_fn": "AcmeScale"},
                {
                    "block_fn": "Subsystem",
                    "sub_blocks": [{"block_fn": "Deep"}, {"block_fn": "Inport"}],
                },
            ]
        }
        assert ub.missing_block_names(data, ["Gain"]) == ["AcmeScale", "Deep"]

    def test_intrinsic_types_are_never_missing(self):
        data = {"blocks_data": [{"block_fn": t} for t in ("Subsystem", "Inport", "Outport")]}
        assert ub.missing_block_names(data, []) == []

    def test_message_names_the_block_and_the_search_paths(self, tmp_path, monkeypatch):
        folder = tmp_path / "custom"
        monkeypatch.setenv(ub.BLOCKS_ENV_VAR, str(folder))

        message = ub.missing_blocks_message(["AcmeScale"], str(tmp_path / "m.diablos"))

        assert "AcmeScale" in message
        assert str(folder) in message
        assert ub.BLOCKS_ENV_VAR in message

    def test_garbage_input_is_tolerated(self):
        assert ub.missing_block_names(None, []) == []
        assert ub.missing_block_names({"blocks_data": "nope"}, []) == []


@pytest.mark.integration
class TestSavedDiagramRoundTrip:
    def _diagram(self, tmp_path):
        from lib.diagram_builder import DiagramBuilder

        builder = DiagramBuilder(sim_time=0.2, sim_dt=0.01)
        step = builder.add_block("Step", 50, 100, params={"value": 1.0, "delay": 0.0})
        scale = builder.add_block("AcmeScale", 200, 100, params={"gain": 2.0})
        scope = builder.add_block("Scope", 350, 100, params={"labels": "default"})
        builder.connect(step, 0, scale, 0)
        builder.connect(scale, 0, scope, 0)
        path = tmp_path / "with_user_block.diablos"
        builder.save(str(path))
        return path

    def _dsim(self):
        from lib.lib import DSim
        from lib.workspace import WorkspaceManager

        WorkspaceManager._instance = None
        return DSim()

    def test_diagram_loads_and_runs_when_the_block_is_available(self, qapp, tmp_path, monkeypatch):
        folder = tmp_path / "blocks"
        _write(folder, "acme.py", _good())
        monkeypatch.setenv(ub.BLOCKS_ENV_VAR, str(folder))
        ub.purge_user_modules()
        path = self._diagram(tmp_path)

        dsim = self._dsim()
        data = dsim.file_service.load(filepath=str(path))
        assert data is not None
        dsim.file_service.apply_loaded_data(data)

        assert [b.block_fn for b in dsim.blocks_list].count("AcmeScale") == 1
        ok, message = dsim.run_tuning_simulation(0.2, 0.01)
        assert ok, message

        from lib.analysis.resim import harvest_scope_signals

        harvested = harvest_scope_signals(dsim)
        trace = next(iter(harvested["signals"].values()))
        # Step (1.0) through a gain of 2.0.
        assert max(abs(v) for v in trace[-5:]) == pytest.approx(2.0, abs=1e-6)

    def test_diagram_reports_the_block_as_missing_without_it(self, qapp, tmp_path, monkeypatch):
        folder = tmp_path / "blocks"
        _write(folder, "acme.py", _good())
        monkeypatch.setenv(ub.BLOCKS_ENV_VAR, str(folder))
        ub.purge_user_modules()
        path = self._diagram(tmp_path)

        # Now the block is gone: a machine that does not have it.
        monkeypatch.delenv(ub.BLOCKS_ENV_VAR)
        ub.purge_user_modules()
        dsim = self._dsim()
        with open(str(path), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        dsim.file_service.apply_loaded_data(data)

        assert not any(b.block_fn == "AcmeScale" for b in dsim.blocks_list)
        known = [mb.block_fn for mb in dsim.menu_blocks]
        missing = ub.missing_block_names(data, known)
        assert missing == ["AcmeScale"]
        assert "AcmeScale" in ub.missing_blocks_message(missing, str(path))


@pytest.mark.qt
class TestMenuActions:
    """The Edit menu must offer the reload and open-folder actions."""

    def test_edit_menu_lists_the_user_block_actions(self, qapp):
        from PyQt5.QtWidgets import QMenuBar

        from modern_ui.builders.menu_builder import MenuBuilder

        calls = []
        window = types.SimpleNamespace(
            copy_diagram_image=lambda: None,
            reload_user_blocks=lambda: calls.append("reload"),
            open_user_blocks_folder=lambda: calls.append("open"),
        )
        builder = MenuBuilder(window)
        menubar = QMenuBar()
        try:
            builder._create_edit_menu(menubar)
            edit_menu = menubar.actions()[0].menu()
            texts = [a.text() for a in edit_menu.actions()]
            assert any("User Blocks" in t for t in texts)
            assert any("Folder" in t and "User Blocks" in t for t in texts)

            for action in edit_menu.actions():
                if action.text().startswith("Reload") and "User" in action.text():
                    action.trigger()
            assert calls == ["reload"]
        finally:
            menubar.deleteLater()

    def test_main_window_exposes_the_facades(self):
        from modern_ui.main_window import ModernDiaBloSWindow

        assert callable(getattr(ModernDiaBloSWindow, "reload_user_blocks", None))
        assert callable(getattr(ModernDiaBloSWindow, "open_user_blocks_folder", None))
        assert callable(getattr(ModernDiaBloSWindow, "_warn_about_missing_block_types", None))
