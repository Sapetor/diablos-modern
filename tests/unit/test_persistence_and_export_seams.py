"""Four audit follow-ups on the persistence / export / plotting seams.

* ``DSim.save`` took no ``filepath``, so the ``_auto_save`` fallback branch in
  ``ModernDiaBloSWindow`` (``dsim.save(autosave=True, filepath=...)``) was dead
  code that would have raised ``TypeError``. ``FileService.save`` already
  accepted one; ``DSim.save`` now forwards it.
* ``lib/services/file_service.py`` imported ``QFileDialog`` at module level,
  putting the widget toolkit in the persistence layer. The "ask the user for a
  path" half moved to ``lib/services/file_dialogs.py`` (lazy Qt import).
* ``lib/plotting/field_scope_mixin.py`` imported ``modern_ui`` (an inverted
  dependency). The GUI now injects the dialog factory; the lazy import is only
  the fallback.
* ``lib/export/python_codegen.py`` silently kept unresolved workspace variables
  in the exported script. It now logs and collects a warning naming the block.
"""

import json
import types

import numpy as np
import pytest


# --------------------------------------------------------------------------
# DSim.save(filepath=...)
# --------------------------------------------------------------------------


@pytest.mark.unit
class TestDsimSaveAcceptsFilepath:
    def test_save_writes_to_the_given_path_without_a_dialog(self, qapp, tmp_path, monkeypatch):
        from lib.lib import DSim
        from PyQt5.QtCore import QPoint

        dsim = DSim()
        menu = {b.fn_name: b for b in dsim.menu_blocks}
        dsim.add_block(menu["step"], QPoint(100, 100))

        # A dialog here would mean the filepath was ignored.
        import lib.services.file_dialogs as file_dialogs

        def fail(*args, **kwargs):
            raise AssertionError("save(filepath=...) must not open a file dialog")

        monkeypatch.setattr(file_dialogs, "prompt_save_path", fail)

        target = tmp_path / "explicit.diablos"
        assert dsim.save(filepath=str(target)) == 0
        assert target.exists()
        assert "blocks_data" in json.loads(target.read_text(encoding="utf-8"))

    def test_autosave_with_filepath_keeps_the_dirty_flag(self, qapp, tmp_path):
        """The _auto_save fallback branch must behave like the primary one."""
        from lib.lib import DSim
        from PyQt5.QtCore import QPoint

        dsim = DSim()
        menu = {b.fn_name: b for b in dsim.menu_blocks}
        dsim.add_block(menu["step"], QPoint(100, 100))
        assert dsim.dirty is True

        target = tmp_path / "auto.diablos"
        assert dsim.save(autosave=True, filepath=str(target)) == 0
        assert target.exists()
        assert dsim.dirty is True, "an autosave is a snapshot, not a save"


# --------------------------------------------------------------------------
# FileService / file_dialogs split
# --------------------------------------------------------------------------


@pytest.mark.unit
class TestFileServiceHasNoWidgetImport:
    def test_module_source_does_not_import_qtwidgets(self):
        import pathlib

        import lib.services.file_service as fs

        source = pathlib.Path(fs.__file__).read_text(encoding="utf-8")
        # QtCore.QRect stays (a block's coords *is* a QRect); QtWidgets must not
        # be imported at module scope.
        assert "from PyQt5.QtWidgets import" not in source.split("def __getattr__")[0]

    def test_qfiledialog_is_still_reachable_for_back_compat(self):
        """Existing tests monkeypatch lib.services.file_service.QFileDialog."""
        from PyQt5.QtWidgets import QFileDialog

        import lib.services.file_service as fs

        assert fs.QFileDialog is QFileDialog

    def test_unknown_attribute_still_raises(self):
        import lib.services.file_service as fs

        with pytest.raises(AttributeError):
            fs.NoSuchThing

    def test_save_with_filepath_never_touches_the_dialog_module(
        self, file_service, tmp_path, monkeypatch
    ):
        import lib.services.file_service as fs

        monkeypatch.setattr(
            fs, "prompt_save_path", lambda *a, **k: pytest.fail("dialog was opened")
        )
        target = tmp_path / "direct.diablos"
        assert file_service.save(filepath=str(target)) == 0
        assert target.exists()

    def test_load_prompts_only_without_a_filepath(self, file_service, tmp_path, monkeypatch):
        import lib.services.file_service as fs

        target = tmp_path / "roundtrip.diablos"
        file_service.save(filepath=str(target))

        monkeypatch.setattr(
            fs, "prompt_open_path", lambda *a, **k: pytest.fail("dialog was opened")
        )
        assert file_service.load(str(target)) is not None

        # With no path it does prompt; a cancel comes back as "".
        monkeypatch.setattr(fs, "prompt_open_path", lambda *a, **k: "")
        assert file_service.load() is None


# --------------------------------------------------------------------------
# field_scope_mixin dialog injection
# --------------------------------------------------------------------------


@pytest.mark.unit
class TestFieldScopeDialogInjection:
    def test_injected_factory_is_used_instead_of_importing_modern_ui(self, qapp, monkeypatch):
        from lib.plotting import field_scope_mixin as fsm

        calls = []

        def fake_default(exporter, block_name):
            raise AssertionError("the lazy modern_ui import must not run")

        monkeypatch.setattr(fsm, "_default_animation_dialog", fake_default)

        mixin = fsm._FieldScopeRenderMixin()
        mixin.animation_dialog_factory = lambda exporter, name: calls.append(name)

        block = types.SimpleNamespace(name="fieldscope1")
        mixin._show_export_dialog(
            block,
            np.zeros((3, 4)),
            np.linspace(0.0, 1.0, 3),
            params={"L": 1.0},
            dimension="1d",
        )
        assert calls == ["fieldscope1"]

    def test_module_level_registration_also_works(self, qapp, monkeypatch):
        from lib.plotting import field_scope_mixin as fsm

        calls = []
        monkeypatch.setattr(
            fsm, "_default_animation_dialog", lambda *a: pytest.fail("fallback used")
        )
        fsm.set_animation_dialog_factory(lambda exporter, name: calls.append(name))
        try:
            mixin = fsm._FieldScopeRenderMixin()
            block = types.SimpleNamespace(name="fieldscope2")
            mixin._show_export_dialog(
                block, np.zeros((3, 4)), np.linspace(0.0, 1.0, 3), {"L": 1.0}, "1d"
            )
        finally:
            fsm.set_animation_dialog_factory(None)
        assert calls == ["fieldscope2"]

    def test_falls_back_to_the_lazy_import_when_nothing_is_injected(self, qapp, monkeypatch):
        from lib.plotting import field_scope_mixin as fsm

        calls = []
        monkeypatch.setattr(fsm, "_default_animation_dialog", lambda e, n: calls.append(n))

        mixin = fsm._FieldScopeRenderMixin()
        block = types.SimpleNamespace(name="fieldscope3")
        mixin._show_export_dialog(
            block, np.zeros((3, 4)), np.linspace(0.0, 1.0, 3), {"L": 1.0}, "1d"
        )
        assert calls == ["fieldscope3"]


# --------------------------------------------------------------------------
# python_codegen warnings channel
# --------------------------------------------------------------------------


@pytest.mark.unit
class TestCodegenWarnsOnUnresolvedParams:
    def test_failed_resolution_names_the_block_and_params(self, monkeypatch, caplog):
        from lib.export import python_codegen

        class Boom:
            def resolve_params(self, params):
                raise RuntimeError("workspace unavailable")

        monkeypatch.setattr("lib.workspace.WorkspaceManager", lambda: Boom())

        block = types.SimpleNamespace(name="gain2", params={"gain": "Kp"}, exec_params=None)
        collected = []
        with caplog.at_level("WARNING"):
            resolved = python_codegen._resolved_params(block, collected)

        # The unresolved value is still returned (the export is best-effort)...
        assert resolved == {"gain": "Kp"}
        # ...but it is no longer silent.
        assert len(collected) == 1
        assert "gain2" in collected[0] and "gain" in collected[0]
        assert "gain2" in caplog.text

    def test_generator_exposes_a_warnings_channel(self):
        from lib.export.python_codegen import PythonCodeGenerator

        generator = PythonCodeGenerator([], [])
        assert generator.warnings == []
