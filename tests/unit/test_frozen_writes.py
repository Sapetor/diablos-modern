"""Regression tests: no runtime write may land on a read-only filesystem.

Packaged (PyInstaller) builds crashed with ``[Errno 30] Read-only file system``
because several writers used a *relative* path:

- ``FileService.save(autosave=True)`` built ``"saves/<name>_AUTOSAVE.diablos"``
  (reported as ``Error saving file saves/data_AUTOSAVE.dat``),
- ``DSim.export_data`` built ``os.path.join("saves", basename)``,
- ``RunHistoryService`` defaulted to ``"saves/run_history.json"``,
- ``file_dialogs.default_directory()`` pointed inside the read-only bundle,
- ``DiagramService.__init__`` did ``os.makedirs(os.getcwd() + "/saves")`` --
  a ``.app`` launched from Finder has ``cwd == "/"``, hence ``'/saves'``.

Two layers of protection:

``TestFrozenModeWrites`` fakes a real frozen launch -- ``sys.frozen`` set,
``sys._MEIPASS`` pointing at a read-only "bundle", the process CWD chdir'ed into
a read-only directory, and the platform data root redirected into ``tmp_path``
-- then runs the actual writers and asserts they neither raise nor write
anywhere but the user data directory.

``TestNoRelativeWrites`` is the static backstop: it walks the AST of every
shipped module and fails on ``open()`` / ``os.makedirs()`` / ``np.save*()`` with
a relative string literal, so the pattern cannot come back in a code path no
test happens to execute.
"""

import ast
import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: Packages whose modules run inside the packaged app and must never write to a
#: relative path. Scripts under ``scripts/``, ``tools/`` and ``tests/`` are
#: developer tooling run from a checkout, so they are out of scope.
SHIPPED_ROOTS = ("lib", "modern_ui", "blocks", "config")
SHIPPED_FILES = ("diablos_modern.py",)

#: lib/app_paths.py is the one place allowed to name the ``saves`` folder.
PATH_AUTHORITY = Path("lib") / "app_paths.py"


# ---------------------------------------------------------------------------
# Frozen-mode behaviour
# ---------------------------------------------------------------------------


@pytest.fixture
def frozen_app(tmp_path, monkeypatch):
    """Simulate a packaged launch with a read-only bundle and a read-only CWD.

    Yields the writable per-user data directory the app is expected to fall
    back to (``<home>/Library/Application Support/DiaBloS`` on macOS,
    ``%APPDATA%/DiaBloS`` on Windows, ``$XDG_DATA_HOME/DiaBloS`` elsewhere).
    """
    import lib.app_paths as app_paths

    bundle = tmp_path / "bundle"
    (bundle / "config").mkdir(parents=True)
    (bundle / "config" / "default_config.json").write_text("{}", encoding="utf-8")

    read_only_cwd = tmp_path / "readonly_cwd"
    read_only_cwd.mkdir()

    home = tmp_path / "home"
    home.mkdir()

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle), raising=False)
    for var in ("HOME", "USERPROFILE", "APPDATA", "XDG_DATA_HOME"):
        monkeypatch.setenv(var, str(home))

    original_cwd = os.getcwd()
    os.chdir(read_only_cwd)
    # 0o555: enterable and listable, but nothing may be created inside it --
    # exactly what a macOS app bundle's CWD looks like.
    bundle.chmod(0o555)
    read_only_cwd.chmod(0o555)
    try:
        data_dir = Path(app_paths.get_user_data_dir())
        assert home in data_dir.parents, (
            f"the user data dir {data_dir} escaped the redirected home {home}"
        )
        yield data_dir
    finally:
        os.chdir(original_cwd)
        read_only_cwd.chmod(0o755)
        bundle.chmod(0o755)


def _assert_written_under(path, data_dir):
    path = Path(path)
    assert path.is_absolute(), f"{path} must be absolute, not relative to the CWD"
    assert path.exists(), f"{path} was never written"
    assert data_dir in path.parents, f"{path} is outside the writable data dir {data_dir}"


@pytest.mark.unit
class TestFrozenModeWrites:
    """Every writer lands in the per-user data dir, and none of them raise."""

    def test_the_cwd_really_is_read_only(self, frozen_app):
        """Guard the guard: if the CWD were writable these tests prove nothing."""
        with pytest.raises(OSError):
            with open(os.path.join(os.getcwd(), "canary.txt"), "w") as fh:
                fh.write("x")

    def test_user_saves_path_is_absolute_and_writable(self, frozen_app):
        from lib.app_paths import user_saves_path

        target = user_saves_path("probe.json")
        with open(target, "w", encoding="utf-8") as fh:
            fh.write("{}")
        _assert_written_under(target, frozen_app)

    def test_autosave_writes_under_the_user_data_dir(self, frozen_app, qapp):
        """The ``saves/data_AUTOSAVE.dat`` crash: FileService's autosave branch."""
        from lib.services.file_service import FileService

        class _Model:
            blocks_list = []
            line_list = []
            dirty = True

        service = FileService(_Model())
        service.filename = "data.dat"
        assert service.save(autosave=True) == 0
        assert service.last_write_error is None

        _assert_written_under(frozen_app / "saves" / "data_AUTOSAVE.dat", frozen_app)

    def test_export_data_writes_under_the_user_data_dir(self, frozen_app, qapp):
        """``DSim.export_data`` used os.path.join("saves", ...)."""
        import numpy as np

        from lib.lib import DSim

        class _ExportBlock:
            block_fn = "Export"
            params = {
                "format": "csv",
                "vec_labels": "sig",
                "vector": np.array([1.0, 2.0, 3.0]),
                "vec_dim": 1,
            }

        dsim = DSim()
        dsim.model.blocks_list.append(_ExportBlock())
        dsim.filename = "frozen_model.diablos"
        dsim.engine.timeline = np.array([0.0, 0.1, 0.2])
        dsim.export_data()

        _assert_written_under(frozen_app / "saves" / "frozen_model.csv", frozen_app)

    def test_run_history_persists_under_the_user_data_dir(self, frozen_app):
        from lib.services.run_history_service import RunHistoryService

        service = RunHistoryService()
        service.set_persist(True)  # writes immediately

        _assert_written_under(frozen_app / "saves" / "run_history.json", frozen_app)

    def test_recent_files_persist_under_the_user_data_dir(self, frozen_app):
        from modern_ui.managers.recent_files_manager import RecentFilesManager

        manager = RecentFilesManager.__new__(RecentFilesManager)
        manager.save(["/somewhere/model.diablos"])

        _assert_written_under(frozen_app / "config" / "recent_files.json", frozen_app)
        assert manager.load() == ["/somewhere/model.diablos"]

    def test_user_config_write_does_not_touch_the_bundle(self, frozen_app):
        """Modified settings go to the user copy; the bundled one stays read-only."""
        import json

        from lib.app_paths import resource_path, user_data_path

        bundled = Path(resource_path("config/default_config.json"))
        user_copy = Path(user_data_path("config/default_config.json"))
        assert user_copy != bundled

        with open(user_copy, "w", encoding="utf-8") as fh:
            json.dump({"display": {"scaling_factor": 1.5}}, fh)

        _assert_written_under(user_copy, frozen_app)
        assert bundled.read_text(encoding="utf-8") == "{}", "the bundle must be untouched"

    def test_dialog_default_directory_is_writable(self, frozen_app):
        """The Save dialog must not open inside the read-only bundle."""
        from lib.services.file_dialogs import default_directory

        directory = Path(default_directory())
        assert directory.is_dir()
        assert os.access(directory, os.W_OK)
        assert frozen_app in directory.parents or directory == frozen_app / "saves"

    def test_log_file_path_is_writable(self, frozen_app):
        from lib.logging_config import _get_log_file_path

        resolved = Path(_get_log_file_path("diablos_modern.log"))
        assert resolved.is_absolute()
        assert os.access(resolved.parent, os.W_OK)

    def test_diagram_service_construction_does_not_mkdir_in_the_cwd(self, frozen_app, qapp):
        """It used to makedirs(os.getcwd() + "/saves") -> '/saves' from Finder."""
        from lib.lib import DSim
        from lib.services.diagram_service import DiagramService

        class _Window:
            def __init__(self):
                self.dsim = DSim()

        service = DiagramService(_Window())
        assert os.access(service.save_directory(), os.W_OK)
        assert not (Path(os.getcwd()) / "saves").exists()


# ---------------------------------------------------------------------------
# Static backstop
# ---------------------------------------------------------------------------


def _shipped_modules():
    seen = []
    for root in SHIPPED_ROOTS:
        seen.extend(sorted((PROJECT_ROOT / root).rglob("*.py")))
    seen.extend(PROJECT_ROOT / name for name in SHIPPED_FILES)
    return [p for p in seen if p.relative_to(PROJECT_ROOT) != PATH_AUTHORITY]


def _literal(node):
    """Return the leading string constant of ``node``, or None.

    Handles both ``"saves/x"`` and ``f"saves/{name}"`` (whose first f-string
    part is a plain constant).
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr) and node.values:
        first = node.values[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            return first.value
    return None


def _called_name(func):
    """Dotted-ish name of a call target: ``open``, ``makedirs``, ``savez``…"""
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


#: Calls whose first positional argument is a filesystem destination.
PATH_FIRST_ARG = {
    "open",
    "makedirs",
    "mkdir",
    "save",
    "savez",
    "savez_compressed",
    "savetxt",
    "savemat",
    "savefig",
    "write_text",
    "write_bytes",
}


@pytest.mark.unit
class TestNoRelativeWrites:
    """No shipped module may name a write destination with a relative literal."""

    def test_no_relative_literal_paths_in_write_calls(self):
        offenders = []
        for module in _shipped_modules():
            tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not node.args:
                    continue
                if _called_name(node.func) not in PATH_FIRST_ARG:
                    continue
                text = _literal(node.args[0])
                if text is None or not text or os.path.isabs(text):
                    continue
                # A bare filename with no separator is almost always a suffix or
                # a mode string ("w"), not a path; only directory-ish literals
                # ("saves/…", "config/…") are the bug this guards against.
                if "/" not in text and os.sep not in text:
                    continue
                offenders.append(
                    "{}:{} -> {!r}".format(module.relative_to(PROJECT_ROOT), node.lineno, text)
                )

        assert not offenders, (
            "Relative path literal passed to a filesystem call. In a packaged "
            "build the CWD is read-only (a macOS .app starts at '/'), so this "
            "fails with [Errno 30]. Route it through lib.app_paths "
            "(user_data_path / user_saves_path / user_logs_path):\n  " + "\n  ".join(offenders)
        )

    def test_only_app_paths_names_the_saves_folder(self):
        """``saves/`` is spelled once, in lib/app_paths.py."""
        offenders = []
        for module in _shipped_modules():
            tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                for arg in list(node.args) + [kw.value for kw in node.keywords]:
                    text = _literal(arg)
                    if text and text.startswith(("saves/", "saves" + os.sep)):
                        offenders.append(
                            "{}:{} -> {!r}".format(
                                module.relative_to(PROJECT_ROOT), node.lineno, text
                            )
                        )

        assert not offenders, (
            "The writable saves/ folder is resolved by lib.app_paths."
            "user_saves_path(); no other module should build the path itself:\n  "
            + "\n  ".join(offenders)
        )
