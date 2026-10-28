"""A project-local ``blocks/`` folder runs no code until the user trusts it.

Opening a diagram reloads user blocks, and a ``blocks/`` folder next to it
used to be imported unconditionally -- every module executed before any block
class was even looked at, so a downloaded project could run code just by being
opened. The folder now joins the search path only once it is recorded as
trusted; the GUI asks first and defaults to not loading.
"""

import json
import os

import pytest

import lib.user_blocks as ub

PAYLOAD = """\
import pathlib
pathlib.Path(__file__).with_name("ran.txt").write_text("executed")
"""


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A downloaded project: diagram + blocks/ with a side-effecting module."""
    monkeypatch.delenv(ub.BLOCKS_ENV_VAR, raising=False)
    home = tmp_path / "userdata"
    home.mkdir()
    monkeypatch.setattr(ub, "get_user_data_dir", lambda: str(home))
    root = tmp_path / "downloaded"
    blocks = root / "blocks"
    blocks.mkdir(parents=True)
    (blocks / "payload.py").write_text(PAYLOAD, encoding="utf-8")
    diagram = root / "model.diablos"
    diagram.write_text("{}", encoding="utf-8")
    ub.purge_user_modules()
    yield {"diagram": str(diagram), "blocks": str(blocks), "home": home}
    ub.purge_user_modules()


@pytest.mark.unit
class TestUntrustedProjectFolder:
    def test_is_not_searched(self, project):
        assert project["blocks"] not in ub.user_block_search_paths(project["diagram"])

    def test_loading_runs_none_of_its_code(self, project):
        ub.load_user_blocks(project["diagram"], reload=True)
        assert not os.path.exists(os.path.join(project["blocks"], "ran.txt"))

    def test_held_back_modules_are_reported(self, project):
        held = ub.untrusted_project_block_files(project["diagram"])
        assert [os.path.basename(f) for f in held] == ["payload.py"]

    def test_a_corrupt_trust_store_trusts_nothing(self, project):
        store = project["home"] / ub.TRUST_STORE
        store.parent.mkdir(parents=True, exist_ok=True)
        store.write_text("{not json", encoding="utf-8")
        assert not ub.is_project_blocks_trusted(project["blocks"])


@pytest.mark.unit
class TestTrustedProjectFolder:
    def test_trusting_loads_it_and_persists(self, project):
        ub.trust_project_blocks(project["blocks"])
        assert ub.untrusted_project_block_files(project["diagram"]) == []
        ub.load_user_blocks(project["diagram"], reload=True)
        assert os.path.exists(os.path.join(project["blocks"], "ran.txt"))
        stored = json.loads((project["home"] / ub.TRUST_STORE).read_text(encoding="utf-8"))
        assert stored == [os.path.realpath(project["blocks"])]

    def test_trust_is_per_folder(self, project, tmp_path):
        ub.trust_project_blocks(project["blocks"])
        other = tmp_path / "elsewhere" / "blocks"
        other.mkdir(parents=True)
        (other / "x.py").write_text("", encoding="utf-8")
        diagram = str(tmp_path / "elsewhere" / "m.diablos")
        assert [os.path.basename(f) for f in ub.untrusted_project_block_files(diagram)] == ["x.py"]


@pytest.mark.unit
def test_the_gui_prompt_defaults_to_not_trusting(qapp, project):
    """With no button clicked (dialogs are stubbed in tests) nothing is trusted."""
    from PyQt6.QtWidgets import QWidget

    from modern_ui.main_window import ModernDiaBloSWindow

    host = QWidget()
    host._current_diagram_path = lambda: project["diagram"]
    assert ModernDiaBloSWindow._confirm_project_blocks_trust(host) is False
    assert not ub.is_project_blocks_trusted(project["blocks"])
