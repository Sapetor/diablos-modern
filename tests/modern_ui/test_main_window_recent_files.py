"""
Characterization tests for ModernDiaBloSWindow's recent-files cluster and the
pure ``_convert_param_value`` helper.

main_window.py has historically had ZERO test coverage. These tests build a
REAL ``ModernDiaBloSWindow`` under offscreen Qt and pin down the observable
behavior of:

  * ``_convert_param_value``        (pure type-coercion helper)
  * ``_load_recent_files`` /
    ``_save_recent_files`` /
    ``_add_recent_file`` /
    ``_update_recent_files_menu`` /
    ``_open_recent_file`` /
    ``_clear_recent_files``         (the recent-files cluster)

They lock in current behavior so the planned extraction of a
``RecentFilesManager`` can be proven behavior-preserving.

The recent-files methods persist to ``user_data_path('config/recent_files.json')``
which resolves through ``lib.app_paths.get_user_data_dir()``. Every test
redirects that to a fresh ``tmp_path`` so the user's real recent-files config is
never touched and each test is fully isolated.

Run with:
    QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python -m pytest \
        tests/modern_ui/test_main_window_recent_files.py -p no:cacheprovider -o addopts="-q"
"""

import json
import os

import pytest

import lib.app_paths


@pytest.fixture(scope="module")
def window(qapp):
    """Build one real ModernDiaBloSWindow for the module (construction is heavy)."""
    from modern_ui.main_window import ModernDiaBloSWindow
    w = ModernDiaBloSWindow()
    yield w
    w.close()


@pytest.fixture
def cfg_dir(tmp_path, monkeypatch):
    """Redirect the user-data dir to a fresh tmp dir for this test only.

    The recent-files methods call ``user_data_path('config/recent_files.json')``
    which resolves via ``get_user_data_dir()`` at call time, so patching here
    fully isolates each test from the real user config.
    """
    monkeypatch.setattr(lib.app_paths, "get_user_data_dir", lambda: str(tmp_path))
    return tmp_path


def _recent_json(cfg_dir):
    return os.path.join(str(cfg_dir), "config", "recent_files.json")


# ---------------------------------------------------------------------------
# _convert_param_value  (pure helper)
# ---------------------------------------------------------------------------

class TestConvertParamValue:
    def test_int(self, window):
        assert window._convert_param_value("5", int) == 5

    def test_float(self, window):
        assert window._convert_param_value("3.14", float) == 3.14

    def test_bool_true_string(self, window):
        assert window._convert_param_value("True", bool) is True
        assert window._convert_param_value("true", bool) is True

    def test_bool_false_string(self, window):
        assert window._convert_param_value("False", bool) is False
        assert window._convert_param_value("anything", bool) is False

    def test_list(self, window):
        assert window._convert_param_value("[1, 2, 3]", list) == [1, 2, 3]

    def test_list_non_list_literal_falls_back_to_string(self, window):
        # ast.literal_eval('5') -> 5 (int), not a list -> TypeError -> str fallback
        assert window._convert_param_value("5", list) == "5"

    def test_str_passthrough(self, window):
        assert window._convert_param_value("hello", str) == "hello"

    def test_unconvertible_expression_kept_as_string(self, window):
        # Expressions / variable names are preserved verbatim for later
        # resolution by the WorkspaceManager.
        assert window._convert_param_value("2*K", float) == "2*K"
        assert window._convert_param_value("[K, K]", list) == "[K, K]"

    def test_bad_int_falls_back_to_string(self, window):
        assert window._convert_param_value("not_a_number", int) == "not_a_number"


# ---------------------------------------------------------------------------
# recent-files cluster
# ---------------------------------------------------------------------------

class TestRecentFiles:
    def test_load_missing_returns_empty(self, window, cfg_dir):
        assert window._load_recent_files() == []

    def test_save_then_load_roundtrip(self, window, cfg_dir):
        window._save_recent_files(["/a/one.diablos", "/b/two.diablos"])
        assert window._load_recent_files() == ["/a/one.diablos", "/b/two.diablos"]
        # File written in the expected location/shape.
        with open(_recent_json(cfg_dir)) as f:
            data = json.load(f)
        assert data == {"recent_files": ["/a/one.diablos", "/b/two.diablos"]}

    def test_add_inserts_at_front(self, window, cfg_dir):
        window._add_recent_file("/x/first.diablos")
        window._add_recent_file("/x/second.diablos")
        assert window._load_recent_files() == [
            "/x/second.diablos",
            "/x/first.diablos",
        ]

    def test_add_empty_path_is_noop(self, window, cfg_dir):
        window._add_recent_file("")
        assert window._load_recent_files() == []

    def test_add_existing_moves_to_front_without_duplicate(self, window, cfg_dir):
        for p in ["/a.diablos", "/b.diablos", "/c.diablos"]:
            window._add_recent_file(p)
        # Re-add /a -> should move to front, no duplicate.
        window._add_recent_file("/a.diablos")
        assert window._load_recent_files() == [
            "/a.diablos",
            "/c.diablos",
            "/b.diablos",
        ]

    def test_capped_at_ten(self, window, cfg_dir):
        for i in range(15):
            window._add_recent_file(f"/file_{i}.diablos")
        recent = window._load_recent_files()
        assert len(recent) == 10
        # Most-recent first; oldest five dropped.
        assert recent[0] == "/file_14.diablos"
        assert recent[-1] == "/file_5.diablos"

    def test_clear_empties_list(self, window, cfg_dir):
        window._add_recent_file("/a.diablos")
        window._clear_recent_files()
        assert window._load_recent_files() == []

    def test_update_menu_empty_shows_disabled_placeholder(self, window, cfg_dir):
        window._update_recent_files_menu()
        actions = window.recent_files_menu.actions()
        assert len(actions) == 1
        assert actions[0].text() == "No recent files"
        assert actions[0].isEnabled() is False

    def test_update_menu_lists_basenames_plus_clear(self, window, cfg_dir):
        window._save_recent_files(["/some/dir/alpha.diablos", "/other/beta.diablos"])
        window._update_recent_files_menu()
        actions = window.recent_files_menu.actions()
        texts = [a.text() for a in actions]
        # filenames (basenames), a separator, then "Clear Recent Files"
        assert "alpha.diablos" in texts
        assert "beta.diablos" in texts
        assert "Clear Recent Files" in texts
        # Full path preserved in action data for the file entries.
        file_actions = [a for a in actions if a.data()]
        assert "/some/dir/alpha.diablos" in [a.data() for a in file_actions]

    def test_open_missing_file_removes_it_from_recent(self, window, cfg_dir):
        missing = os.path.join(str(cfg_dir), "gone.diablos")
        window._save_recent_files([missing, "/keep.diablos"])
        # File does not exist -> warning path -> removed from recent list.
        window._open_recent_file(missing)
        assert window._load_recent_files() == ["/keep.diablos"]
