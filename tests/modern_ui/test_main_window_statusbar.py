"""
Characterization tests for ModernDiaBloSWindow's status-bar cluster.

main_window.py historically had zero coverage. These tests build a REAL
``ModernDiaBloSWindow`` under offscreen Qt and pin down the observable behavior
of the status-bar cluster before it is extracted into a dedicated manager:

  * ``_setup_statusbar``         (builds the pills/labels, installs the
                                  status_message.setText -> pill propagation shim)
  * ``_refresh_status_counts``   (blocks / wires / scopes pill)
  * ``_refresh_file_status``     (filename + unsaved indicator)

The status bar exposes its segments as attributes on the window
(``status_message``, ``status_pill``, ``file_status``, ``file_unsaved_status``,
``counts_status``, ``cursor_status``, ``zoom_status``, ``theme_status``); those
names are referenced widely across the codebase, so the tests assert on them
directly to lock the public surface.

An autouse fixture snapshots/restores the dsim collections the refresh methods
read, so tests can't leak state into each other.

Run with:
    QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python -m pytest \
        tests/modern_ui/test_main_window_statusbar.py -p no:cacheprovider \
        -o addopts="" --timeout=60 --timeout-method=signal
"""

import types

import pytest


@pytest.fixture(scope="module")
def window(qapp):
    from modern_ui.main_window import ModernDiaBloSWindow
    w = ModernDiaBloSWindow()
    yield w
    w.close()


@pytest.fixture(autouse=True)
def _restore_dsim(window):
    """Snapshot/restore the dsim state the status bar reads from."""
    dsim = window.dsim
    saved_blocks = list(getattr(dsim, 'blocks_list', []) or [])
    saved_lines = list(getattr(dsim, 'line_list', []) or [])
    saved_dirty = getattr(dsim, 'dirty', False)
    saved_fp = getattr(dsim, 'current_filepath', None)
    yield
    dsim.blocks_list[:] = saved_blocks
    dsim.line_list[:] = saved_lines
    dsim.dirty = saved_dirty
    dsim.current_filepath = saved_fp
    # Return the pill to a clean idle state for the next test.
    window.status_message.setText("")


def _fake_block(block_fn):
    return types.SimpleNamespace(block_fn=block_fn)


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------

class TestStatusbarConstruction:
    def test_segments_exist(self, window):
        for attr in ("status_pill", "status_message", "file_status",
                     "file_unsaved_status", "counts_status", "cursor_status",
                     "zoom_status", "theme_status"):
            assert hasattr(window, attr), f"missing status segment: {attr}"

    def test_status_message_is_hidden_shim(self, window):
        # The status_message label is a hidden compatibility shim.
        assert window.status_message.isHidden()

    def test_counts_pill_initial_text(self, window):
        window._refresh_status_counts()
        assert window.counts_status.text() == "blocks 0 · wires 0 · scopes 0"


# ---------------------------------------------------------------------------
# status_message.setText -> status_pill state propagation
# ---------------------------------------------------------------------------

class TestStatusMessagePropagation:
    def test_setText_updates_underlying_label(self, window):
        window.status_message.setText("hello world")
        assert window.status_message.text() == "hello world"

    def test_running_sets_running_state(self, window):
        window.status_message.setText("Running simulation...")
        assert window.status_pill.property("state") == "running"

    def test_paused_sets_paused_state(self, window):
        window.status_message.setText("Simulation paused")
        assert window.status_pill.property("state") == "paused"

    def test_error_sets_error_state(self, window):
        window.status_message.setText("Error: something failed")
        assert window.status_pill.property("state") == "error"

    def test_plain_text_sets_idle_state(self, window):
        window.status_message.setText("Ready")
        assert window.status_pill.property("state") == "idle"

    def test_running_takes_priority_only_without_paused(self, window):
        # 'run' present but 'paus' also present -> paused branch wins.
        window.status_message.setText("run paused")
        assert window.status_pill.property("state") == "paused"


# ---------------------------------------------------------------------------
# _refresh_status_counts
# ---------------------------------------------------------------------------

class TestRefreshCounts:
    def test_counts_blocks_wires_scopes(self, window):
        dsim = window.dsim
        dsim.blocks_list[:] = [
            _fake_block("Gain"),
            _fake_block("Scope"),
            _fake_block("FieldScope"),
            _fake_block("Sum"),
        ]
        dsim.line_list[:] = [object(), object()]
        window._refresh_status_counts()
        assert window.counts_status.text() == "blocks 4 · wires 2 · scopes 2"

    def test_counts_empty(self, window):
        window.dsim.blocks_list[:] = []
        window.dsim.line_list[:] = []
        window._refresh_status_counts()
        assert window.counts_status.text() == "blocks 0 · wires 0 · scopes 0"


# ---------------------------------------------------------------------------
# _refresh_file_status
# ---------------------------------------------------------------------------

class TestRefreshFileStatus:
    def test_untitled_when_no_file(self, window):
        window.dsim.current_filepath = None
        window.dsim.dirty = False
        window._refresh_file_status()
        assert window.file_status.text() == "untitled"
        assert window.file_unsaved_status.text() == ""

    def test_filename_basename_and_unsaved(self, window):
        window.dsim.current_filepath = "/some/dir/my_diagram.diablos"
        window.dsim.dirty = True
        window._refresh_file_status()
        assert window.file_status.text() == "my_diagram.diablos"
        assert window.file_unsaved_status.text() == "unsaved"

    def test_saved_clears_unsaved_indicator(self, window):
        window.dsim.current_filepath = "/x/saved.diablos"
        window.dsim.dirty = False
        window._refresh_file_status()
        assert window.file_status.text() == "saved.diablos"
        assert window.file_unsaved_status.text() == ""
