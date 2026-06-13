"""
Regression test for the file-dialog extension mismatch.

Bug: FileService.save/load showed file dialogs filtered to '*.dat' and
defaulted new saves to 'data.dat', while every shipped example and the menu
use the '.diablos' extension. As a result users could not see their own
.diablos files in the Open dialog, and new saves got the wrong extension.

Fix: the open/save dialog filters now include '*.diablos' as the
primary/default option (while still accepting '*.dat' for backward
compatibility), and new saves default to '.diablos'.
"""

import pytest


@pytest.mark.regression
class TestFileDialogFilters:
    """Open/save dialogs must surface .diablos files and default to .diablos."""

    def test_default_filename_uses_diablos(self, file_service):
        """A fresh FileService must default new saves to a .diablos name."""
        assert file_service.filename.endswith('.diablos')

    def test_open_dialog_filter_includes_diablos(self, file_service, monkeypatch):
        """The Open dialog filter string must include 'diablos' (primary)."""
        captured = {}

        def fake_open(parent, caption, directory, filter_str, options=None):
            captured['filter'] = filter_str
            # Simulate the user cancelling so load() returns early.
            return ("", "")

        monkeypatch.setattr(
            'lib.services.file_service.QFileDialog.getOpenFileName', fake_open
        )

        result = file_service.load()  # no filepath -> opens dialog
        assert result is None  # cancelled
        assert 'diablos' in captured['filter']
        # Backward compatibility: legacy .dat files must still be selectable.
        assert 'dat' in captured['filter']

    def test_save_dialog_filter_includes_diablos(self, file_service, monkeypatch):
        """The Save dialog filter string must include 'diablos' (primary)."""
        captured = {}

        def fake_save(parent, caption, directory, filter_str, options=None):
            captured['filter'] = filter_str
            # Simulate the user cancelling so save() returns early (code 1).
            return ("", "")

        monkeypatch.setattr(
            'lib.services.file_service.QFileDialog.getSaveFileName', fake_save
        )

        rc = file_service.save()  # not autosave, no filepath -> opens dialog
        assert rc == 1  # cancelled
        # .diablos must be the primary/default filter group.
        assert captured['filter'].lower().startswith(('diablos', 'data files'))
        assert 'diablos' in captured['filter'].lower()

    def test_save_defaults_extension_to_diablos(self, file_service, monkeypatch, tmp_path):
        """A chosen name with no recognized extension gains '.diablos'."""
        chosen = str(tmp_path / "my_diagram")  # no extension
        saved_paths = []

        def fake_save(parent, caption, directory, filter_str, options=None):
            return (chosen, "DiaBloS Files (*.diablos)")

        def fake_save_to_file(data, filename):
            saved_paths.append(filename)
            return True

        monkeypatch.setattr(
            'lib.services.file_service.QFileDialog.getSaveFileName', fake_save
        )
        monkeypatch.setattr(file_service, 'save_to_file', fake_save_to_file)

        rc = file_service.save()
        assert rc == 0
        assert saved_paths == [chosen + '.diablos']

    def test_save_preserves_explicit_dat_extension(self, file_service, monkeypatch, tmp_path):
        """An explicit legacy .dat name is kept (no double extension)."""
        chosen = str(tmp_path / "legacy.dat")
        saved_paths = []

        def fake_save(parent, caption, directory, filter_str, options=None):
            return (chosen, "Data Files (*.dat)")

        def fake_save_to_file(data, filename):
            saved_paths.append(filename)
            return True

        monkeypatch.setattr(
            'lib.services.file_service.QFileDialog.getSaveFileName', fake_save
        )
        monkeypatch.setattr(file_service, 'save_to_file', fake_save_to_file)

        rc = file_service.save()
        assert rc == 0
        assert saved_paths == [chosen]
