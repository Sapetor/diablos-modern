"""Qt file-choosing dialogs for the diagram persistence layer.

Split out of :mod:`lib.services.file_service` so the persistence layer itself
(serialize / deserialize / read / write) has no ``PyQt6.QtWidgets`` dependency:
asking the user *where* to put a file is a UI concern, reading and writing it is
not. ``FileService.save``/``load`` only reach in here when no explicit
``filepath`` was supplied, so every headless caller (CLI, tests, autosave,
scripts) stays clear of QtWidgets.

QtWidgets is imported lazily inside the functions so importing this module -- or
``lib.services`` as a whole -- never pulls in the widget toolkit.
"""

import logging
import os

from lib.i18n import tr

logger = logging.getLogger(__name__)


def save_filter() -> str:
    """Extensions accepted for a diagram, newest/canonical first (translated)."""
    return (
        tr("DiaBloS Files")
        + " (*.diablos);;"
        + tr("Data Files")
        + " (*.dat);;"
        + tr("All Files")
        + " (*)"
    )


def open_filter() -> str:
    """Extensions accepted when opening a diagram (translated)."""
    return tr("DiaBloS Files") + " (*.diablos *.dat *.json);;" + tr("All Files") + " (*)"


#: Untranslated forms, kept for callers/tests that compare against the English text.
SAVE_FILTER = "DiaBloS Files (*.diablos);;Data Files (*.dat);;All Files (*)"
OPEN_FILTER = "DiaBloS Files (*.diablos *.dat *.json);;All Files (*)"


def default_directory() -> str:
    """The ``saves/`` directory the dialogs open in by default.

    Resolved through :func:`lib.app_paths.user_saves_dir`, so it is the project
    root's ``saves/`` in dev and the per-user data directory when frozen. It
    used to be ``<this file>/../../saves``, which in a PyInstaller build points
    inside the read-only bundle (``sys._MEIPASS/saves``): the Save dialog then
    opened on a directory that did not exist and could not be written.
    """
    from lib.app_paths import user_saves_dir

    return user_saves_dir(create=True)


def prompt_save_path(suggested_name: str, directory: str = None) -> str:
    """Ask the user where to save a diagram. Returns "" when cancelled."""
    from PyQt6.QtWidgets import QFileDialog

    directory = default_directory() if directory is None else directory
    filepath, _ = QFileDialog.getSaveFileName(
        None,
        tr("Save File"),
        os.path.join(directory, suggested_name),
        save_filter(),
    )
    return filepath or ""


def prompt_open_path(directory: str = None) -> str:
    """Ask the user which diagram to open. Returns "" when cancelled."""
    from PyQt6.QtWidgets import QFileDialog

    directory = default_directory() if directory is None else directory
    filepath, _ = QFileDialog.getOpenFileName(
        None,
        tr("Open File"),
        directory,
        open_filter(),
    )
    return filepath or ""
