import os
import logging
import time
from PyQt6.QtWidgets import QMessageBox
from lib.app_paths import user_data_path
from lib.i18n import tr
from lib.services.diagram_service import DiagramService

logger = logging.getLogger(__name__)


class ProjectManager:
    """
    Manages project state, file operations, and autosave.
    """

    def __init__(self, main_window):
        self.window = main_window
        self.diagram_service = DiagramService(self.window)
        # Use the canonical autosave path the main window writes to
        # (user_data_path('config/.autosave.diablos')). Deriving a path from
        # main_window.__module__ resolved to <cwd>/autosave.json, which never
        # matched the real autosave file, so recovery/cleanup were no-ops.
        self.autosave_path = user_data_path("config/.autosave.diablos")

        # Initialize diagram service with window context if needed
        # (MainWindow code passed self to DiagramService, so we pass window)

    def new_diagram(self):
        """Create new diagram."""
        if self.diagram_service:
            self.diagram_service.new_diagram()
        elif hasattr(self.window.dsim, "clear_all"):
            self.window.dsim.clear_all()
        self.window.status_message.setText(tr("New diagram created"))
        # cleanup autosave on new? Maybe not necessary depending on logic

    def open_diagram(self, filename=None):
        """Open ``filename`` (asks for one when None); return the loaded data or None."""
        logger.info("open_diagram called")
        result = None
        if self.diagram_service:
            logger.info("Calling diagram_service.load_diagram()")
            result = self.diagram_service.load_diagram(filename)
            logger.info(f"load_diagram returned: {result is not None}")
            if result:
                self.window.status_message.setText(tr("Diagram opened"))
                # Single source of truth for the Recent Files menu is
                # RecentFilesManager (config/recent_files.json). Routing here
                # avoids a second, divergent QSettings-backed store writing the
                # same menu widget.
                if getattr(self.window, "recent_files_manager", None):
                    self.window.recent_files_manager.add(self.diagram_service.current_file)
        else:
            logger.warning("diagram_service is None!")
        return result

    def open_example(self, filename):
        """Open an example diagram; return the loaded data or None."""
        result = None
        if self.diagram_service:
            result = self.diagram_service.load_diagram(filename)
        self.window.status_message.setText(
            tr("Example {name} opened", name=os.path.basename(filename))
        )
        return result

    def save_diagram(self):
        """Save diagram."""
        if self.diagram_service:
            self.diagram_service.save_diagram()
            # Add to recent files if save successful
            # DiagramService usually returns success?
        self.window.status_message.setText(tr("Diagram saved"))

    # Recent files are owned by RecentFilesManager (config/recent_files.json),
    # which backs the Recent Files menu the user sees. ProjectManager used to
    # keep a second, divergent QSettings-backed store that cleared and
    # repopulated the same menu widget, producing nondeterministic contents and
    # purging entries on transient os.path.exists misses. That code was removed;
    # open_diagram now routes through window.recent_files_manager.add(...).

    # Autosave Logic
    def check_autosave_recovery(self):
        """Check for auto-save file on startup."""
        try:
            if os.path.exists(self.autosave_path):
                reply = QMessageBox.question(
                    self.window,
                    tr("Recover Auto-save?"),
                    tr(
                        "An auto-save file was found. Do you want to recover your previous session?"
                    ),
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes,
                )

                if reply == QMessageBox.StandardButton.Yes:
                    self.recover_autosave()
                else:
                    os.remove(self.autosave_path)
                    logger.info("Auto-save file deleted (user declined recovery)")
        except Exception as e:
            logger.error(f"Error checking auto-save recovery: {str(e)}")

    def recover_autosave(self):
        """Recover diagram from auto-save file.

        The auto-save is deleted only after it has actually loaded. A failed
        recovery moves it aside instead (see :meth:`_preserve_failed_autosave`):
        deleting it lost the user's only copy, and leaving it in place would let
        the next periodic auto-save overwrite it with the empty diagram.
        """
        try:
            loaded = (
                self.diagram_service.load_diagram(self.autosave_path)
                if self.diagram_service
                else None
            )
            if loaded is None:
                raise RuntimeError(tr("The auto-save file could not be loaded."))
        except Exception as e:
            logger.error(f"Error recovering from auto-save: {str(e)}")
            kept = self._preserve_failed_autosave()
            message = tr(
                "Could not recover auto-save file. Starting fresh.\n\nError: {error}", error=str(e)
            )
            if kept:
                message += "\n\n" + tr("The auto-save file was kept as:\n{path}", path=kept)
            QMessageBox.warning(self.window, tr("Recovery Failed"), message)
            return

        try:
            os.remove(self.autosave_path)
        except OSError:
            logger.debug("Could not remove the recovered auto-save file", exc_info=True)

        self.window.canvas.update()
        self.window.status_message.setText(tr("Diagram recovered from auto-save"))
        logger.info("Diagram successfully recovered from auto-save")

        QMessageBox.information(
            self.window,
            tr("Recovery Successful"),
            tr("Your diagram has been successfully recovered from the auto-save file."),
        )

    def _preserve_failed_autosave(self):
        """Rename an unrecoverable auto-save to a timestamped sibling.

        Returns the new path, or ``None`` if there was nothing to keep or the
        rename failed.
        """
        if not os.path.exists(self.autosave_path):
            return None
        stem, ext = os.path.splitext(self.autosave_path)
        kept = "{}.failed-{}{}".format(stem, time.strftime("%Y%m%d-%H%M%S"), ext)
        try:
            os.replace(self.autosave_path, kept)
        except OSError:
            logger.warning("Could not set the failed auto-save file aside", exc_info=True)
            return None
        logger.info("Kept the unrecoverable auto-save file as %s", kept)
        return kept

    def cleanup_autosave(self):
        """Clean up auto-save file on normal shutdown."""
        try:
            if os.path.exists(self.autosave_path):
                os.remove(self.autosave_path)
                logger.debug("Auto-save file cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up auto-save: {str(e)}")
