
import os
import logging
from PyQt5.QtWidgets import QMessageBox
from lib.app_paths import user_data_path
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
        self.autosave_path = user_data_path('config/.autosave.diablos')

        # Initialize diagram service with window context if needed
        # (MainWindow code passed self to DiagramService, so we pass window)

    def new_diagram(self):
        """Create new diagram."""
        if self.diagram_service:
            self.diagram_service.new_diagram()
        elif hasattr(self.window.dsim, 'clear_all'):
            self.window.dsim.clear_all()
        self.window.status_message.setText("New diagram created")
        # cleanup autosave on new? Maybe not necessary depending on logic

    def open_diagram(self):
        """Open diagram."""
        logger.info("open_diagram called")
        if self.diagram_service:
            logger.info("Calling diagram_service.load_diagram()")
            # DiagramService handles dialogs?
            # Existing mainWindow implementation: self.diagram_service.load_diagram()
            result = self.diagram_service.load_diagram()
            logger.info(f"load_diagram returned: {result is not None}")
            if result:
                 self.window.status_message.setText("Diagram opened")
                 # Single source of truth for the Recent Files menu is
                 # RecentFilesManager (config/recent_files.json). Routing here
                 # avoids a second, divergent QSettings-backed store writing the
                 # same menu widget.
                 if getattr(self.window, 'recent_files_manager', None):
                     self.window.recent_files_manager.add(self.diagram_service.current_file)
        else:
            logger.warning("diagram_service is None!")

    def open_example(self, filename):
        """Open an example diagram."""
        if self.diagram_service:
            self.diagram_service.load_diagram(filename)
        self.window.status_message.setText(f"Example {os.path.basename(filename)} opened")

    def save_diagram(self):
        """Save diagram."""
        if self.diagram_service:
            self.diagram_service.save_diagram()
            # Add to recent files if save successful
            # DiagramService usually returns success?
        self.window.status_message.setText("Diagram saved")

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
                    "Recover Auto-save?",
                    "An auto-save file was found. Do you want to recover your previous session?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )

                if reply == QMessageBox.Yes:
                    self.recover_autosave()
                else:
                    os.remove(self.autosave_path)
                    logger.info("Auto-save file deleted (user declined recovery)")
        except Exception as e:
            logger.error(f"Error checking auto-save recovery: {str(e)}")

    def recover_autosave(self):
        """Recover diagram from auto-save file."""
        try:
            # Use diagram service to load
            if self.diagram_service:
                 # DiagramService internally uses file_service.load
                 # We might need to bypass dialog if we just want to load a specific file
                 # DiagramService has load_diagram(filename) support
                 self.diagram_service.load_diagram(self.autosave_path)

            # Remove autosave file after successful recovery
            if os.path.exists(self.autosave_path):
                os.remove(self.autosave_path)

            self.window.canvas.update()
            self.window.status_message.setText("Diagram recovered from auto-save")
            logger.info("Diagram successfully recovered from auto-save")

            QMessageBox.information(
                self.window,
                "Recovery Successful",
                "Your diagram has been successfully recovered from the auto-save file."
            )

        except Exception as e:
            logger.error(f"Error recovering from auto-save: {str(e)}")
            try:
                if os.path.exists(self.autosave_path):
                    os.remove(self.autosave_path)
            except OSError:
                pass
            QMessageBox.warning(
                self.window,
                "Recovery Failed",
                f"Could not recover auto-save file. Starting fresh.\n\nError: {str(e)}"
            )

    def cleanup_autosave(self):
        """Clean up auto-save file on normal shutdown."""
        try:
            if os.path.exists(self.autosave_path):
                os.remove(self.autosave_path)
                logger.debug("Auto-save file cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up auto-save: {str(e)}")
