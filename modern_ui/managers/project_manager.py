
import os
import logging
from PyQt5.QtWidgets import QMessageBox, QAction
from PyQt5.QtCore import QSettings
from lib.services.diagram_service import DiagramService

logger = logging.getLogger(__name__)

class ProjectManager:
    """
    Manages project state, file operations, and autosave.
    """
    def __init__(self, main_window):
        self.window = main_window
        self.diagram_service = DiagramService(self.window)
        self.settings = QSettings("Sapetor", "DiaBloS")
        self.autosave_path = os.path.join(os.path.dirname(os.path.abspath(main_window.__module__)), 'autosave.json')

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
        if self.diagram_service:
            # DiagramService handles dialogs?
            # Existing mainWindow implementation: self.diagram_service.load_diagram()
            if self.diagram_service.load_diagram():
                 self.window.status_message.setText("Diagram opened")
                 self.add_recent_file(self.diagram_service.current_file) # Assuming service tracks it? 
                 # DiagramService doesn't expose current_file easily maybe? 
                 # Let's check DiagramService later. For now assume we need to handle recent files here.

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

    def update_recent_files_menu(self):
        """Update the Recent Files menu."""
        if not hasattr(self.window, 'recent_files_menu'):
            return

        self.window.recent_files_menu.clear()
        recent_files = self.settings.value("recent_files", [], type=list)
        
        # Filter existing files
        recent_files = [f for f in recent_files if os.path.exists(f)]
        self.settings.setValue("recent_files", recent_files)

        for file_path in recent_files:
            action = QAction(os.path.basename(file_path), self.window)
            action.setData(file_path)
            action.setStatusTip(file_path)
            action.triggered.connect(lambda checked, path=file_path: self.diagram_service.load_diagram(path))
            self.window.recent_files_menu.addAction(action)

        if not recent_files:
            self.window.recent_files_menu.addAction("No recent files").setEnabled(False)
            
    def add_recent_file(self, file_path):
        """Add file to recent list."""
        if not file_path: return
        recent_files = self.settings.value("recent_files", [], type=list)
        if file_path in recent_files:
            recent_files.remove(file_path)
        recent_files.insert(0, file_path)
        recent_files = recent_files[:10]  # Keep last 10
        self.settings.setValue("recent_files", recent_files)
        self.update_recent_files_menu()

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
            except:
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
