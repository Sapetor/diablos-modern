
import json
import logging
import os
from PyQt5.QtWidgets import QFileDialog, QMessageBox

logger = logging.getLogger(__name__)

class DiagramService:
    """
    Service responsible for loading and saving diagram state.
    Decouples file I/O from the Main Window UI.
    """

    def __init__(self, main_window):
        """
        Initialize with reference to main window to access dsim and UI elements.
        
        Args:
            main_window: Reference to ModernDiaBloSWindow
        """
        self.main_window = main_window
        self.dsim = main_window.dsim
        
        # Set default directory to 'saves' relative to current working directory
        project_root = os.getcwd()
        self.last_directory = os.path.join(project_root, 'saves')
        
        if not os.path.exists(self.last_directory):
             os.makedirs(self.last_directory, exist_ok=True)
             
        self.current_file = None

    def save_diagram(self, filename=None):
        """
        Save the current diagram to a file.
        
        Args:
            filename (str, optional): Path to save to. If None, asks user.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        if not filename:
            filename, _ = QFileDialog.getSaveFileName(
                self.main_window,
                "Save Diagram",
                self.last_directory,
                "DiaBloS Files (*.dbs *.json *.dat);;All Files (*)"
            )

        if not filename:
            return False

        self.last_directory = os.path.dirname(filename)

        try:
            # Gather state from engine
            diagram_data = self.dsim.serialize()
            
            # Add UI-specific state
            ui_state = {
                "theme": "dark", # simple default or fetch from theme manager
                "zoom_factor": self.main_window.canvas.zoom_factor,
                # Store splitter sizes to restore layout
                "main_splitter_sizes": self.main_window.main_splitter.sizes(),
                "center_splitter_sizes": self.main_window.center_splitter.sizes() if hasattr(self.main_window, 'center_splitter') else []
            }
            diagram_data["ui_state"] = ui_state

            with open(filename, 'w') as f:
                json.dump(diagram_data, f, indent=4)
            
            logger.info(f"Diagram saved to {filename}")
            self.current_file = filename
            return True
            
        except Exception as e:
            logger.error(f"Failed to save diagram: {e}")
            QMessageBox.critical(self.main_window, "Save Error", f"Could not save diagram:\n{str(e)}")
            return False

    def load_diagram(self, filename=None):
        """
        Load a diagram from a file.
        
        Args:
            filename (str, optional): Path to load from. If None, asks user.
            
        Returns:
            dict: Loaded data if successful, None otherwise.
        """
        if not filename:
            filename, _ = QFileDialog.getOpenFileName(
                self.main_window,
                "Open Diagram",
                self.last_directory,
                "DiaBloS Files (*.dbs *.json *.dat);;All Files (*)"
            )

        if not filename:
            return None

        self.last_directory = os.path.dirname(filename)

        try:
            with open(filename, 'r') as f:
                data = json.load(f)

            # Clear current diagram
            # self.dsim.new_diagram() # Engine should handle this?
            # For now, let's assume dsim.load() handles clearing or we need manual clear
            
            self.dsim.deserialize(data)
            
            # Restore UI state if present
            ui_state = data.get("ui_state", {})
            if "zoom_factor" in ui_state:
                self.main_window.set_zoom(ui_state["zoom_factor"])
            
            if "main_splitter_sizes" in ui_state:
                self.main_window.main_splitter.setSizes(ui_state["main_splitter_sizes"])

            logger.info(f"Diagram loaded from {filename}")
            self.current_file = filename
            return data

        except Exception as e:
            logger.error(f"Failed to load diagram: {e}")
            QMessageBox.critical(self.main_window, "Load Error", f"Could not load diagram:\n{str(e)}")
            return None

    def new_diagram(self):
        """Reset the diagram to a clean state."""
        # Confirm with user if unsaved changes? (Future feature)
        self.dsim.new_diagram()
        self.current_file = None
        logger.info("New diagram created")
