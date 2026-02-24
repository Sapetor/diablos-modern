
import json
import logging
import os
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtCore import QSize

logger = logging.getLogger(__name__)


def _create_styled_file_dialog(parent, title, directory, filters, save=False):
    """Create a styled QFileDialog that works well on all platforms."""
    dialog = QFileDialog(parent, title, directory, filters)

    # Use Qt dialog instead of native (fixes macOS click issues)
    dialog.setOption(QFileDialog.DontUseNativeDialog, True)

    # Set dialog mode
    if save:
        dialog.setAcceptMode(QFileDialog.AcceptSave)
        dialog.setFileMode(QFileDialog.AnyFile)
    else:
        dialog.setAcceptMode(QFileDialog.AcceptOpen)
        dialog.setFileMode(QFileDialog.ExistingFile)

    # Make dialog larger for better usability
    dialog.resize(900, 600)

    # Apply styling to fix visual issues
    dialog.setStyleSheet("""
        QFileDialog {
            background-color: #f5f5f5;
        }
        QFileDialog QListView, QFileDialog QTreeView {
            background-color: white;
            border: 1px solid #ccc;
            border-radius: 4px;
            padding: 4px;
        }
        QFileDialog QListView::item, QFileDialog QTreeView::item {
            padding: 4px;
            border-radius: 2px;
        }
        QFileDialog QListView::item:selected, QFileDialog QTreeView::item:selected {
            background-color: #0078d4;
            color: white;
        }
        QFileDialog QLineEdit {
            padding: 6px;
            border: 1px solid #ccc;
            border-radius: 4px;
            background-color: white;
        }
        QFileDialog QComboBox {
            padding: 6px;
            border: 1px solid #ccc;
            border-radius: 4px;
            background-color: white;
            min-width: 200px;
        }
        QFileDialog QComboBox::drop-down {
            border: none;
            width: 20px;
        }
        QFileDialog QComboBox QAbstractItemView {
            background-color: white;
            border: 1px solid #ccc;
            selection-background-color: #0078d4;
        }
        QFileDialog QPushButton {
            padding: 8px 16px;
            border: 1px solid #ccc;
            border-radius: 4px;
            background-color: #f0f0f0;
            min-width: 80px;
        }
        QFileDialog QPushButton:hover {
            background-color: #e0e0e0;
        }
        QFileDialog QPushButton:pressed {
            background-color: #d0d0d0;
        }
        QFileDialog QToolButton {
            padding: 4px;
            border: 1px solid transparent;
            border-radius: 4px;
        }
        QFileDialog QToolButton:hover {
            background-color: #e0e0e0;
            border: 1px solid #ccc;
        }
        QFileDialog QLabel {
            padding: 2px;
        }
        QFileDialog QSplitter::handle {
            background-color: #ddd;
        }
        QFileDialog QHeaderView::section {
            background-color: #f0f0f0;
            padding: 6px;
            border: none;
            border-bottom: 1px solid #ccc;
        }
    """)

    return dialog

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
        
        # Set default directory to 'examples' relative to current working directory
        project_root = os.getcwd()
        self.last_directory = os.path.join(project_root, 'examples')

        # Fall back to 'saves' if examples doesn't exist
        if not os.path.exists(self.last_directory):
            self.last_directory = os.path.join(project_root, 'saves')
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
            # Use styled Qt dialog (fixes macOS native dialog click issues)
            dialog = _create_styled_file_dialog(
                self.main_window,
                "Save Diagram",
                self.last_directory,
                "DiaBloS Files (*.diablos);;All Files (*)",
                save=True
            )
            if dialog.exec_() == QFileDialog.Accepted:
                files = dialog.selectedFiles()
                filename = files[0] if files else ""
            else:
                filename = ""

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

            with open(filename, 'w', encoding='utf-8') as f:
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
        logger.info(f"load_diagram called, filename={filename}, last_directory={self.last_directory}")
        if not filename:
            logger.info("Opening file dialog...")
            # Use styled Qt dialog (fixes macOS native dialog click issues)
            dialog = _create_styled_file_dialog(
                self.main_window,
                "Open Diagram",
                self.last_directory,
                "DiaBloS Files (*.diablos *.dbs *.json *.dat);;All Files (*)",
                save=False
            )
            if dialog.exec_() == QFileDialog.Accepted:
                files = dialog.selectedFiles()
                filename = files[0] if files else ""
            else:
                filename = ""
            logger.info(f"File dialog returned: {filename}")

        if not filename:
            return None

        self.last_directory = os.path.dirname(filename)

        try:
            with open(filename, 'r', encoding='utf-8') as f:
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
