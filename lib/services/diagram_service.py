
import json
import logging
import os
import re
import sys
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtCore import QSize

logger = logging.getLogger(__name__)


def _create_styled_file_dialog(parent, title, directory, filters, save=False):
    """Create a styled QFileDialog that works well on all platforms."""
    dialog = QFileDialog(parent, title, directory)

    # Use Qt dialog instead of native (fixes macOS click issues)
    dialog.setOption(QFileDialog.DontUseNativeDialog, True)

    # Apply name filters explicitly. Passing the filter via the constructor
    # is unreliable when DontUseNativeDialog is toggled afterward — on some
    # Qt5/macOS builds the first filter group is dropped or not selected,
    # which makes saved diagrams (*.diablos) look invisible in the dialog.
    name_filters = [f.strip() for f in filters.split(';;') if f.strip()]
    if name_filters:
        dialog.setNameFilters(name_filters)
        dialog.selectNameFilter(name_filters[0])

    if save:
        dialog.setAcceptMode(QFileDialog.AcceptSave)
        dialog.setFileMode(QFileDialog.AnyFile)
        # Default-suffix the first wildcard in the first filter so users
        # who type "myfile" still get the right extension.
        first = name_filters[0] if name_filters else ''
        m = re.search(r'\*\.([A-Za-z0-9]+)', first)
        if m:
            dialog.setDefaultSuffix(m.group(1))
    else:
        dialog.setAcceptMode(QFileDialog.AcceptOpen)
        dialog.setFileMode(QFileDialog.ExistingFile)

    # Make dialog larger for better usability
    dialog.resize(900, 600)

    # Theme-aware styling — replaces hardcoded light-mode colors so the
    # dialog reads correctly in dark mode too.
    from modern_ui.themes.theme_manager import theme_manager
    qss = """
        QFileDialog {
            background-color: @background_secondary;
            color: @text_primary;
        }
        QFileDialog QLabel {
            color: @text_primary;
            background: transparent;
            padding: 2px;
        }
        QFileDialog QListView, QFileDialog QTreeView {
            background-color: @surface;
            color: @text_primary;
            border: 1px solid @border_primary;
            border-radius: 4px;
            padding: 4px;
            alternate-background-color: @surface_variant;
        }
        QFileDialog QListView::item, QFileDialog QTreeView::item {
            padding: 4px;
            border-radius: 2px;
            color: @text_primary;
        }
        QFileDialog QListView::item:hover, QFileDialog QTreeView::item:hover {
            background-color: @background_tertiary;
        }
        QFileDialog QListView::item:selected, QFileDialog QTreeView::item:selected {
            background-color: @accent_primary;
            color: white;
        }
        QFileDialog QLineEdit {
            padding: 6px;
            border: 1px solid @border_primary;
            border-radius: 4px;
            background-color: @surface;
            color: @text_primary;
            selection-background-color: @accent_primary;
            selection-color: white;
        }
        QFileDialog QLineEdit:focus { border-color: @border_focus; }
        QFileDialog QComboBox {
            padding: 6px;
            border: 1px solid @border_primary;
            border-radius: 4px;
            background-color: @surface;
            color: @text_primary;
            min-width: 200px;
        }
        QFileDialog QComboBox:hover  { border-color: @border_hover; }
        QFileDialog QComboBox::drop-down { border: none; width: 20px; }
        QFileDialog QComboBox::down-arrow {
            image: none;
            border-left: 3.5px solid transparent;
            border-right: 3.5px solid transparent;
            border-top: 5px solid @text_secondary;
            margin-right: 8px;
        }
        QFileDialog QComboBox QAbstractItemView {
            background-color: @surface;
            color: @text_primary;
            border: 1px solid @border_primary;
            selection-background-color: @accent_primary;
            selection-color: white;
        }
        QFileDialog QPushButton {
            padding: 8px 16px;
            border: 1px solid @border_primary;
            border-radius: 4px;
            background-color: @surface;
            color: @text_primary;
            min-width: 80px;
        }
        QFileDialog QPushButton:hover {
            background-color: @background_tertiary;
            border-color: @border_hover;
        }
        QFileDialog QPushButton:pressed {
            background-color: @accent_pressed;
            color: white;
        }
        QFileDialog QPushButton:default {
            background-color: @accent_primary;
            color: white;
            border-color: @accent_secondary;
        }
        QFileDialog QPushButton:default:hover { background-color: @accent_hover; }
        QFileDialog QToolButton {
            padding: 4px;
            border: 1px solid transparent;
            border-radius: 4px;
            color: @text_primary;
        }
        QFileDialog QToolButton:hover {
            background-color: @background_tertiary;
            border-color: @border_hover;
        }
        QFileDialog QSplitter::handle { background-color: @border_primary; }
        QFileDialog QHeaderView::section {
            background-color: @surface_variant;
            color: @text_primary;
            padding: 6px;
            border: none;
            border-bottom: 1px solid @border_primary;
        }
    """
    for var, color in sorted(theme_manager.get_qss_variables().items(),
                             key=lambda x: len(x[0]), reverse=True):
        qss = qss.replace(var, color)
    dialog.setStyleSheet(qss)

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
        # In frozen mode, use bundled examples or ~/Documents/DiaBloS
        if getattr(sys, 'frozen', False):
            from lib.app_paths import resource_path
            self.last_directory = resource_path('examples')
            if not os.path.exists(self.last_directory):
                self.last_directory = os.path.expanduser('~/Documents/DiaBloS')
                os.makedirs(self.last_directory, exist_ok=True)
        else:
            project_root = os.getcwd()
            self.last_directory = os.path.join(project_root, 'examples')
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
