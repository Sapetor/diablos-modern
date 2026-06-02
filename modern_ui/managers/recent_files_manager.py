"""
RecentFilesManager -- owns the JSON-backed "Recent Files" list and menu.

Extracted verbatim (behavior-preserving) from ``ModernDiaBloSWindow`` so the
main window keeps only thin one-line facades. Follows the same manager pattern
as the other ``modern_ui/managers`` (constructed with the main window, holds it
as ``self.window``).

The list is persisted to ``user_data_path('config/recent_files.json')`` as
``{"recent_files": [...]}``; it is deduplicated, most-recent-first, and capped
at 10 entries.

Note: this is independent of ``ProjectManager``'s ``QSettings``-based recent
files helpers, which are not wired into the menu. This manager backs the
``Recent Files`` menu the user actually sees (built by ``MenuBuilder`` and the
command palette).
"""

import os
import json
import logging

from PyQt5.QtWidgets import QMessageBox

logger = logging.getLogger(__name__)


class RecentFilesManager:
    """Manages the recent-files list, its JSON persistence, and its menu."""

    MAX_RECENT = 10

    def __init__(self, main_window):
        self.window = main_window

    # -- persistence --------------------------------------------------------

    def load(self):
        """Load recent files list from config."""
        from lib.app_paths import user_data_path
        config_path = user_data_path('config/recent_files.json')
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    data = json.load(f)
                    return data.get('recent_files', [])
        except Exception as e:
            logger.error(f"Error loading recent files: {e}")
        return []

    def save(self, recent_files):
        """Save recent files list to config."""
        from lib.app_paths import user_data_path
        config_path = user_data_path('config/recent_files.json')
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump({'recent_files': recent_files}, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving recent files: {e}")

    # -- list mutation ------------------------------------------------------

    def add(self, filepath):
        """Add a file to the recent files list (front, deduped, capped)."""
        if not filepath:
            return

        recent_files = self.load()

        # Remove if already in list
        if filepath in recent_files:
            recent_files.remove(filepath)

        # Add to front
        recent_files.insert(0, filepath)

        # Keep only last MAX_RECENT
        recent_files = recent_files[:self.MAX_RECENT]

        self.save(recent_files)
        self.update_menu()

    def clear(self):
        """Clear the recent files list."""
        self.save([])
        self.update_menu()
        self.window.status_message.setText("Recent files cleared")

    # -- menu ---------------------------------------------------------------

    def update_menu(self):
        """Update the recent files menu."""
        menu = self.window.recent_files_menu
        menu.clear()

        recent_files = self.load()

        if not recent_files:
            action = menu.addAction("No recent files")
            action.setEnabled(False)
            return

        for filepath in recent_files:
            # Show only filename, but store full path
            filename = os.path.basename(filepath)
            action = menu.addAction(filename)
            action.setData(filepath)
            action.triggered.connect(
                lambda checked, path=filepath: self.open(path)
            )

        menu.addSeparator()
        clear_action = menu.addAction("Clear Recent Files")
        clear_action.triggered.connect(self.clear)

    # -- open ---------------------------------------------------------------

    def open(self, filepath):
        """Open a file from the recent files list."""
        window = self.window
        if os.path.exists(filepath):
            try:
                # Use DSim's file service if available, else fall back.
                if hasattr(window.dsim, 'file_service'):
                    block_data = window.dsim.file_service.load(filepath=filepath)
                    window.dsim.deserialize(block_data)
                else:
                    window.dsim.open(filepath)

                self.add(filepath)
                window.status_message.setText(
                    f"Opened: {os.path.basename(filepath)}"
                )
                logger.info(f"Opening recent file: {filepath}")
                window.canvas.update()

            except Exception as e:
                logger.error(f"Failed to open recent file: {e}")
                QMessageBox.critical(
                    window, "Error", f"Failed to open file:\n{str(e)}"
                )
        else:
            QMessageBox.warning(
                window,
                "File Not Found",
                f"The file '{filepath}' no longer exists."
            )
            # Remove from recent files
            recent_files = self.load()
            if filepath in recent_files:
                recent_files.remove(filepath)
                self.save(recent_files)
                self.update_menu()
