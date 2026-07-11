"""Shared file-dialog helpers for export/save actions."""

import os

from PyQt5.QtWidgets import QFileDialog


def ask_save_path(parent, caption, default_path, filters):
    """Show a save dialog and return a path guaranteed to carry an extension.

    ``filters`` is an ordered list of ``(qt_filter, extension)`` pairs, e.g.
    ``[("PNG Image (*.png)", ".png"), ("SVG Image (*.svg)", ".svg")]``. A typed
    extension always wins; when the user omits one, the extension matching the
    selected filter (falling back to the first) is appended. Returns ``None``
    when the dialog is cancelled.
    """
    path, selected = QFileDialog.getSaveFileName(
        parent, caption, default_path, ";;".join(f for f, _ in filters)
    )
    if not path:
        return None
    if not os.path.splitext(path)[1]:
        path += next((ext for f, ext in filters if f == selected), filters[0][1])
    return path
