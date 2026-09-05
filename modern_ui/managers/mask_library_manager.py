"""MaskLibraryManager -- the window-side actions behind masks and user libraries.

Five actions, all reachable from the Edit menu and the block context menu:

  * **Edit mask...**          open :class:`MaskEditorDialog` on the selected
                              Subsystem and commit the result (undoable).
  * **Look under mask**       navigate into the subsystem, exactly as an
                              unmasked one.
  * **Save as library block...**  write the subsystem to a library file so it
                              shows up in the palette for any diagram.
  * **Reload from library**   re-sync an instance's contents from its source
                              file while keeping its mask parameter values.
  * **Refresh block library** re-scan the library folders and rebuild the
                              palette.

Every mutation pushes an undo entry and sets the diagram's dirty flag before
touching the block, so undo/redo and the unsaved-changes prompt cover mask
edits like any other diagram change.
"""

import logging
import os
from typing import Any, Optional

from PyQt5.QtWidgets import QFileDialog, QMessageBox

from lib.i18n import tr
from lib.library import (
    LibraryError,
    attach_library_ref,
    discover_library_blocks,
    find_library_block,
    get_library_ref,
    read_library_file,
    default_library_dir,
    library_search_paths,
    slugify,
    write_library_file,
)
from lib.masks import (
    MaskError,
    get_mask,
    is_subsystem,
    mask_parameters,
    refresh_saveable_params,
    set_mask,
)

logger = logging.getLogger(__name__)


class MaskLibraryManager:
    """Mask editing and user-library actions for the main window."""

    def __init__(self, main_window):
        self.window = main_window

    # -- context -----------------------------------------------------------

    @property
    def canvas(self):
        return getattr(self.window, "canvas", None)

    @property
    def dsim(self):
        canvas = self.canvas
        return getattr(canvas, "dsim", None) if canvas is not None else None

    def current_diagram_path(self) -> Optional[str]:
        """Path of the open diagram, for the project-local ``library/`` folder."""
        service = getattr(self.window, "diagram_service", None)
        path = getattr(service, "current_file", None)
        if path:
            return path
        return getattr(self.dsim, "current_filepath", None)

    def selected_subsystem(self, block: Any = None) -> Optional[Any]:
        """The Subsystem to act on: the argument, else the selection."""
        if block is not None and is_subsystem(block):
            return block
        dsim = self.dsim
        if dsim is None:
            return None
        for candidate in getattr(dsim, "blocks_list", []) or []:
            if candidate.selected and is_subsystem(candidate):
                return candidate
        return None

    # -- helpers -----------------------------------------------------------

    def _require_subsystem(self, block, action) -> Optional[Any]:
        target = self.selected_subsystem(block)
        if target is None:
            self._warn(
                tr("No subsystem selected"),
                tr("Select a Subsystem block first, then use '{action}'.", action=action),
            )
        return target

    def _warn(self, title, text):
        logger.info("%s: %s", title, text)
        try:
            QMessageBox.warning(self.window, title, text)
        except Exception:  # pragma: no cover - headless / neutered dialogs
            logger.debug("Could not show warning dialog", exc_info=True)

    def _notify(self, message):
        logger.info(message)
        notify = getattr(self.window, "_notify", None)
        if callable(notify):
            try:
                notify(message)
            except Exception:  # pragma: no cover - status bar is optional
                logger.debug("Could not post status message", exc_info=True)

    def _mark_dirty(self, description):
        canvas = self.canvas
        if canvas is not None and hasattr(canvas, "_push_undo"):
            canvas._push_undo(description)
        dsim = self.dsim
        if dsim is not None:
            dsim.dirty = True

    def _refresh_views(self, block=None):
        canvas = self.canvas
        if canvas is not None:
            canvas.update()
        editor = getattr(self.window, "property_editor", None)
        if editor is not None and block is not None:
            try:
                editor.set_block(block)
            except Exception:  # pragma: no cover - panel refresh is cosmetic
                logger.debug("Could not refresh the property panel", exc_info=True)

    # -- actions -----------------------------------------------------------

    def edit_mask(self, block: Any = None) -> bool:
        """Open the mask editor on ``block`` and apply the result."""
        target = self._require_subsystem(block, tr("Edit Mask..."))
        if target is None:
            return False

        from modern_ui.widgets.mask_editor_dialog import edit_block_mask

        mask = edit_block_mask(target, parent=self.window)
        if mask is None:
            return False
        return self.apply_mask(target, mask)

    def apply_mask(self, block: Any, mask) -> bool:
        """Commit ``mask`` onto ``block`` as one undoable edit."""
        self._mark_dirty("Edit Mask")
        try:
            set_mask(block, mask)
        except MaskError as exc:
            self._warn(tr("Invalid mask"), str(exc))
            return False
        self._refresh_views(block)
        self._notify(tr("Mask updated: {name}", name=mask.get("name", block.name)))
        return True

    def remove_mask(self, block: Any = None) -> bool:
        """Drop the mask from ``block``, leaving a plain subsystem."""
        target = self._require_subsystem(block, tr("Remove Mask"))
        if target is None or get_mask(target) is None:
            return False
        self._mark_dirty("Remove Mask")
        set_mask(target, None)
        self._refresh_views(target)
        self._notify(tr("Mask removed from {name}", name=target.name))
        return True

    def look_under_mask(self, block: Any = None) -> bool:
        """Navigate into the subsystem, mask or not."""
        target = self._require_subsystem(block, tr("Look Under Mask"))
        if target is None:
            return False
        dsim = self.dsim
        if dsim is None or not hasattr(dsim, "enter_subsystem"):
            return False
        dsim.enter_subsystem(target)
        canvas = self.canvas
        if canvas is not None:
            canvas.update()
            # Mirror the canvas double-click path so the view and the
            # breadcrumb trail (fed by scope_changed) stay consistent.
            try:
                canvas.zoom_pan_manager.reset_view()
                canvas.zoom_to_fit()
            except Exception:  # pragma: no cover - view fitting is cosmetic
                logger.debug("Could not refit the view after entering", exc_info=True)
            try:
                canvas.scope_changed.emit(dsim.get_current_path())
            except Exception:  # pragma: no cover - breadcrumb is cosmetic
                logger.debug("Could not update the breadcrumb bar", exc_info=True)
        return True

    # -- library -----------------------------------------------------------

    def save_as_library_block(self, block: Any = None, path: Optional[str] = None) -> Optional[str]:
        """Write the subsystem to a library file; returns the path written.

        ``path`` skips the file dialog (used by tests and scripted callers).
        """
        target = self._require_subsystem(block, tr("Save as Library Block..."))
        if target is None:
            return None

        mask = get_mask(target)
        suggested = slugify((mask or {}).get("name") or target.username or target.name)

        if path is None:
            path = self._ask_library_path(
                os.path.join(default_library_dir(), suggested + ".diablos")
            )
            if not path:
                return None

        from lib.services.file_service import FileService

        model = getattr(self.dsim, "model", None)
        try:
            block_data = FileService(model)._serialize_block(target)
        except Exception as exc:
            self._warn(
                tr("Could not save library block"), tr("Serialization failed: {error}", error=exc)
            )
            return None
        return self._write(target, path, mask, block_data)

    def _ask_library_path(self, suggested_path) -> Optional[str]:
        try:
            os.makedirs(os.path.dirname(suggested_path), exist_ok=True)
        except OSError:
            logger.debug("Could not pre-create the library folder", exc_info=True)
        chosen, _ = QFileDialog.getSaveFileName(
            self.window,
            tr("Save as Library Block"),
            suggested_path,
            tr("DiaBloS Library Blocks") + " (*.diablos);;" + tr("All Files") + " (*)",
        )
        if not chosen:
            return None
        if not chosen.lower().endswith(".diablos"):
            chosen += ".diablos"
        return chosen

    def _write(self, target, path, mask, block_data) -> Optional[str]:
        try:
            written = write_library_file(
                block_data,
                directory=os.path.dirname(path) or None,
                block_id=os.path.splitext(os.path.basename(path))[0],
                mask=mask,
            )
        except LibraryError as exc:
            self._warn(tr("Could not save library block"), str(exc))
            return None

        # Stamp the instance from the file we just wrote, not from a rescan:
        # the user may have saved outside the search paths, and the block is
        # still legitimately the source of this instance.
        lib_block = read_library_file(written)
        if lib_block is not None:
            attach_library_ref(target, lib_block.reference())
            refresh_saveable_params(target)

        self.refresh_library()

        folder = os.path.dirname(os.path.abspath(written))
        searched = [os.path.abspath(p) for p in library_search_paths(self.current_diagram_path())]
        if folder not in searched:
            self._notify(
                tr(
                    "Saved library block to {path} -- it is outside the library search "
                    "path, so it will not appear in the palette until you point "
                    "DIABLOS_LIBRARY_PATH at that folder.",
                    path=written,
                )
            )
        else:
            self._notify(tr("Saved library block to {path}", path=written))
        return written

    def reload_from_library(self, block: Any = None) -> bool:
        """Re-sync an instance's contents from its library file.

        The instance's mask parameter *values* are preserved -- only the
        contents (and the mask definition) come from the file -- so a tuned
        instance keeps its tuning across a library update.
        """
        target = self._require_subsystem(block, tr("Reload from Library"))
        if target is None:
            return False
        ref = get_library_ref(target)
        if not ref:
            self._warn(
                tr("Not a library instance"),
                tr(
                    "This subsystem was not created from a library block, so there is "
                    "nothing to reload from."
                ),
            )
            return False

        lib_block = find_library_block(ref.get("id", ""), self.current_diagram_path())
        if lib_block is None:
            self._warn(
                tr("Library block not found"),
                tr(
                    "No library block with id '{block_id}' was found in:\n  {paths}\n\n"
                    "The diagram still works: library instances are self-contained "
                    "copies.",
                    block_id=ref.get("id", "?"),
                    paths="\n  ".join(library_search_paths(self.current_diagram_path())),
                ),
            )
            return False

        # Keep the instance's own values for parameters the new mask still has.
        kept = {
            spec["name"]: target.params[spec["name"]]
            for spec in mask_parameters(get_mask(target))
            if spec["name"] in target.params
        }

        self._mark_dirty("Reload from Library")

        data = lib_block.instance_block_data()
        from lib.services.file_service import FileService

        model = getattr(self.dsim, "model", None)
        fresh = FileService(model)._construct_block(data)
        if fresh is None:
            self._warn(
                tr("Reload failed"), tr("The library file could not be rebuilt into a subsystem.")
            )
            return False

        target.sub_blocks = fresh.sub_blocks
        target.sub_lines = fresh.sub_lines
        target.ports = getattr(fresh, "ports", {}) or {}
        target.ports_map = getattr(fresh, "ports_map", {}) or {}
        target.in_ports = fresh.in_ports
        target.out_ports = fresh.out_ports

        if lib_block.mask is not None:
            set_mask(target, lib_block.mask)
            for name, value in kept.items():
                if name in target.params:
                    target.params[name] = value
        attach_library_ref(target, lib_block.reference())
        refresh_saveable_params(target)
        try:
            target.update_Block()
        except Exception:  # pragma: no cover - geometry refresh is best-effort
            logger.debug("update_Block failed after library reload", exc_info=True)

        self._refresh_views(target)
        self._notify(tr("Reloaded {name} from {file}", name=target.name, file=lib_block.file_name))
        return True

    def refresh_library(self) -> int:
        """Re-scan the library folders and rebuild the palette."""
        dsim = self.dsim
        model = getattr(dsim, "model", None)
        count = 0
        if model is not None and hasattr(model, "load_library_blocks"):
            count = model.load_library_blocks(self.current_diagram_path())
            if dsim is not None:
                dsim.menu_blocks = model.menu_blocks
        palette = getattr(self.window, "block_palette", None) or getattr(
            self.window, "palette_widget", None
        )
        if palette is not None and hasattr(palette, "refresh_blocks"):
            palette.refresh_blocks()
        return count

    def library_blocks(self):
        """Discovered library blocks (thin passthrough used by the UI/tests)."""
        return discover_library_blocks(self.current_diagram_path())
