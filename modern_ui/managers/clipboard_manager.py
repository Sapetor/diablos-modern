"""
Clipboard Manager for ModernCanvas
Handles copy, paste, cut, and duplicate operations.
"""

import logging
import copy
from PyQt5.QtCore import QRect, QPoint

from blocks.subsystem import Subsystem
from lib.i18n import tr
from lib.masks import apply_mask_appearance
from lib.simulation.block import DBlock
from lib.simulation.connection import DLine

logger = logging.getLogger(__name__)

# Keyboard paste (Ctrl+V) lands this far right/down from the copied position so
# the pasted blocks don't sit exactly on top of the originals.
KEYBOARD_PASTE_OFFSET = 30


def _paste_offset(pos, first_coords):
    """Translation applied to every copied rect.

    ``pos`` (context-menu "paste here") puts the first copied block's top-left
    on it; ``None`` (Ctrl+V) uses the fixed keyboard offset.
    """
    if pos is None:
        return QPoint(KEYBOARD_PASTE_OFFSET, KEYBOARD_PASTE_OFFSET)
    return QPoint(pos.x() - first_coords.x(), pos.y() - first_coords.y())


def _next_block_sid(blocks_list, block_fn):
    """Next free numeric id for ``block_fn`` (same rule as SimulationModel.add_block)."""
    id_list = [
        int(b_elem.name[len(b_elem.block_fn) :])
        for b_elem in blocks_list
        if b_elem.block_fn == block_fn
    ]
    return max(id_list) + 1 if id_list else 0


def _next_line_sid(line_list):
    """Next free line id."""
    line_ids = [line.sid for line in line_list]
    return max(line_ids) + 1 if line_ids else 0


def _find_block_class(menu_blocks, block_fn):
    """Block class of the palette entry for ``block_fn`` (None when there is none)."""
    for menu_block in menu_blocks:
        if menu_block.block_fn == block_fn:
            return menu_block.block_class
    return None


def _user_param_keys(params):
    """Keys that DBlock would put in ``init_params_list``: everything not ``_dunder_``."""
    return [key for key in params if not (key.startswith("_") and key.endswith("_"))]


def _resolve_endpoints(conn_data, pasted_blocks):
    """Map a copied connection onto the pasted blocks.

    Returns ``(start_block, start_port, end_block, end_port)``, or ``None``
    (after logging a warning) when a block or port index is out of range.
    """
    start_idx = conn_data["start_index"]
    end_idx = conn_data["end_index"]
    start_port = conn_data["start_port"]
    end_port = conn_data["end_port"]
    n_pasted = len(pasted_blocks)

    logger.info(
        f"Paste: Connection start_idx={start_idx}, end_idx={end_idx}, ports=({start_port},{end_port}), pasted_blocks len={n_pasted}"
    )
    if start_idx >= n_pasted:
        logger.warning(
            f"Skipping connection: start_index {start_idx} >= pasted_blocks length {n_pasted}"
        )
        return None
    if end_idx >= n_pasted:
        logger.warning(
            f"Skipping connection: end_index {end_idx} >= pasted_blocks length {n_pasted}"
        )
        return None

    start_block = pasted_blocks[start_idx]
    end_block = pasted_blocks[end_idx]
    if start_port >= len(start_block.out_coords):
        logger.warning(
            f"Skipping connection: start_port {start_port} >= out_coords length {len(start_block.out_coords)} for {start_block.name}"
        )
        return None
    if end_port >= len(end_block.in_coords):
        logger.warning(
            f"Skipping connection: end_port {end_port} >= in_coords length {len(end_block.in_coords)} for {end_block.name}"
        )
        return None
    return start_block, start_port, end_block, end_port


class ClipboardManager:
    """
    Manages clipboard operations for blocks and connections.
    """

    def __init__(self, canvas):
        self.canvas = canvas
        self.dsim = canvas.dsim
        self.clipboard_blocks = []
        self.clipboard_connections = []

    def copy_selected_blocks(self):
        """Copy selected blocks to clipboard."""
        try:
            # Find all selected blocks
            selected_blocks = [block for block in self.dsim.blocks_list if block.selected]

            if not selected_blocks:
                logger.info("No blocks selected to copy")
                return

            # Deep copy the block data (not the actual block objects)
            self.clipboard_blocks = []
            selected_indices = {}  # Map original block object to its index in clipboard

            # 1. Copy Blocks
            for i, block in enumerate(selected_blocks):
                block_data = {
                    "block_fn": block.block_fn,
                    "coords": QRect(block.left, block.top, block.width, block.height_base),
                    "color": block.b_color.name(),
                    "category": getattr(block, "category", "Other"),
                    "in_ports": block.in_ports,
                    "out_ports": block.out_ports,
                    "b_type": block.b_type,
                    "io_edit": block.io_edit,
                    "fn_name": block.fn_name,
                    "params": copy.deepcopy(block.params),
                    "external": block.external,
                    "flipped": getattr(block, "flipped", False),
                }

                # SPECIAL HANDLING FOR SUBSYSTEM COPY
                if block.block_fn == "Subsystem":
                    # We need to deepcopy the internal structure!
                    # block.sub_blocks and block.sub_lines contain DBlock/DLine objects.
                    # We can use copy.deepcopy for this as they should be pickleable (mostly).
                    try:
                        block_data["sub_blocks"] = copy.deepcopy(block.sub_blocks)
                        block_data["sub_lines"] = copy.deepcopy(block.sub_lines)
                        block_data["ports"] = (
                            copy.deepcopy(block.ports) if hasattr(block, "ports") else {}
                        )
                        block_data["ports_map"] = (
                            copy.deepcopy(block.ports_map) if hasattr(block, "ports_map") else {}
                        )
                    except Exception as e:
                        logger.error(f"Error deepcopying subsystem contents for {block.name}: {e}")
                        # Fallback? If we fail, paste will create empty subsystem.

                self.clipboard_blocks.append(block_data)
                selected_indices[block] = i

            # 2. Copy Internal Connections
            self.clipboard_connections = []

            # Map names to objects for lookup
            name_to_block = {b.name: b for b in self.dsim.blocks_list}

            # Debug: Log state for diagnosing connection copy issues
            logger.info(
                f"Copy: {len(self.dsim.line_list)} connections in dsim, {len(selected_blocks)} selected blocks"
            )
            logger.info(f"Copy: Selected block names: {[b.name for b in selected_blocks]}")

            for line in self.dsim.line_list:
                # Resolve block objects from names
                src_obj = name_to_block.get(line.srcblock)
                dst_obj = name_to_block.get(line.dstblock)

                # Debug logging for each connection
                if not src_obj:
                    logger.debug(
                        f"Copy: Connection {line.name} srcblock '{line.srcblock}' not found in blocks"
                    )
                if not dst_obj:
                    logger.debug(
                        f"Copy: Connection {line.name} dstblock '{line.dstblock}' not found in blocks"
                    )
                if src_obj and src_obj not in selected_indices:
                    logger.debug(
                        f"Copy: Connection {line.name} srcblock '{line.srcblock}' not selected"
                    )
                if dst_obj and dst_obj not in selected_indices:
                    logger.debug(
                        f"Copy: Connection {line.name} dstblock '{line.dstblock}' not selected"
                    )

                if (
                    src_obj
                    and dst_obj
                    and src_obj in selected_indices
                    and dst_obj in selected_indices
                ):
                    start_idx = selected_indices[src_obj]
                    end_idx = selected_indices[dst_obj]
                    conn_data = {
                        "start_index": start_idx,
                        "start_port": line.srcport,
                        "end_index": end_idx,
                        "end_port": line.dstport,
                    }
                    self.clipboard_connections.append(conn_data)
                    logger.info(
                        f"Copy: Connection {line.srcblock}:{line.srcport} -> {line.dstblock}:{line.dstport} stored with indices ({start_idx}, {end_idx})"
                    )

            logger.info(
                f"Copied {len(self.clipboard_blocks)} blocks and {len(self.clipboard_connections)} connections"
            )
            # Log connection indices for debugging
            for i, conn in enumerate(self.clipboard_connections):
                logger.info(
                    f"  Connection {i}: start_index={conn['start_index']}, end_index={conn['end_index']}"
                )
        except Exception as e:
            logger.error(f"Error copying blocks: {str(e)}")
            if hasattr(self.canvas, "simulation_status_changed"):
                self.canvas.simulation_status_changed.emit(tr("Copy failed: {error}", error=e))

    def paste_blocks(self, pos=None):
        """Paste blocks from clipboard.

        Args:
            pos: Optional world-coordinate QPoint to paste at (context-menu
                "paste here"): the first copied block's top-left lands on it.
                When None (keyboard Ctrl+V), paste at a +30,+30 offset from
                the copied position so pasted blocks don't overlap exactly.
        """
        try:
            if not self.clipboard_blocks:
                logger.info("Clipboard is empty")
                return

            # Push undo state before pasting
            if hasattr(self.canvas, "history_manager"):
                self.canvas.history_manager.push_undo("Paste")
            self._deselect_all_blocks()

            offset = _paste_offset(pos, self.clipboard_blocks[0]["coords"])
            pasted_blocks = self._instantiate_pasted_blocks(offset)
            self._recreate_connections(pasted_blocks)
            self._finish_paste(pasted_blocks)

        except Exception as e:
            logger.error(f"Error pasting blocks: {str(e)}")
            if hasattr(self.canvas, "simulation_status_changed"):
                self.canvas.simulation_status_changed.emit(tr("Paste failed: {error}", error=e))

    def _deselect_all_blocks(self):
        """Clear the selection so only the pasted blocks end up selected."""
        for block in self.dsim.blocks_list:
            block.selected = False

    def _instantiate_pasted_blocks(self, offset):
        """Create one new block per clipboard entry and append it to the diagram in order."""
        pasted_blocks = []
        for block_data in self.clipboard_blocks:
            new_block = self._instantiate_block(block_data, block_data["coords"].translated(offset))
            new_block.flipped = block_data["flipped"]
            new_block.selected = True  # Select the pasted blocks
            self._keep_mask_appearance(new_block)
            self.dsim.blocks_list.append(new_block)
            pasted_blocks.append(new_block)
        return pasted_blocks

    def _instantiate_block(self, block_data, coords):
        """Build the DBlock or Subsystem for one clipboard entry at ``coords``."""
        block_fn = block_data["block_fn"]
        # Computed against the live list, so blocks pasted earlier in this batch count.
        sid = _next_block_sid(self.dsim.blocks_list, block_fn)
        if block_fn == "Subsystem":
            return self._build_subsystem(block_data, sid, coords)
        block_class = _find_block_class(self.dsim.menu_blocks, block_fn)
        return self._build_dblock(block_data, sid, coords, block_class)

    def _build_dblock(self, block_data, sid, coords, block_class):
        """Plain block: a DBlock reconstructed from the copied attributes."""
        return DBlock(
            block_fn=block_data["block_fn"],
            sid=sid,
            coords=coords,
            color=block_data["color"],
            in_ports=block_data["in_ports"],
            out_ports=block_data["out_ports"],
            b_type=block_data["b_type"],
            io_edit=block_data["io_edit"],
            fn_name=block_data["fn_name"],
            # Deep copy so repeated pastes get independent (possibly nested) params
            params=copy.deepcopy(block_data["params"]),
            external=block_data["external"],
            username="",  # Let it default to new name
            block_class=block_class,
            colors=self.dsim.colors,
            category=block_data.get("category", "Other"),
        )

    def _build_subsystem(self, block_data, sid, coords):
        """Subsystem: built with its own constructor, then attributes and contents restored."""
        new_block = Subsystem(
            block_name=f"Subsystem{sid}",
            sid=sid,
            coords=coords,
            color=block_data["color"],
        )
        new_block.io_edit = block_data["io_edit"]
        new_block.fn_name = block_data["fn_name"]
        # Deep copy so repeated pastes get independent (possibly nested) params
        new_block.params = copy.deepcopy(block_data["params"])
        # Subsystem() starts with an empty params dict, so its init_params_list
        # (which gates saving_params) would drop everything pasted here -- the
        # mask, its parameter values, a library back-reference -- on the next
        # save. Recompute it with the same rule DBlock uses.
        new_block.init_params_list = _user_param_keys(new_block.params)
        new_block.params["_name_"] = new_block.name  # Ensure params name matches
        new_block.external = block_data["external"]
        new_block.category = block_data.get("category", "Other")
        if "sub_blocks" in block_data:
            self._restore_subsystem_contents(new_block, block_data)
        self._restore_subsystem_ports(new_block, block_data)
        return new_block

    @staticmethod
    def _restore_subsystem_contents(new_block, block_data):
        """Copy the internal blocks and lines captured at copy time onto ``new_block``."""
        try:
            new_block.sub_blocks = copy.deepcopy(block_data["sub_blocks"])
            new_block.sub_lines = copy.deepcopy(block_data["sub_lines"])
            logger.info(
                f"Restored {len(new_block.sub_blocks)} internal blocks for {new_block.name}"
            )
        except Exception as e:
            logger.error(f"Error restoring subsystem contents for {new_block.name}: {e}")

    @staticmethod
    def _restore_subsystem_ports(new_block, block_data):
        """Give the pasted Subsystem the copied external ports and their geometry.

        ``Subsystem()`` is constructed with 0 in/out ports and no ``ports`` dict,
        so without this step ``in_coords``/``out_coords`` stay empty and every
        copied connection touching the subsystem is skipped when the lines are
        recreated. Mirrors ``FileService._construct_subsystem``: restore the
        port counts, the ``ports`` layout and the index->name map, then run
        ``update_Block()`` so the port coordinates exist before any line is built.
        """
        new_block.in_ports = block_data.get("in_ports", 0)
        new_block.out_ports = block_data.get("out_ports", 0)
        new_block.ports = copy.deepcopy(block_data.get("ports") or {})
        new_block.ports_map = copy.deepcopy(block_data.get("ports_map") or {})
        try:
            new_block.update_Block()
        except Exception as e:
            logger.error(
                f"Subsystem update_Block failed after paste (name={new_block.name}): {e}. "
                f"Pasted subsystem may have stale/empty port geometry."
            )

    @staticmethod
    def _keep_mask_appearance(new_block):
        """A pasted masked subsystem keeps showing its mask name."""
        try:
            apply_mask_appearance(new_block)
        except Exception as e:  # pragma: no cover - defensive
            logger.debug(f"Could not apply mask appearance on paste: {e}")

    def _recreate_connections(self, pasted_blocks):
        """Re-point every copied connection at the pasted blocks and add it to the diagram."""
        logger.info(
            f"Paste: {len(pasted_blocks)} pasted blocks, {len(self.clipboard_connections)} connections to recreate"
        )
        logger.info(f"Paste: clipboard_blocks has {len(self.clipboard_blocks)} entries")
        for conn_data in self.clipboard_connections:
            try:
                self._paste_connection(conn_data, pasted_blocks)
            except IndexError:
                logger.warning("Skipping connection: Block index out of range")
            except Exception as e:
                logger.error(f"Error pasting connection: {e}")

    def _paste_connection(self, conn_data, pasted_blocks):
        """Create one DLine between two pasted blocks; skipped when the endpoints don't resolve."""
        endpoints = _resolve_endpoints(conn_data, pasted_blocks)
        if endpoints is None:
            return
        start_block, start_port, end_block, end_port = endpoints

        new_sid = _next_line_sid(self.dsim.line_list)
        new_line = DLine(
            sid=new_sid,
            srcblock=start_block.name,
            srcport=start_port,
            dstblock=end_block.name,
            dstport=end_port,
            # Minimal points (start/end); update_line computes the real path
            points=[start_block.out_coords[start_port], end_block.in_coords[end_port]],
        )
        # Add to list FIRST so update_line can see it if it needs to check existence
        self.dsim.line_list.append(new_line)
        try:
            # Pass the full block list including new ones
            new_line.update_line(self.dsim.blocks_list)
        except Exception as e:
            logger.warning(f"Failed to update trajectory for pasted line {new_sid}: {e}")

    def _finish_paste(self, pasted_blocks):
        """Mark the diagram dirty, redraw, and announce the first pasted block."""
        self.dsim.dirty = True
        self.canvas.update()
        logger.info(f"Pasted {len(pasted_blocks)} block(s)")
        if pasted_blocks:
            self.canvas.block_selected.emit(pasted_blocks[0])

    def cut_selected_blocks(self):
        """Cut selected blocks to clipboard."""
        self.copy_selected_blocks()
        if hasattr(self.canvas, "selection_manager"):
            self.canvas.selection_manager.remove_selected_items()

    def _duplicate_block(self, block):
        """Duplicate a single block.

        Args:
            block: The block to duplicate
        """
        try:
            from lib.simulation.menu_block import MenuBlocks

            # Push undo state before duplication
            if hasattr(self.canvas, "history_manager"):
                self.canvas.history_manager.push_undo("Duplicate")

            offset = 30  # Offset for duplicated block
            new_position = QPoint(
                block.rect.x() + block.rect.width() // 2 + offset,
                block.rect.y() + block.rect.height() // 2 + offset,
            )

            # Create a MenuBlocks object from the existing block
            io_params = {
                "inputs": block.in_ports,
                "outputs": block.out_ports,
                "b_type": block.b_type,
                "io_edit": block.io_edit,
            }

            menu_block = MenuBlocks(
                block_fn=block.block_fn,
                fn_name=block.fn_name,
                io_params=io_params,
                ex_params=block.params.copy() if hasattr(block, "params") else {},
                b_color=block.b_color,
                coords=(block.rect.width(), block.rect.height()),
                external=block.external,
                block_class=getattr(block, "block_class", None),
            )
            # Preserve category so add_block resolves the correct theme color.
            menu_block.category = getattr(block, "category", "Other")

            # Use add_block with the MenuBlocks object
            new_block = self.dsim.add_block(menu_block, new_position)

            if new_block:
                logger.info(f"Duplicated block: {block.fn_name} -> {new_block.name}")
                if hasattr(self.canvas, "selection_manager"):
                    self.canvas.selection_manager.clear_selections()
                new_block.selected = True
                self.canvas.update()

        except Exception as e:
            logger.error(f"Error duplicating block: {str(e)}")
