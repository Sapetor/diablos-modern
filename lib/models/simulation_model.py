"""
SimulationModel - Data layer for DiaBloS simulation.
Manages blocks, lines, and diagram state.
"""

import logging
import copy
from typing import List, Dict, Optional, Tuple, Any
from PyQt6.QtGui import QColor
from PyQt6.QtCore import QRect, QPoint
from lib.simulation.block import DBlock
from lib.simulation.connection import DLine
from lib.block_loader import load_blocks
from lib.user_blocks import is_user_block, user_block_source
from lib.simulation.menu_block import MenuBlocks

# Import block size configuration
from config.block_sizes import get_block_size

logger = logging.getLogger(__name__)


class SimulationModel:
    """
    Data model for simulation diagrams.
    Manages blocks, connections, and diagram state without UI dependencies.

    Attributes:
        colors: Color palette for blocks
        menu_blocks: Available block types from blocks/ directory
        blocks_list: Instantiated blocks in the current diagram
        line_list: Connections between blocks
        dirty: Flag indicating if diagram has unsaved changes
    """

    def __init__(self) -> None:
        """Initialize the simulation model with empty state."""
        logger.debug("Initializing SimulationModel")

        # Color palette for blocks
        self.colors: Dict[str, QColor] = {
            "black": QColor(0, 0, 0),
            "red": QColor(255, 0, 0),
            "green": QColor(0, 255, 0),
            "blue": QColor(0, 0, 255),
            "yellow": QColor(255, 255, 0),
            "magenta": QColor(255, 0, 255),
            "cyan": QColor(0, 255, 255),
            "purple": QColor(128, 0, 255),
            "orange": QColor(255, 128, 0),
            "aqua": QColor(0, 255, 128),
            "pink": QColor(255, 0, 128),
            "lime_green": QColor(128, 255, 0),
            "light_blue": QColor(0, 128, 255),
            "dark_red": QColor(128, 0, 0),
            "dark_green": QColor(0, 128, 0),
            "dark_blue": QColor(0, 0, 128),
            "dark_gray": QColor(64, 64, 64),
            "gray": QColor(128, 128, 128),
            "light_gray": QColor(192, 192, 192),
            "white": QColor(255, 255, 255),
        }

        # Data containers
        self.menu_blocks: List[MenuBlocks] = []  # Available block types
        self.blocks_list: List[DBlock] = []  # Instantiated blocks in diagram
        self.line_list: List[DLine] = []  # Connections between blocks

        # State flags
        self.dirty: bool = False  # Has diagram been modified?

        # Load available block types
        self.load_all_blocks()
        # ...then the user's own library blocks (see lib/library.py). Missing
        # folders are simply skipped, so this is a no-op for a fresh install.
        self.load_library_blocks()

    def _get_category_color(self, category: str) -> QColor:
        """
        Get theme-aware color for a block category.

        Args:
            category: Block category (Sources, Math, Control, Sinks, Other)

        Returns:
            QColor from theme manager for the category
        """
        from lib.theming.theme_manager import theme_manager

        category_lower = category.lower() if isinstance(category, str) else str(category).lower()

        if "source" in category_lower:
            return theme_manager.get_color("block_source")
        elif "math" in category_lower:
            return theme_manager.get_color("block_process")
        elif "control" in category_lower:
            return theme_manager.get_color("block_control")
        elif "sink" in category_lower:
            return theme_manager.get_color("block_sink")
        elif "routing" in category_lower:
            return theme_manager.get_color("block_routing")
        elif "analysis" in category_lower:
            return theme_manager.get_color("block_analysis")
        elif "pde" in category_lower:
            return theme_manager.get_color("block_pde")
        elif "optim" in category_lower:
            return theme_manager.get_color("block_optimization")
        else:
            return theme_manager.get_color("block_other")

    def load_all_blocks(
        self, diagram_path: Optional[str] = None, reload_user: bool = False
    ) -> None:
        """
        Load all available block types (built-in and user) with theme-aware colors.
        Creates MenuBlock instances for each available block type.

        ``diagram_path`` lets ``lib.user_blocks`` also scan a ``blocks/`` folder
        next to the open diagram; ``reload_user`` re-reads user modules from
        disk. The list is mutated in place because DSim aliases it (see
        ``lib/lib.py``), so rebinding it here would leave stale palette entries
        behind after a reload.
        """
        self.menu_blocks[:] = []
        block_classes = load_blocks(diagram_path, reload_user=reload_user)

        for block_class in block_classes:
            block = block_class()

            # Determine I/O editability - only blocks that explicitly declare
            # io_editable get port editing; all others default to 'none'
            io_editable = block.io_editable
            io_edit = io_editable if io_editable is not None else "none"

            # Get block type
            b_type = getattr(block, "b_type", 2)

            # Process parameters
            ex_params = {}
            if hasattr(block, "params") and block.params:
                for param_name, param_info in block.params.items():
                    if isinstance(param_info, dict) and "default" in param_info:
                        ex_params[param_name] = param_info["default"]
                    else:
                        ex_params[param_name] = param_info
            param_metadata = getattr(block, "params", {})

            # Determine function name
            if hasattr(block, "fn_name"):
                fn_name = block.fn_name
            else:
                fn_name = block.block_name.lower()

            # Get category and determine theme-aware color
            category = getattr(block, "category", "Other")
            block_color = self._get_category_color(category)

            # Get block-specific size from configuration
            block_size = get_block_size(block.block_name)

            menu_block = MenuBlocks(
                block_fn=block.block_name,
                fn_name=fn_name,
                io_params={
                    "inputs": len(block.inputs),
                    "outputs": len(block.outputs),
                    "b_type": b_type,
                    "io_edit": io_edit,
                },
                ex_params=ex_params,
                b_color=block_color,
                coords=block_size,  # Use configured block size
                external=getattr(block, "external", False),
                block_class=block_class,
                colors=self.colors,
            )

            # Store category on menu block for later reference
            menu_block.category = category
            # User blocks (lib/user_blocks.py) are flagged so the palette can
            # mark them and the reload action can count them.
            menu_block.user_block = is_user_block(block_class)
            menu_block.source_file = user_block_source(block_class)
            # Store full param metadata for tooltips
            menu_block.param_meta = param_metadata
            self.menu_blocks.append(menu_block)

    @staticmethod
    def _parse_id_suffix(name: str, prefix_len: int) -> Optional[int]:
        """
        Parse the trailing numeric suffix of a name (the part after prefix_len chars).

        Args:
            name: Full element name (e.g. 'Gain3', 'Line12')
            prefix_len: Number of leading characters that make up the prefix

        Returns:
            The integer suffix, or None if the suffix is not a valid integer.
        """
        try:
            return int(name[prefix_len:])
        except (ValueError, TypeError):
            return None

    def add_block(self, block: MenuBlocks, m_pos: QPoint) -> DBlock:
        """
        Add a new block instance to the diagram.

        Args:
            block: MenuBlock template containing block type and parameters
            m_pos: QPoint position for the block's center

        Returns:
            The newly created and added block instance

        Raises:
            None - errors are logged but not raised
        """
        # Library blocks are not a block *class* -- they are a stored subsystem
        # that gets copied into the diagram (see lib/library.py).
        library_def = getattr(block, "library_def", None)
        if library_def is not None:
            return self.instantiate_library_block(library_def, m_pos)

        logger.debug(f"Adding new block of type {block.block_fn} at position {m_pos}")

        # Find next available ID for this block type. Names that do not match
        # the '<block_fn><int>' convention are skipped rather than raising.
        id_list = [
            parsed
            for b_elem in self.blocks_list
            if b_elem.block_fn == block.block_fn
            for parsed in (self._parse_id_suffix(b_elem.name, len(b_elem.block_fn)),)
            if parsed is not None
        ]
        sid = max(id_list) + 1 if id_list else 0

        try:
            # Calculate block position (centered on mouse)
            mouse_x = int(m_pos.x() - block.side_length[0] // 2)
            mouse_y = int(m_pos.y() - block.side_length[1] // 2)
            width = int(block.size[0])
            height = int(block.size[1])

            block_collision = QRect(mouse_x, mouse_y, width, height)
        except Exception as e:
            logger.error(f"Error creating QRect: {str(e)}")
            # Fallback with explicit integer conversion
            mouse_x = int(float(m_pos.x()) - float(block.side_length[0]) // 2)
            mouse_y = int(float(m_pos.y()) - float(block.side_length[1]) // 2)
            width = int(float(block.size[0]))
            height = int(float(block.size[1]))
            block_collision = QRect(mouse_x, mouse_y, width, height)

        # Create the block instance with category information
        category = getattr(block, "category", "Other")
        new_block = DBlock(
            block.block_fn,
            sid,
            block_collision,
            block.b_color,
            block.ins,
            block.outs,
            block.b_type,
            block.io_edit,
            block.fn_name,
            copy.deepcopy(block.params),
            block.external,
            block_class=block.block_class,
            colors=self.colors,
            category=category,
        )

        self.blocks_list.append(new_block)
        self.dirty = True
        logger.debug(f"New block created: {new_block.name} (category: {category})")
        return new_block

    # ------------------------------------------------------------------
    # User library blocks
    # ------------------------------------------------------------------

    def next_sid_for(self, block_fn: str) -> int:
        """Lowest unused sequential id for ``block_fn`` in the current scope."""
        id_list = [
            parsed
            for b_elem in self.blocks_list
            if b_elem.block_fn == block_fn
            for parsed in (self._parse_id_suffix(b_elem.name, len(b_elem.block_fn)),)
            if parsed is not None
        ]
        return max(id_list) + 1 if id_list else 0

    def reload_blocks(self, diagram_path: Optional[str] = None) -> int:
        """Re-scan user block modules *and* library files, rebuilding the palette.

        Returns the number of user block classes now registered. Built-ins are
        rebuilt too so a user block that stops loading disappears cleanly.
        """
        self.load_all_blocks(diagram_path, reload_user=True)
        user_count = sum(1 for mb in self.menu_blocks if getattr(mb, "user_block", False))
        self.load_library_blocks(diagram_path)
        return user_count

    def load_library_blocks(self, diagram_path: Optional[str] = None) -> int:
        """(Re)scan the user library folders and register the blocks found.

        Previously registered library entries are dropped first, so this
        doubles as the palette's "Refresh" action.  Returns the number of
        library blocks now registered.
        """
        from lib.library import discover_library_blocks

        # Mutate in place: DSim (and anything else) aliases this list, so
        # rebinding it here would leave stale palette entries behind.
        self.menu_blocks[:] = [
            mb for mb in self.menu_blocks if getattr(mb, "library_def", None) is None
        ]

        try:
            found = discover_library_blocks(diagram_path)
        except Exception:
            logger.exception("Library discovery failed; palette keeps the built-in blocks only")
            return 0

        for lib_block in found:
            try:
                self.menu_blocks.append(self._make_library_menu_block(lib_block))
            except Exception:
                logger.exception("Could not register library block %r", lib_block.block_id)
        logger.info("Registered %d user library block(s)", len(found))
        return len(found)

    def _make_library_menu_block(self, lib_block) -> MenuBlocks:
        """Wrap a discovered library block in a palette entry."""
        from lib.masks import mask_parameters

        block_data = lib_block.block_data
        width = int(block_data.get("coords_width", 100) or 100)
        height = int(block_data.get("coords_height", 80) or 80)
        specs = mask_parameters(lib_block.mask)

        menu_block = MenuBlocks(
            block_fn="Subsystem",
            fn_name=lib_block.name,
            io_params={
                "inputs": int(block_data.get("in_ports", 0) or 0),
                "outputs": int(block_data.get("out_ports", 0) or 0),
                "b_type": int(block_data.get("b_type", 2) or 2),
                "io_edit": "none",
            },
            ex_params={spec["name"]: spec.get("default") for spec in specs},
            b_color=self._get_category_color(lib_block.category),
            coords=(width, height),
            block_class=None,
            colors=self.colors,
        )
        menu_block.category = lib_block.category
        menu_block.param_meta = {spec["name"]: spec for spec in specs}
        menu_block.library_def = lib_block
        menu_block.doc = lib_block.description
        return menu_block

    def instantiate_library_block(self, lib_block, m_pos: QPoint) -> Optional[DBlock]:
        """Drop a *copy* of a library block onto the diagram.

        The instance owns its contents outright -- deleting or renaming the
        library file later cannot break the diagram -- and only remembers a
        ``library_ref`` so "Reload from library" can re-sync it.
        """
        from lib.library import attach_library_ref
        from lib.masks import apply_mask_appearance, refresh_saveable_params
        from lib.services.file_service import FileService

        data = lib_block.instance_block_data()
        sid = self.next_sid_for("Subsystem")
        width = int(data.get("coords_width", 100) or 100)
        height = int(data.get("coords_height", 80) or 80)

        data["sid"] = sid
        data["name"] = f"Subsystem{sid}"
        data["coords_left"] = int(m_pos.x() - width // 2)
        data["coords_top"] = int(m_pos.y() - height // 2)
        data["username"] = lib_block.name

        try:
            block = FileService(self)._construct_block(data)
        except Exception:
            logger.exception("Could not instantiate library block %r", lib_block.block_id)
            return None
        if block is None:
            return None

        block.name = data["name"]
        block.params["_name_"] = block.name
        block.username = lib_block.name
        apply_mask_appearance(block)
        attach_library_ref(block, lib_block.reference())
        refresh_saveable_params(block)

        self.blocks_list.append(block)
        self.dirty = True
        logger.info("Instantiated library block %r as %s", lib_block.block_id, block.name)
        return block

    def link_goto_from(self) -> None:
        """
        Automatically connect Goto/From blocks that share the same tag.
        For each From(tag) with no incoming line, connect the first matching Goto(tag).
        Also sync line labels to the configured signal name.
        """

        # Local neighbor helper to avoid dependency on DSim.get_neighbors
        def _get_neighbors(block_name):
            inputs = []
            outputs = []
            for line in self.line_list:
                if line.dstblock == block_name:
                    inputs.append(
                        {
                            "srcblock": line.srcblock,
                            "srcport": line.srcport,
                            "dstport": line.dstport,
                        }
                    )
                if line.srcblock == block_name:
                    outputs.append(
                        {
                            "dstblock": line.dstblock,
                            "srcport": line.srcport,
                            "dstport": line.dstport,
                        }
                    )
            return inputs, outputs

        # Collect goto blocks by tag
        goto_map = {}
        for b in self.blocks_list:
            if b.block_fn == "Goto":
                tag = str(b.params.get("tag", ""))
                # Ensure signal_name has fallback
                if not b.params.get("signal_name"):
                    b.params["signal_name"] = tag
                if tag not in goto_map:
                    goto_map[tag] = []
                goto_map[tag].append(b)

        # Remove any previous virtual lines (in place to keep shared reference)
        self.line_list[:] = [ln for ln in self.line_list if not getattr(ln, "hidden", False)]

        # For each From, ensure an incoming line from matching Goto
        for b in self.blocks_list:
            if b.block_fn != "From":
                continue
            tag = str(b.params.get("tag", ""))
            # Ensure signal_name fallback
            if not b.params.get("signal_name"):
                b.params["signal_name"] = tag

            # Update labels on outgoing visible lines from From
            for line in self.line_list:
                if line.srcblock == b.name:
                    line.label = b.params.get("signal_name") or tag

            if tag not in goto_map:
                continue

            # Skip if already connected
            inputs, _ = _get_neighbors(b.name)
            if inputs:
                continue

            src_block = goto_map[tag][0]  # deterministic choice of first Goto
            # Ensure coords are up to date
            src_block.update_Block()
            b.update_Block()

            # Find source of the goto (its incoming line)
            src_inputs, _ = _get_neighbors(src_block.name)
            if not src_inputs:
                continue
            src_line = src_inputs[0]

            # Create hidden virtual line from goto input source to From block
            from lib.simulation.connection import DLine

            src_point = src_block.in_coords[0] if src_block.in_coords else src_block.rect.center()
            dst_point = b.in_coords[0] if b.in_coords else b.rect.center()
            signal_name = b.params.get("signal_name") or b.params.get("tag", "")
            # Use max(sid)+1 (mirroring add_line) to avoid duplicate line
            # names/sids when lines have been deleted; len(line_list) can
            # collide with an existing sid.
            vline_sid = max([ln.sid for ln in self.line_list] + [-1]) + 1
            vline = DLine(
                sid=vline_sid,
                srcblock=src_line["srcblock"],
                srcport=src_line["srcport"],
                dstblock=b.name,
                dstport=0,
                points=[src_point, dst_point],
                hidden=True,
            )
            vline.label = signal_name
            self.line_list.append(vline)

        # Sync labels on incoming lines to Goto blocks as well
        for tag, gotos in goto_map.items():
            label = (gotos[0].params.get("signal_name") or tag) if gotos else tag
            for g in gotos:
                inputs, _ = _get_neighbors(g.name)
                for line in self.line_list:
                    if line.dstblock == g.name:
                        line.label = label

    def add_line(
        self, srcData: Optional[Tuple[str, int, QPoint]], dstData: Optional[Tuple[str, int, QPoint]]
    ) -> Optional[DLine]:
        """
        Add a connection line between two blocks.

        Args:
            srcData: Tuple (block_name, port_num, coordinates) for source port
            dstData: Tuple (block_name, port_num, coordinates) for destination port

        Returns:
            The newly created line, or None if srcData/dstData is invalid

        Raises:
            None - errors are logged but not raised
        """
        if srcData is None or dstData is None:
            logger.debug("Error: Invalid line data")
            return None

        # Find next available ID. Names not matching 'Line<int>' are skipped
        # rather than raising (len('Line') == 4).
        id_list = [
            parsed
            for line in self.line_list
            for parsed in (self._parse_id_suffix(line.name, 4),)
            if parsed is not None
        ]
        sid = max(id_list) + 1 if id_list else 0

        try:
            line = DLine(
                sid,
                srcblock=srcData[0],
                srcport=srcData[1],
                dstblock=dstData[0],
                dstport=dstData[1],
                points=(srcData[2], dstData[2]),
            )
            line.color = QColor(255, 0, 0)  # Red lines

            self.line_list.append(line)
            self.dirty = True
            logger.debug(f"Line created: {line.name}")
            return line
        except Exception as e:
            logger.error(f"Error creating line: {e}")
            return None

    def remove_block(self, block: DBlock) -> None:
        """
        Remove a block and all associated connections.

        Args:
            block: DBlock instance to remove from the diagram
        """
        self.blocks_list.remove(block)
        # Remove all lines connected to this block
        self.line_list = [
            line
            for line in self.line_list
            if not self._is_line_connected_to_block(line, block.name)
        ]
        self.dirty = True
        logger.debug(f"Removed block {block.name} and associated lines")

    def remove_line(self, line: DLine) -> None:
        """
        Remove a connection line from the diagram.

        Args:
            line: DLine instance to remove
        """
        if line in self.line_list:
            self.line_list.remove(line)
            self.dirty = True
            logger.debug(f"Removed line {line.name}")

    def clear_all(self) -> None:
        """Clear all blocks and lines from the diagram, reset dirty flag."""
        self.blocks_list.clear()
        self.line_list.clear()
        self.dirty = False
        logger.debug("Cleared all blocks and lines")

    def get_block_by_name(self, name: str) -> Optional[DBlock]:
        """
        Find a block by its name.

        Args:
            name: Block name to search for

        Returns:
            DBlock if found, None otherwise
        """
        for block in self.blocks_list:
            if block.name == name:
                return block
        return None

    def is_port_available(self, dst_line: Tuple[str, int, QPoint]) -> bool:
        """
        Check if an input port is already connected.

        Args:
            dst_line: Tuple (block_name, port_num, coordinates) for destination

        Returns:
            True if port is available (not connected), False if already connected
        """
        for line in self.line_list:
            if line.dstblock == dst_line[0] and line.dstport == dst_line[1]:
                return False
        return True

    def update_lines(self) -> None:
        """Update all line positions based on current block positions."""
        for line in self.line_list:
            line.update_line(self.blocks_list)

    def _is_line_connected_to_block(self, line: DLine, block_name: str) -> bool:
        """
        Check if a line is connected to a specific block.

        Args:
            line: DLine to check
            block_name: Name of block to check connection to

        Returns:
            True if line is connected to the block, False otherwise
        """
        return line.dstblock == block_name or line.srcblock == block_name

    def get_diagram_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current diagram.

        Returns:
            Dictionary with keys: 'blocks', 'lines', 'modified', 'block_types'
        """
        return {
            "blocks": len(self.blocks_list),
            "lines": len(self.line_list),
            "modified": self.dirty,
            "block_types": len(set(b.block_fn for b in self.blocks_list)),
        }
