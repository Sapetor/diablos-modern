"""
SubsystemManager - Handles subsystem navigation and creation.

Extracted from lib.py to reduce file size and improve modularity.
"""

import logging
from PyQt5.QtCore import QPoint, QRect
from blocks.subsystem import Subsystem
from blocks.inport import Inport
from blocks.outport import Outport
from lib.simulation.connection import DLine

logger = logging.getLogger(__name__)


class SubsystemManager:
    """
    Manages subsystem navigation, creation, and port synchronization.

    Handles:
    - Entering/exiting subsystem scopes
    - Creating subsystems from selected blocks
    - Synchronizing internal ports with external subsystem block
    - Navigation stack management
    """

    def __init__(self, model, dsim):
        """
        Initialize the SubsystemManager.

        Args:
            model: SimulationModel instance (for blocks_list, line_list)
            dsim: DSim instance (for accessing other components)
        """
        self.model = model
        self.dsim = dsim

        # Navigation Context
        # Stack of (blocks_list, line_list, parent_subsystem_name) tuples
        self.navigation_stack = []
        self.current_subsystem = None  # Top level is None

    def enter_subsystem(self, subsystem_block):
        """
        Enter a subsystem block to edit its contents.
        Pushes the current context to the stack and enters the subsystem.
        """
        # Save current context (references to the lists)
        self.navigation_stack.append(
            (self.model.blocks_list, self.model.line_list, self.current_subsystem)
        )

        # Switch to subsystem context
        # Use the subsystem's internal lists as the active model lists.
        self.model.blocks_list = subsystem_block.sub_blocks
        self.model.line_list = subsystem_block.sub_lines

        # Sync DSim references to match model's new active lists
        self.dsim.blocks_list = self.model.blocks_list
        self.dsim.line_list = self.model.line_list
        self.dsim.connections_list = self.dsim.line_list

        self.current_subsystem = subsystem_block.name

        # Reset selection state when changing scope
        for block in self.dsim.blocks_list:
            block.selected = False
        for line in self.dsim.line_list:
            line.selected = False

        # Reset view state variables.
        # NOTE: do NOT clear ``dsim.dirty`` here.  Navigating into a subsystem
        # is a view change, not a save: ``DSim.dirty`` is a live view of
        # ``model.dirty``, so clearing it would silently discard the
        # unsaved-changes flag for edits the user made before drilling in
        # (the close prompt and the status bar both read it).
        self.dsim.ss_count = 0

        logger.info(f"Entered subsystem: {subsystem_block.name}")

    def exit_subsystem(self):
        """
        Exit the current subsystem and return to the parent scope.
        Syncs external ports with internal Inport/Outport blocks.
        """
        if not self.navigation_stack:
            logger.warning("Already at top level.")
            return

        # 1. Capture current internal state before popping
        internal_blocks = self.dsim.blocks_list
        current_subsystem_name = self.current_subsystem

        # 2. Restore previous context
        (prev_blocks, prev_lines, prev_subsystem) = self.navigation_stack.pop()

        # 3. Restore model lists
        self.model.blocks_list = prev_blocks
        self.model.line_list = prev_lines

        # 4. Sync DSim references
        self.dsim.blocks_list = self.model.blocks_list
        self.dsim.line_list = self.model.line_list
        self.dsim.connections_list = self.dsim.line_list

        self.current_subsystem = prev_subsystem

        # 5. Find the parent Subsystem block in the restored scope
        parent_block = None
        for block in self.dsim.blocks_list:
            if block.name == current_subsystem_name:
                parent_block = block
                break

        if parent_block:
            logger.info(f"Syncing ports for subsystem {current_subsystem_name}")

            # Find all internal ports
            inports = [b for b in internal_blocks if getattr(b, "block_fn", "") == "Inport"]
            outports = [b for b in internal_blocks if getattr(b, "block_fn", "") == "Outport"]

            # Sort by Y position to allow user to reorder ports by moving blocks
            inports.sort(key=lambda b: b.top)
            outports.sort(key=lambda b: b.top)

            # Update parent block ports definition
            if not hasattr(parent_block, "ports"):
                parent_block.ports = {}

            # Input Ports
            parent_block.ports["in"] = []
            for idx, inp in enumerate(inports):
                rel_y = (parent_block.height / (len(inports) + 1)) * (idx + 1)
                parent_block.ports["in"].append(
                    {"pos": (0, rel_y), "type": "input", "name": str(idx + 1)}
                )

            # Output Ports
            parent_block.ports["out"] = []
            for idx, outp in enumerate(outports):
                rel_y = (parent_block.height / (len(outports) + 1)) * (idx + 1)
                parent_block.ports["out"].append(
                    {"pos": (parent_block.width, rel_y), "type": "output", "name": str(idx + 1)}
                )

            # Update geometry (port coordinates)
            if hasattr(parent_block, "update_Block"):
                parent_block.update_Block()

            # Update parameters to reflect new port counts
            if parent_block.in_ports != parent_block.params.get("_inputs_"):
                parent_block.params["_inputs_"] = parent_block.in_ports
                logger.debug(f"Synced subsystem '_inputs_' param to {parent_block.in_ports}")

            if parent_block.out_ports != parent_block.params.get("_outputs_"):
                parent_block.params["_outputs_"] = parent_block.out_ports
                logger.debug(f"Synced subsystem '_outputs_' param to {parent_block.out_ports}")

            # Mark dirty to ensure save
            self.dsim.dirty = True
        else:
            logger.error(
                f"Could not find parent block {current_subsystem_name} after exiting scope"
            )

        # Reset selection
        for block in self.dsim.blocks_list:
            block.selected = False
        for line in self.dsim.line_list:
            line.selected = False

        logger.info("Exited subsystem, returned to parent scope")

    def get_current_path(self):
        """
        Return the current navigation path as a list of strings.
        Example: ['Top Level', 'Subsystem1', 'Nested2']
        """
        path = ["Top Level"]
        for _, _, name in self.navigation_stack:
            if name:
                path.append(name)

        if self.current_subsystem:
            path.append(self.current_subsystem)

        return path

    def get_root_context(self):
        """
        Get the root context (blocks_list, line_list) of the simulation model.
        Used for execution to ensure we always simulate the full system.
        """
        if not self.navigation_stack:
            return self.dsim.blocks_list, self.dsim.line_list
        else:
            # The bottom of the stack (index 0) contains the references to root lists
            # Stack format: (prev_blocks, prev_lines, prev_subsystem)
            return self.navigation_stack[0][0], self.navigation_stack[0][1]

    # ------------------------------------------------------------------
    # Subsystem creation
    # ------------------------------------------------------------------

    #: Selected blocks keep their relative layout, shifted so the group's
    #: top-left corner lands here inside the subsystem.
    _INTERNAL_ORIGIN = 100
    #: Inports sit in a column at this x; Outports this far right of the
    #: rightmost internal block.
    _INPORT_X = 20
    _OUTPORT_GAP = 50

    def create_subsystem_from_selection(self, selected_blocks):
        """
        Create a subsystem containing the selected blocks.

        The selection moves into a new Subsystem block placed at the group's
        centre. Lines between two selected blocks move inside with it; a line
        crossing the boundary is cut at a new Inport/Outport and re-attached to
        the subsystem's external port; every unconnected input/output port of a
        selected block also gets an Inport/Outport so the subsystem exposes it.

        Args:
            selected_blocks: List of blocks to include in subsystem

        Returns:
            Subsystem block instance, or None if selected_blocks is empty.
        """
        if not selected_blocks:
            return None

        min_x, min_y, max_x, max_y = _bounding_box(selected_blocks)
        subsys = self._new_subsystem_block((min_x + max_x) // 2, (min_y + max_y) // 2)
        self.dsim.blocks_list.append(subsys)

        current_lines = list(self.model.line_list)
        selected_names = {b.name for b in selected_blocks}
        internal_lines, boundary_lines = _classify_lines(current_lines, selected_names)
        unconnected_inputs, unconnected_outputs = _unconnected_ports(selected_blocks, current_lines)
        logger.debug(
            f"Subsystem creation: {len(selected_blocks)} blocks selected, {len(current_lines)} lines"
        )
        logger.debug(
            f"Unconnected ports: {len(unconnected_inputs)} inputs, {len(unconnected_outputs)} outputs"
        )

        offset = QPoint(-min_x + self._INTERNAL_ORIGIN, -min_y + self._INTERNAL_ORIGIN)
        self._move_into_subsystem(subsys, selected_blocks, internal_lines, offset)
        block_map = {b.name: b for b in subsys.sub_blocks}
        _reroute_internal_lines(internal_lines, block_map, offset, subsys.sub_blocks)

        # Port Y-positions are spaced evenly per direction over the total
        # number of ports that direction will end up with.
        n_inputs = sum(1 for _, d in boundary_lines if d == "in") + len(unconnected_inputs)
        n_outputs = sum(1 for _, d in boundary_lines if d == "out") + len(unconnected_outputs)
        inport_idx = 1
        outport_idx = 1

        for line, direction in boundary_lines:
            if direction == "in":
                # External source -> [Inport -> internal destination]
                port = self._add_inport(
                    subsys,
                    block_map,
                    inport_idx,
                    block_map.get(line.dstblock),
                    line.dstblock,
                    line.dstport,
                    line.points[-1],
                    n_inputs,
                )
                line.dstblock, line.dstport = subsys.name, port
                inport_idx += 1
            else:
                # [Internal source -> Outport] -> external destination
                port = self._add_outport(
                    subsys,
                    block_map,
                    outport_idx,
                    block_map.get(line.srcblock),
                    line.srcblock,
                    line.srcport,
                    line.points[0],
                    n_outputs,
                )
                line.srcblock, line.srcport = subsys.name, port
                outport_idx += 1

        for block, port_idx in unconnected_inputs:
            self._add_inport(
                subsys, block_map, inport_idx, block, block.name, port_idx, None, n_inputs
            )
            inport_idx += 1

        for block, port_idx in unconnected_outputs:
            self._add_outport(
                subsys, block_map, outport_idx, block, block.name, port_idx, None, n_outputs
            )
            outport_idx += 1

        self.dsim.dirty = True
        subsys.update_Block()
        logger.info(f"Subsystem {subsys.name} created with {len(subsys.sub_blocks)} blocks.")
        return subsys

    def _new_subsystem_block(self, center_x, center_y):
        """A fresh Subsystem block in the current scope, centred on (center_x, center_y)."""
        subsys = Subsystem()
        subsys.sid = _next_sid(self.dsim.blocks_list)
        subsys.name = f"Subsystem{subsys.sid}"
        subsys.ports = {}  # external boundary ports, filled by _add_inport / _add_outport
        subsys.relocate_Block(QPoint(center_x - subsys.width // 2, center_y - subsys.height // 2))
        return subsys

    def _move_into_subsystem(self, subsys, selected_blocks, internal_lines, offset):
        """Move the selection and its internal lines out of the current scope into ``subsys``."""
        for b in selected_blocks:
            if b in self.dsim.blocks_list:
                self.dsim.blocks_list.remove(b)
            b.relocate_Block(b.rect.topLeft() + offset)
            subsys.sub_blocks.append(b)

        for line in internal_lines:
            if line in self.dsim.line_list:
                self.dsim.line_list.remove(line)
            # connections_list aliases line_list (see enter/exit_subsystem), so the
            # line is already removed above; guard the non-aliased case defensively.
            if line in self.dsim.connections_list:
                self.dsim.connections_list.remove(line)
            subsys.sub_lines.append(line)

    def _add_inport(
        self, subsys, block_map, index, target_block, target_name, target_port, fallback_end, total
    ):
        """Add ``inport<index>`` feeding ``target_name[target_port]`` and expose it as an
        external input port of ``subsys``. Returns the external port index."""
        inport = Inport(block_name=f"In{index}")
        inport.sid = _next_sid(subsys.sub_blocks)
        # The flattener looks for inport1, inport2, ...
        inport.name = f"inport{index}"

        target_p = _port_point(target_block, "in_coords", target_port)
        y = target_p.y() - inport.height // 2 if target_p is not None else 50 * index
        inport.rect = QRect(self._INPORT_X, y, inport.width, inport.height)
        inport.relocate_Block(inport.rect.topLeft())
        subsys.sub_blocks.append(inport)
        block_map[inport.name] = inport

        start = inport.out_coords[0]
        end = target_p if target_p is not None else fallback_end
        _add_internal_line(subsys, inport.name, 0, target_name, target_port, start, end or start)

        ports = subsys.ports.setdefault("in", [])
        ports.append(
            {
                "pos": (0, (subsys.height / (total + 1)) * index),
                "type": "input",
                "name": str(index),
            }
        )
        return len(ports) - 1

    def _add_outport(
        self,
        subsys,
        block_map,
        index,
        source_block,
        source_name,
        source_port,
        fallback_start,
        total,
    ):
        """Add ``outport<index>`` fed by ``source_name[source_port]`` and expose it as an
        external output port of ``subsys``. Returns the external port index."""
        outport = Outport(block_name=f"Out{index}")
        outport.sid = _next_sid(subsys.sub_blocks)
        # The flattener looks for outport1, outport2, ...
        outport.name = f"outport{index}"

        max_internal_x = max(b.rect.right() for b in subsys.sub_blocks)
        source_p = _port_point(source_block, "out_coords", source_port)
        y = source_p.y() - outport.height // 2 if source_p is not None else 50 * index
        outport.rect = QRect(max_internal_x + self._OUTPORT_GAP, y, outport.width, outport.height)
        outport.relocate_Block(outport.rect.topLeft())
        subsys.sub_blocks.append(outport)
        block_map[outport.name] = outport

        end = outport.in_coords[0]
        start = source_p if source_p is not None else fallback_start
        _add_internal_line(subsys, source_name, source_port, outport.name, 0, start or end, end)

        ports = subsys.ports.setdefault("out", [])
        ports.append(
            {
                "pos": (subsys.width, (subsys.height / (total + 1)) * index),
                "type": "output",
                "name": str(index),
            }
        )
        return len(ports) - 1


# ----------------------------------------------------------------------
# Subsystem-creation helpers (pure functions over blocks and lines)
# ----------------------------------------------------------------------


def _next_sid(items):
    return max([item.sid for item in items] + [0]) + 1


def _bounding_box(blocks):
    """``(min_x, min_y, max_x, max_y)`` over the blocks' rects."""
    return (
        min(b.rect.left() for b in blocks),
        min(b.rect.top() for b in blocks),
        max(b.rect.right() for b in blocks),
        max(b.rect.bottom() for b in blocks),
    )


def _classify_lines(lines, selected_names):
    """Split ``lines`` into those inside the selection and those crossing it.

    Returns ``(internal_lines, boundary_lines)`` where each boundary entry is
    ``(line, "in")`` for an external source feeding a selected block, or
    ``(line, "out")`` for a selected block feeding an external destination.
    Lines touching no selected block are dropped.
    """
    internal, boundary = [], []
    for line in lines:
        src_in = line.srcblock in selected_names
        dst_in = line.dstblock in selected_names
        if src_in and dst_in:
            internal.append(line)
        elif src_in:
            boundary.append((line, "out"))
        elif dst_in:
            boundary.append((line, "in"))
    return internal, boundary


def _unconnected_ports(blocks, lines):
    """``(inputs, outputs)`` as lists of ``(block, port_idx)`` with no line attached."""
    inputs, outputs = [], []
    for block in blocks:
        for port_idx in range(block.in_ports):
            if not any(l.dstblock == block.name and l.dstport == port_idx for l in lines):
                inputs.append((block, port_idx))
        for port_idx in range(block.out_ports):
            if not any(l.srcblock == block.name and l.srcport == port_idx for l in lines):
                outputs.append((block, port_idx))
    return inputs, outputs


def _port_point(block, coords_attr, port_idx):
    """The QPoint of ``block``'s port, or None when the block or port is missing."""
    if block is None:
        return None
    coords = getattr(block, coords_attr, ())
    return coords[port_idx] if port_idx < len(coords) else None


def _reroute_internal_lines(lines, block_map, offset, sub_blocks):
    """Shift moved lines by ``offset``, snap their ends to the (moved) ports and re-route."""
    for line in lines:
        if not hasattr(line, "points"):
            continue
        line.points = [p + offset for p in line.points]

        start_p = _port_point(block_map.get(line.srcblock), "out_coords", line.srcport)
        end_p = _port_point(block_map.get(line.dstblock), "in_coords", line.dstport)
        line.points[0] = start_p if start_p is not None else line.points[0]
        line.points[-1] = end_p if end_p is not None else line.points[-1]

        try:
            line.path, line.points, line.segments = line.create_trajectory(
                line.points[0], line.points[-1], sub_blocks, points=line.points
            )
        except Exception as e:
            logger.error(f"Failed to update line trajectory in subsystem: {e}")


def _add_internal_line(subsys, src, srcport, dst, dstport, start, end):
    """Append a new routed line ``src[srcport] -> dst[dstport]`` to ``subsys.sub_lines``."""
    line = DLine(
        sid=_next_sid(subsys.sub_lines),
        srcblock=src,
        srcport=srcport,
        dstblock=dst,
        dstport=dstport,
        points=(start, end),
    )
    try:
        line.path, line.points, line.segments = line.create_trajectory(
            start, end, subsys.sub_blocks
        )
    except Exception as e:
        logger.warning(f"Trajectory calculation failed for internal line {src} -> {dst}: {e}")
    subsys.sub_lines.append(line)
    return line
