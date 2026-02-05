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
        self.navigation_stack.append((
            self.model.blocks_list,
            self.model.line_list,
            self.current_subsystem
        ))

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

        # Reset view state variables
        self.dsim.ss_count = 0
        self.dsim.dirty = False

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
            inports = [b for b in internal_blocks if getattr(b, 'block_type', '') == 'Inport']
            outports = [b for b in internal_blocks if getattr(b, 'block_type', '') == 'Outport']

            # Sort by Y position to allow user to reorder ports by moving blocks
            inports.sort(key=lambda b: b.top)
            outports.sort(key=lambda b: b.top)

            # Update parent block ports definition
            if not hasattr(parent_block, 'ports'):
                parent_block.ports = {}

            # Input Ports
            parent_block.ports['in'] = []
            for idx, inp in enumerate(inports):
                rel_y = (parent_block.height / (len(inports) + 1)) * (idx + 1)
                parent_block.ports['in'].append({
                    'pos': (0, rel_y),
                    'type': 'input',
                    'name': str(idx+1)
                })

            # Output Ports
            parent_block.ports['out'] = []
            for idx, outp in enumerate(outports):
                rel_y = (parent_block.height / (len(outports) + 1)) * (idx + 1)
                parent_block.ports['out'].append({
                    'pos': (parent_block.width, rel_y),
                    'type': 'output',
                    'name': str(idx+1)
                })

            # Update geometry (port coordinates)
            if hasattr(parent_block, 'update_Block'):
                parent_block.update_Block()

            # Update parameters to reflect new port counts
            if parent_block.in_ports != parent_block.params.get('_inputs_'):
                parent_block.params['_inputs_'] = parent_block.in_ports
                logger.debug(f"Synced subsystem '_inputs_' param to {parent_block.in_ports}")

            if parent_block.out_ports != parent_block.params.get('_outputs_'):
                parent_block.params['_outputs_'] = parent_block.out_ports
                logger.debug(f"Synced subsystem '_outputs_' param to {parent_block.out_ports}")

            # Mark dirty to ensure save
            self.dsim.dirty = True
        else:
            logger.error(f"Could not find parent block {current_subsystem_name} after exiting scope")

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
        path = ['Top Level']
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

    def create_subsystem_from_selection(self, selected_blocks):
        """
        Create a subsystem containing the selected blocks.

        Args:
            selected_blocks: List of blocks to include in subsystem

        Returns:
            Subsystem block instance
        """
        if not selected_blocks:
            return

        # Calculate bounding box of selected blocks
        min_x = min(b.rect.left() for b in selected_blocks)
        min_y = min(b.rect.top() for b in selected_blocks)
        max_x = max(b.rect.right() for b in selected_blocks)
        max_y = max(b.rect.bottom() for b in selected_blocks)
        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2

        # Create Subsystem block
        subsys = Subsystem()
        subsys.sid = max([b.sid for b in self.dsim.blocks_list] + [0]) + 1
        subsys.name = f"Subsystem{subsys.sid}"
        subsys.ports = {}  # Initialize ports dict for boundary connections

        # Position at center of selected blocks using relocate_Block to update all coords
        target_pos = QPoint(center_x - subsys.width // 2, center_y - subsys.height // 2)
        subsys.relocate_Block(target_pos)

        # Add to current scope
        self.dsim.blocks_list.append(subsys)

        # Logic to move blocks and handle connections
        internal_lines = []
        boundary_lines = []

        selected_names = {b.name for b in selected_blocks}
        current_lines = list(self.model.line_list)

        logger.debug(f"Subsystem creation: {len(selected_blocks)} blocks selected, {len(current_lines)} lines")

        for line in current_lines:
            src_in = line.srcblock in selected_names
            dst_in = line.dstblock in selected_names

            if src_in and dst_in:
                internal_lines.append(line)
            elif src_in and not dst_in:
                boundary_lines.append((line, 'out'))
            elif not src_in and dst_in:
                boundary_lines.append((line, 'in'))

        # Detect unconnected ports on selected blocks
        # These need Inport/Outport blocks created for them
        unconnected_inputs = []   # List of (block, port_idx)
        unconnected_outputs = []  # List of (block, port_idx)

        for block in selected_blocks:
            # Check each input port
            for port_idx in range(block.in_ports):
                # Is there ANY line connecting to this input?
                has_connection = any(
                    line.dstblock == block.name and line.dstport == port_idx
                    for line in current_lines
                )
                if not has_connection:
                    unconnected_inputs.append((block, port_idx))

            # Check each output port
            for port_idx in range(block.out_ports):
                # Is there ANY line connecting from this output?
                has_connection = any(
                    line.srcblock == block.name and line.srcport == port_idx
                    for line in current_lines
                )
                if not has_connection:
                    unconnected_outputs.append((block, port_idx))

        logger.debug(f"Unconnected ports: {len(unconnected_inputs)} inputs, {len(unconnected_outputs)} outputs")

        # Move blocks and internal lines to subsystem
        for b in selected_blocks:
            if b in self.dsim.blocks_list:
                self.dsim.blocks_list.remove(b)
            new_pos = QPoint(b.rect.left() - min_x + 100, b.rect.top() - min_y + 100)
            b.relocate_Block(new_pos)
            subsys.sub_blocks.append(b)

        for l in internal_lines:
            if l in self.dsim.line_list:
                self.dsim.line_list.remove(l)
            if l in self.dsim.connections_list:
                if l in self.dsim.connections_list:
                    self.dsim.connections_list.remove(l)
            subsys.sub_lines.append(l)

        # Recalculate internal lines visually
        dx = -min_x + 100
        dy = -min_y + 100
        block_map = {b.name: b for b in subsys.sub_blocks}

        for line in internal_lines:
            if hasattr(line, 'points'):
                new_points = []
                for p in line.points:
                    new_points.append(QPoint(p.x() + dx, p.y() + dy))
                line.points = new_points

                src = block_map.get(line.srcblock)
                dst = block_map.get(line.dstblock)

                start_p = line.points[0]
                end_p = line.points[-1]

                if src and line.srcport < len(src.out_coords):
                    start_p = src.out_coords[line.srcport]
                if dst and line.dstport < len(dst.in_coords):
                    end_p = dst.in_coords[line.dstport]

                line.points[0] = start_p
                line.points[-1] = end_p

                try:
                    line.path, line.points, line.segments = line.create_trajectory(
                        start_p, end_p, subsys.sub_blocks, points=line.points
                    )
                except Exception as e:
                    logger.error(f"Failed to update line trajectory in subsystem: {e}")

        # Handle Ports
        inport_idx = 1
        outport_idx = 1

        for line, direction in boundary_lines:
            if direction == 'in':
                # External Source -> Subsystem (Inport) -> Internal Dest
                inport = Inport(block_name=f"In{inport_idx}")
                inport.sid = max([b.sid for b in subsys.sub_blocks] + [0]) + 1
                # Update name to match inport_idx (flattener looks for inport1, inport2, etc.)
                inport.name = f"inport{inport_idx}"

                target_block = block_map.get(line.dstblock)
                if target_block and line.dstport < len(target_block.in_coords):
                    target_y = target_block.in_coords[line.dstport].y()
                    inport.rect = QRect(20, target_y - inport.height//2,
                                       inport.width, inport.height)
                else:
                    inport.rect = QRect(20, 50 * inport_idx, inport.width, inport.height)

                inport.relocate_Block(inport.rect.topLeft())
                subsys.sub_blocks.append(inport)
                block_map[inport.name] = inport

                internal_line = DLine(
                    sid=max([l.sid for l in subsys.sub_lines] + [0]) + 1,
                    srcblock=inport.name, srcport=0,
                    dstblock=line.dstblock, dstport=line.dstport,
                    points=(inport.out_coords[0], line.points[-1])
                )

                try:
                    target_p = None
                    if target_block:
                        target_p = target_block.in_coords[line.dstport]

                    internal_line.path, internal_line.points, internal_line.segments = \
                        internal_line.create_trajectory(
                            inport.out_coords[0],
                            target_p if target_p else line.points[-1],
                            subsys.sub_blocks
                        )
                except Exception as e:
                    logger.warning(f"Traj calc failed for new internal line: {e}")

                subsys.sub_lines.append(internal_line)

                if 'in' not in subsys.ports:
                    subsys.ports['in'] = []

                port_pos = (0, (subsys.height / (len(boundary_lines) + 1)) * inport_idx)
                subsys.ports['in'].append({
                    'pos': port_pos,
                    'type': 'input',
                    'name': str(inport_idx)
                })

                port_idx = len(subsys.ports['in']) - 1
                line.dstblock = subsys.name
                line.dstport = port_idx

                inport_idx += 1

            elif direction == 'out':
                # Internal Source -> Subsystem (Outport) -> External Dest
                outport = Outport(block_name=f"Out{outport_idx}")
                outport.sid = max([b.sid for b in subsys.sub_blocks] + [0]) + 1
                # Update name to match outport_idx (flattener looks for outport1, outport2, etc.)
                outport.name = f"outport{outport_idx}"

                max_internal_x = max(b.rect.right() for b in subsys.sub_blocks)
                source_block = block_map.get(line.srcblock)

                if source_block and line.srcport < len(source_block.out_coords):
                    source_y = source_block.out_coords[line.srcport].y()
                    outport.rect = QRect(max_internal_x + 50, source_y - outport.height//2,
                                        outport.width, outport.height)
                else:
                    outport.rect = QRect(max_internal_x + 50, 50 * outport_idx,
                                        outport.width, outport.height)

                outport.relocate_Block(outport.rect.topLeft())
                subsys.sub_blocks.append(outport)
                block_map[outport.name] = outport

                internal_line = DLine(
                    sid=max([l.sid for l in subsys.sub_lines] + [0]) + 1,
                    srcblock=line.srcblock, srcport=line.srcport,
                    dstblock=outport.name, dstport=0,
                    points=(line.points[0], outport.in_coords[0])
                )

                try:
                    src_p = None
                    if source_block:
                        src_p = source_block.out_coords[line.srcport]

                    internal_line.path, internal_line.points, internal_line.segments = \
                        internal_line.create_trajectory(
                            src_p if src_p else line.points[0],
                            outport.in_coords[0],
                            subsys.sub_blocks
                        )
                except Exception:
                    pass

                subsys.sub_lines.append(internal_line)

                if 'out' not in subsys.ports:
                    subsys.ports['out'] = []
                port_pos = (subsys.width, (subsys.height / (len(boundary_lines) + 1)) * outport_idx)
                subsys.ports['out'].append({
                    'pos': port_pos,
                    'type': 'output',
                    'name': str(outport_idx)
                })

                port_idx = len(subsys.ports['out']) - 1
                line.srcblock = subsys.name
                line.srcport = port_idx

                outport_idx += 1

        # Handle unconnected input ports - create Inport blocks for them
        for block, port_idx in unconnected_inputs:
            inport = Inport(block_name=f"In{inport_idx}")
            inport.sid = max([b.sid for b in subsys.sub_blocks] + [0]) + 1
            # Update name to match inport_idx (flattener looks for inport1, inport2, etc.)
            inport.name = f"inport{inport_idx}"

            # Position Inport to the left of the target block's input port
            target_block = block_map.get(block.name)
            if target_block and port_idx < len(target_block.in_coords):
                target_y = target_block.in_coords[port_idx].y()
                inport.rect = QRect(20, target_y - inport.height // 2,
                                   inport.width, inport.height)
            else:
                inport.rect = QRect(20, 50 * inport_idx, inport.width, inport.height)

            inport.relocate_Block(inport.rect.topLeft())
            subsys.sub_blocks.append(inport)
            block_map[inport.name] = inport

            # Create internal line from Inport to the block's input port
            target_p = None
            if target_block and port_idx < len(target_block.in_coords):
                target_p = target_block.in_coords[port_idx]

            internal_line = DLine(
                sid=max([l.sid for l in subsys.sub_lines] + [0]) + 1,
                srcblock=inport.name, srcport=0,
                dstblock=block.name, dstport=port_idx,
                points=(inport.out_coords[0], target_p if target_p else inport.out_coords[0])
            )

            try:
                internal_line.path, internal_line.points, internal_line.segments = \
                    internal_line.create_trajectory(
                        inport.out_coords[0],
                        target_p if target_p else inport.out_coords[0],
                        subsys.sub_blocks
                    )
            except Exception as e:
                logger.warning(f"Traj calc failed for unconnected input line: {e}")

            subsys.sub_lines.append(internal_line)

            # Add external input port to subsystem
            if 'in' not in subsys.ports:
                subsys.ports['in'] = []

            total_inputs = len(boundary_lines) + len(unconnected_inputs)
            port_pos = (0, (subsys.height / (total_inputs + 1)) * inport_idx)
            subsys.ports['in'].append({
                'pos': port_pos,
                'type': 'input',
                'name': str(inport_idx)
            })

            inport_idx += 1

        # Handle unconnected output ports - create Outport blocks for them
        for block, port_idx in unconnected_outputs:
            outport = Outport(block_name=f"Out{outport_idx}")
            outport.sid = max([b.sid for b in subsys.sub_blocks] + [0]) + 1
            # Update name to match outport_idx (flattener looks for outport1, outport2, etc.)
            outport.name = f"outport{outport_idx}"

            # Position Outport to the right of all internal blocks
            max_internal_x = max(b.rect.right() for b in subsys.sub_blocks)
            source_block = block_map.get(block.name)

            if source_block and port_idx < len(source_block.out_coords):
                source_y = source_block.out_coords[port_idx].y()
                outport.rect = QRect(max_internal_x + 50, source_y - outport.height // 2,
                                    outport.width, outport.height)
            else:
                outport.rect = QRect(max_internal_x + 50, 50 * outport_idx,
                                    outport.width, outport.height)

            outport.relocate_Block(outport.rect.topLeft())
            subsys.sub_blocks.append(outport)
            block_map[outport.name] = outport

            # Create internal line from the block's output port to Outport
            src_p = None
            if source_block and port_idx < len(source_block.out_coords):
                src_p = source_block.out_coords[port_idx]

            internal_line = DLine(
                sid=max([l.sid for l in subsys.sub_lines] + [0]) + 1,
                srcblock=block.name, srcport=port_idx,
                dstblock=outport.name, dstport=0,
                points=(src_p if src_p else outport.in_coords[0], outport.in_coords[0])
            )

            try:
                internal_line.path, internal_line.points, internal_line.segments = \
                    internal_line.create_trajectory(
                        src_p if src_p else outport.in_coords[0],
                        outport.in_coords[0],
                        subsys.sub_blocks
                    )
            except Exception:
                pass

            subsys.sub_lines.append(internal_line)

            # Add external output port to subsystem
            if 'out' not in subsys.ports:
                subsys.ports['out'] = []

            total_outputs = len(boundary_lines) + len(unconnected_outputs)
            port_pos = (subsys.width, (subsys.height / (total_outputs + 1)) * outport_idx)
            subsys.ports['out'].append({
                'pos': port_pos,
                'type': 'output',
                'name': str(outport_idx)
            })

            outport_idx += 1

        self.dsim.dirty = True
        subsys.update_Block()
        logger.info(f"Subsystem {subsys.name} created with {len(subsys.sub_blocks)} blocks.")
        return subsys
