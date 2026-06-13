
import logging
import copy
import re
from lib.simulation.connection import DLine

logger = logging.getLogger(__name__)

class Flattener:
    """
    Helper class to flatten a hierarchical block diagram into a list of primitive blocks and connections.
    """

    def flatten(self, top_blocks, top_lines):
        """
        Flatten the hierarchy.
        
        Args:
            top_blocks: List of DBlock objects at top level.
            top_lines: List of DLine objects at top level.
            
        Returns:
            (flat_blocks, flat_lines): Lists of primitive blocks and resolved connections.
            All blocks are renamed to fully qualified paths (e.g. "Subsystem1/Gain1").
        """
        self.flat_blocks = []
        self.flat_lines = []
        # keys: (full_block_name, port_idx). value: (full_src_name, src_port_idx)
        self.input_drivers = {} 
        self.block_map = {} # full_name -> block (clones for primitives)
        
        # 1. Recursively collect primitives and build input map
        self._collect_recursive(top_blocks, top_lines, "")
        
        # 2. Filter primitives (exclude container blocks)
        # Note: Inport/Outport are primitives in the sense they are leaves, but we treat them as wires.
        primitives = []
        for b in self.flat_blocks:
             # Force parameter resolution if needed? No, execution copies should keep params.
             # Access params directly or use property?
             # block_type is usually a property of DBlock based on params['block_type']? 
             # No, block_type is usually set in params or derived.
             # Let's check params first.
             b_type = b.params.get('block_type', '') if b.params else ''
             if not b_type:
                 b_type = getattr(b, 'block_type', '')
                 
             if b_type not in ('Inport', 'Outport', 'Subsystem'):
                 primitives.append(b)
        
        logger.info(f"FLATTENER: Primitives found: {[b.name for b in primitives]}")
        
        final_lines = []
        lid = 0
        
        # 3. Resolve connections for each primitive input
        for block in primitives:
             for i in range(block.in_ports):
                 driver = self._resolve_driver(block.name, i)
                 if driver:
                     src, src_p = driver
                     # Create new flat line
                     final_lines.append(DLine(lid, src, src_p, block.name, i, [(0,0), (10,10)]))
                     lid += 1
                     
        return primitives, final_lines

    def _collect_recursive(self, blocks, lines, prefix):
        # 1. Map all lines in this scope
        for line in lines:
            src = f"{prefix}{line.srcblock}"
            dst = f"{prefix}{line.dstblock}"
            key = (dst, line.dstport)
            if key in self.input_drivers:
                # One input port cannot legally have two sources. The new line
                # would silently overwrite the existing driver and drop a
                # connection, so surface it instead of losing it quietly.
                prev_src, prev_port = self.input_drivers[key]
                logger.warning(
                    f"Flattener: duplicate driver for input port {dst}:{line.dstport}. "
                    f"Keeping previous source '{prev_src}':{prev_port}, "
                    f"ignoring new source '{src}':{line.srcport}."
                )
                continue
            self.input_drivers[key] = (src, line.srcport)
            
        # 2. Process blocks
        for block in blocks:
            full_name = f"{prefix}{block.name}"
            # Prefer params for block_type as getattr might be unreliable on copies
            b_type = block.params.get('block_type', '') if block.params else ''
            if not b_type:
                b_type = getattr(block, 'block_type', '')
            
            # Robust verification: check if explicit Subsystem class
            is_subsystem = (b_type == 'Subsystem')
            if not is_subsystem and block.__class__.__name__ == 'Subsystem':
                is_subsystem = True
                b_type = 'Subsystem' # ensure consistency
                
            # DEBUG LOGGING for Subsystem detection
            if 'subsystem' in block.name.lower():
                 logger.info(f"Flattener Inspecting {block.name}: Class={block.__class__.__name__}, b_type={b_type}, is_subsystem={is_subsystem}")
            
            if is_subsystem:
                self.block_map[full_name] = block # Store original (container) for reference if needed
                self._collect_recursive(block.sub_blocks, block.sub_lines, f"{full_name}/")
            else:
                new_b = copy.deepcopy(block)
                new_b.name = full_name
                # Store hierarchy path for debugging/inspector
                new_b.hierarchy_path = prefix[:-1] if prefix else ""
                
                self.flat_blocks.append(new_b)
                self.block_map[full_name] = new_b # Store clone (primitive)

    def _resolve_driver(self, block_name, port_idx):
        """
        Trace back from (block_name, port_idx) until a functional primitive output is found.
        Navigate through Inports, Outports, and Subsystem boundaries.
        """
        curr = (block_name, port_idx)
        visited = set()
        
        while curr:
            if curr in visited:
                # A loop among passthrough boundaries (Inport/Outport/Subsystem)
                # is a malformed diagram. Fail loudly for consistency with the
                # other unresolved-boundary branches below, instead of silently
                # dropping the connection (which would wire downstream zeros).
                msg = (
                    f"Flattener: cycle detected while resolving driver for "
                    f"'{block_name}':{port_idx}. Revisited node "
                    f"'{curr[0]}':{curr[1]}; visited path: {sorted(visited)}."
                )
                logger.error(msg)
                raise RuntimeError(msg)
            visited.add(curr)
            
            # Find what drives 'curr'
            driver = self.input_drivers.get(curr)
            if not driver:
                return None
                
            src_name, src_port = driver
            src_block = self.block_map.get(src_name)
            
            if not src_block:
                return None
                
            b_type = getattr(src_block, 'block_type', '')
            
            if b_type == 'Inport':
                # Inport inside a subsystem.
                # It passes signal from ParentSubsystem input.
                # Find parent subsystem.
                # Name format: "GrandParent/Parent/InportName"
                if '/' in src_name:
                    parent_path, my_name = src_name.rsplit('/', 1)
                    parent_block = self.block_map.get(parent_path)
                    
                    # 1. Try robust map
                    found_idx = None
                    if parent_block and hasattr(parent_block, 'ports_map'):
                         # Reverse lookup: find index where name == my_name
                         for idx, name in parent_block.ports_map.get('input', {}).items():
                             if name == my_name:
                                 found_idx = idx
                                 break
                    
                    # 2. Fallback to name parsing if map failed.
                    #    Restrict to the conventional Inport naming convention
                    #    ("In1", "inport1", "Inport1") so an arbitrary name with
                    #    a trailing digit (e.g. "Sensor3") is not silently
                    #    mapped to the wrong parent input port.
                    if found_idx is None:
                        match = re.match(r'(?:in|inport)(\d+)$', my_name, re.IGNORECASE)
                        if match:
                            found_idx = int(match.group(1)) - 1

                    # Validate the resolved index against the parent's declared
                    # input count when known; an out-of-range index would point
                    # at a non-existent port and mis-wire the diagram. A count
                    # of 0 means the parent's port metadata was never synced
                    # (e.g. ports declared only via sub_blocks), so the upper
                    # bound is unreliable and only the lower bound is enforced.
                    if found_idx is not None:
                        parent_in_ports = getattr(parent_block, 'in_ports', 0) or 0
                        if found_idx < 0 or (parent_in_ports > 0 and found_idx >= parent_in_ports):
                            found_idx = None

                    if found_idx is not None:
                        # Now we look for what drives ParentSubsystem:found_idx
                        curr = (parent_path, found_idx)
                        continue
                    else:
                         # All resolution paths failed. Fail loudly instead of silently
                         # dropping the connection (which would cause downstream blocks
                         # to read zeros with no user-visible error).
                         msg = (
                             f"Flattener: Could not map Inport '{src_name}' to a "
                             f"parent Subsystem input port. Expected 'ports_map' "
                             f"entry or a conventional name ending with a number "
                             f"(e.g. 'In1', 'inport1')."
                         )
                         logger.error(msg)
                         raise RuntimeError(msg)

                else:
                     # Top-level Inport (no parent Subsystem). Inports are
                     # boundary wires, not functional primitives, so they were
                     # excluded from the emitted block list. Returning it as a
                     # connection source would emit a DLine referencing a block
                     # that does not exist in the flattened diagram (a dangling
                     # source). A top-level Inport has no parent to resolve to,
                     # so fail loudly rather than mis-wire the diagram.
                     msg = (
                         f"Flattener: top-level Inport '{src_name}' drives "
                         f"'{block_name}':{port_idx} but has no parent Subsystem "
                         f"to resolve to. Top-level Inports are unsupported as "
                         f"connection sources; wrap the diagram in a Subsystem "
                         f"or replace the Inport with a source block."
                     )
                     logger.error(msg)
                     raise RuntimeError(msg)
            
            elif b_type == 'Subsystem':
                # Driver is a Subsystem block.
                # Find the internal Outport block.
                
                # 1. Try robust map
                outport_name = None
                if src_block and hasattr(src_block, 'ports_map'):
                     outport_name = src_block.ports_map.get('output', {}).get(src_port)
                     
                # 2. Fallback to convention
                if not outport_name:
                     outport_name = f"Out{src_port + 1}"
                     # Also try lowercase "outport"
                     if f"{src_name}/{outport_name}" not in self.block_map:
                         outport_name = f"outport{src_port + 1}"

                full_outport_name = f"{src_name}/{outport_name}"
                
                # Check if this Outport exists
                if full_outport_name in self.block_map:
                    # We need to find what drives this Outport's input (port 0)
                    curr = (full_outport_name, 0)
                    continue
                else:
                     # 3. Last Resort: Scan children for matching "Out*" name or loose match
                     found_child = None
                     prefix = f"{src_name}/"
                     for k in self.block_map:
                         if k.startswith(prefix):
                             child_name = k[len(prefix):]
                             if child_name.lower() == outport_name.lower():
                                 found_child = k
                                 break
                                 
                     if found_child:
                         curr = (found_child, 0)
                         continue

                     # All resolution paths failed. Fail loudly instead of silently
                     # dropping the connection (which would cause downstream blocks
                     # to read zeros with no user-visible error).
                     available_children = [
                         k[len(prefix):] for k in self.block_map if k.startswith(prefix)
                     ]
                     msg = (
                         f"Flattener: Could not resolve Subsystem '{src_name}' "
                         f"output port {src_port} to an internal Outport block. "
                         f"Tried 'ports_map', conventional name '{outport_name}', "
                         f"and case-insensitive scan. Available children: "
                         f"{available_children}."
                     )
                     logger.error(msg)
                     raise RuntimeError(msg)
                     
            elif b_type == 'Outport':
                 # Outport consuming signal. Should not be a driver.
                 return driver
                 
            else:
                # Functional primitive found!
                return driver
                
        return None
