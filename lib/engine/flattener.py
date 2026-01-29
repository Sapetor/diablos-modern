
import logging
import copy
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
            self.input_drivers[(dst, line.dstport)] = (src, line.srcport)
            
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
                logger.error("Cycle detected during flattening")
                return None
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
                    
                    # 2. Fallback to name parsing if map failed
                    if found_idx is None:
                        try:
                             # Handle "In1", "inport1", "Inport1"
                             import re
                             match = re.search(r'(\d+)$', my_name)
                             if match:
                                 found_idx = int(match.group(1)) - 1
                        except (ValueError, AttributeError):
                            pass
                            
                    if found_idx is not None:
                        # Now we look for what drives ParentSubsystem:found_idx
                        curr = (parent_path, found_idx)
                        continue
                    else:
                         logger.warning(f"Could not map Inport {src_name} to parent port.")
                         return None

                else:
                     # Top level Inport - Valid Source
                     return driver
            
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
                         
                     logger.warning(f"Could not find Outport {full_outport_name}. Available: {list(self.block_map.keys())}")
                     return None
                     
            elif b_type == 'Outport':
                 # Outport consuming signal. Should not be a driver.
                 return driver
                 
            else:
                # Functional primitive found!
                return driver
                
        return None
