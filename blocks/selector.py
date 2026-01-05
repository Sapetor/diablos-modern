import numpy as np
from blocks.base_block import BaseBlock


class SelectorBlock(BaseBlock):
    """
    Selects specific elements from a vector signal.
    Useful for extracting specific states or outputs from MIMO systems.
    """

    @property
    def block_name(self):
        return "Selector"

    @property
    def category(self):
        return "Routing"

    @property
    def color(self):
        return "orange"

    @property
    def doc(self):
        return "Selects specific elements from a vector. Use comma-separated indices (0-based)."

    @property
    def params(self):
        return {
            "indices": {
                "type": "string", 
                "default": "0", 
                "doc": "Comma-separated indices to select (0-based). E.g., '0,2,4' or '1:3' for range."
            },
        }

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    def execute(self, time, inputs, params):
        # Get input vector
        u = np.atleast_1d(inputs.get(0, 0)).flatten()
        
        indices_str = params.get("indices", "0")
        
        try:
            # Parse indices
            indices = self._parse_indices(indices_str, len(u))
            
            # Extract selected elements
            if len(indices) == 1:
                y = np.array([u[indices[0]]])
            else:
                y = u[indices]
            
            return {0: y}
        except (IndexError, ValueError) as e:
            return {'E': True, 'error': f"Selector error: {str(e)}"}
    
    def _parse_indices(self, indices_str, max_len):
        """Parse index string into list of indices."""
        indices = []
        
        for part in indices_str.split(','):
            part = part.strip()
            
            if ':' in part:
                # Range notation (e.g., "1:3" means indices 1, 2)
                parts = part.split(':')
                start = int(parts[0]) if parts[0] else 0
                end = int(parts[1]) if parts[1] else max_len
                indices.extend(range(start, min(end, max_len)))
            else:
                # Single index
                idx = int(part)
                if idx < 0:
                    idx = max_len + idx  # Support negative indices
                if 0 <= idx < max_len:
                    indices.append(idx)
        
        return indices if indices else [0]
