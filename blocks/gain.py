
import logging
import numpy as np
from blocks.base_block import BaseBlock

logger = logging.getLogger(__name__)


class GainBlock(BaseBlock):
    """
    A block that scales an input by a factor.
    Supports scalar, vector, and matrix gains for MIMO systems.
    
    - Scalar gain: y = K * u (element-wise)
    - Vector gain: y = K * u (element-wise, same length required)
    - Matrix gain: y = K @ u (matrix-vector multiplication)
    """

    @property
    def block_name(self):
        return "Gain"

    @property
    def category(self):
        return "Math"

    @property
    def color(self):
        return "yellow"

    @property
    def doc(self):
        return "Scales input by gain. Supports scalar, vector (element-wise), or matrix (K @ u) gains."

    @property
    def params(self):
        return {
            "gain": {
                "type": "matrix",
                "default": 1.0,
                "doc": "Gain value: scalar, vector, or matrix. Matrix uses y = K @ u."
            }
        }

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    @property
    def use_port_grid_snap(self):
        """
        Gain block uses triangular shape, so ports should not snap to grid
        for perfect alignment with triangle geometry.
        """
        return False

    def draw_icon(self, block_rect):
        """Gain uses triangular shape - handled specially in DBlock.draw_Block."""
        return None

    def execute(self, time, inputs, params):
        try:
            # Get input as numpy array
            u = np.atleast_1d(inputs.get(0, 0)).astype(float)
            
            # Get gain parameter
            K_raw = params.get('gain', 1.0)
            
            # Parse gain - handle string representation of matrices
            if isinstance(K_raw, str):
                try:
                    # Try to evaluate as numpy expression (e.g., "[[1,0],[0,1]]")
                    K = np.array(eval(K_raw), dtype=float)
                except:
                    K = np.array([float(K_raw)], dtype=float)
            else:
                K = np.atleast_1d(K_raw).astype(float)
            
            # Determine operation based on gain shape
            if K.ndim == 2:
                # Matrix gain: y = K @ u
                u = u.flatten()
                if K.shape[1] != len(u):
                    # Dimension mismatch - try to pad or truncate
                    if len(u) < K.shape[1]:
                        u = np.pad(u, (0, K.shape[1] - len(u)))
                    else:
                        u = u[:K.shape[1]]
                y = K @ u
            elif K.ndim == 1 and len(K) > 1:
                # Vector gain: element-wise multiplication
                if len(K) == len(u):
                    y = K * u
                elif len(K) == 1:
                    y = K[0] * u
                else:
                    # Broadcast to minimum length
                    min_len = min(len(K), len(u))
                    y = K[:min_len] * u[:min_len]
            else:
                # Scalar gain: simple multiplication
                y = K.flatten()[0] * u
            
            return {0: y}
        except (ValueError, TypeError) as e:
            logger.error(f"Gain block error: {str(e)}")
            return {'E': True, 'error': str(e)}

