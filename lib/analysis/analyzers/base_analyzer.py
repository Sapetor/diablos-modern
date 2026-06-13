import logging
import numpy as np
import scipy.signal as signal
from PyQt5.QtWidgets import QVBoxLayout
import pyqtgraph as pg
from lib.safe_eval import safe_literal, SafeEvalError

logger = logging.getLogger(__name__)

class BaseAnalyzer:
    """Base class for all control system analyzers."""
    
    def __init__(self, parent=None):
        self.parent = parent
        # We assume self.parent.canvas exists or is passed? 
        # Actually ControlSystemAnalyzer passes 'parent' as 'ModernCanvas' if interacting directly.
        # But BaseAnalyzer init only takes parent.
        # We need access to the canvas/dsim.
        # Let's change analyze signature to take 'canvas' instead of 'connection_manager'.
        
    def analyze(self, source_block, canvas, **kwargs):
        """Perform analysis on the given source block.
        
        Args:
            source_block: The block to analyze.
            canvas: The ModernCanvas instance (access to dsim.model).
            
        Returns:
            The created plot window, or None if analysis failed.
        """
        raise NotImplementedError("Subclasses must implement analyze()")

    def _get_input_connections(self, block_name, canvas):
        """Find lines connected to inputs of the given block."""
        # lines are in canvas.dsim.model.line_list
        inputs = []
        try:
            # Accessing model safely
            line_list = canvas.dsim.model.line_list
            for line in line_list:
                if line.dstblock == block_name:
                    inputs.append(line)
        except AttributeError:
            logger.error("Could not access line_list from canvas")
        return inputs

    def _extract_system_model(self, block, canvas, visited=None):
        """Extract LTI system model (num, den, dt) from a block or subsystem."""
        if visited is None:
            visited = set()
            
        if block in visited:
            logger.warning(f"Loop detected at {block.name}")
            return None
            
        visited.add(block)
        
        # 1. Direct TransferFunction/DiscreteTransferFunction/StateSpace
        if hasattr(block, 'get_transfer_function'):
            return block.get_transfer_function()
            
        # 2. Subsystem Support (Simplified)
        if block.block_fn == 'Subsystem':
            # TODO: Robust subsystem tracing requires flattening logic or recursive graph search.
            # For now, we abort if not implementing get_transfer_function (which compiles it).
            pass

        # 3. Parameter-based Extraction (fallback for legacy blocks)
        params = getattr(block, 'params', {})
        
        # A) Explicit Transfer Function
        if 'numerator' in params and 'denominator' in params:
            try:
                # Ensure we handle string lists or mixed types by forcing float conversion
                # Safe eval if string? Usually params are already lists of values from PropertyEditor.
                # If PropertyEditor stores strings "[1, 2]", we might need safe parsing.
                # But standard DBlock params should be values.
                
                # Check if it's a string representation of a list
                def clean_param(val):
                    if isinstance(val, str):
                        # Try to parse "[1, 2]" string params if they exist
                        val = val.strip()
                        if val.startswith('[') and val.endswith(']'):
                            return safe_literal(val)
                    return val

                n_val = clean_param(params['numerator'])
                d_val = clean_param(params['denominator'])
                
                num = np.array(n_val, dtype=float)
                den = np.array(d_val, dtype=float)
                dt = params.get('sampling_time', 0.0)
                return num, den, float(dt)
            except Exception:
                logger.warning("Failed to extract transfer-function model from "
                               "numerator/denominator params", exc_info=True)
                return None
                
        # B) State Space 
        if 'A' in params and 'B' in params and 'C' in params and 'D' in params:
             try:
                sys = signal.StateSpace(np.array(params['A']), np.array(params['B']), 
                                      np.array(params['C']), np.array(params['D']), 
                                      dt=params.get('sampling_time', 0) or None)
                sys_tf = sys.to_tf()
                return sys_tf.num, sys_tf.den, float(sys.dt) if sys.dt else 0.0
             except Exception:
                 logger.warning("Failed to extract state-space model from "
                                "A/B/C/D params", exc_info=True)
                 return None

        # C) PID Controller
        if block.block_fn == 'PID':
             try:
                 # Param keys match the PID block definition (capitalized).
                 kp = float(params.get('Kp', params.get('kp', 1.0)))
                 ki = float(params.get('Ki', params.get('ki', 0.0)))
                 kd = float(params.get('Kd', params.get('kd', 0.0)))
                 n = float(params.get('N', 20.0))
                 # Match the PID block's filtered-derivative form:
                 #   C(s) = Kp + Ki/s + Kd*N*s/(s + N)
                 # Combined over the common denominator s*(s + N):
                 #   num = (Kp + Kd*N)*s^2 + (Kp*N + Ki)*s + Ki*N
                 #   den = s^2 + N*s
                 # This yields a proper TF (deg 2 / deg 2) instead of the
                 # improper ideal form (Kd*s^2 + Kp*s + Ki)/s, which crashed
                 # downstream root-locus computation when Kd != 0.
                 if n != 0.0:
                     num = np.array([kp + kd * n, kp * n + ki, ki * n], dtype=float)
                     den = np.array([1.0, n, 0.0], dtype=float)
                 else:
                     # Without a derivative filter the term Kd*N*s/(s+N) -> 0,
                     # leaving the ideal PI form (Kp*s + Ki)/s (proper).
                     num = np.array([kp, ki], dtype=float)
                     den = np.array([1.0, 0.0], dtype=float)
                 return num, den, 0.0
             except Exception:
                 logger.warning("Failed to extract transfer-function model "
                                "from PID params", exc_info=True)
                 return None

        # D) Generic check
        if hasattr(block, 'get_transfer_function'):
             return block.get_transfer_function()

        return None
        
    def _find_connected_transfer_function(self, start_block, canvas):
        """Trace backwards to find a valid LTI system model."""
        # Check start block first
        model = self._extract_system_model(start_block, canvas)
        if model:
            return model, start_block
            
        # Limit recursion
        queue = [(start_block, 0)]
        visited = {start_block.name}
        
        while queue:
            current, depth = queue.pop(0)
            if depth > 5: continue
            
            # Get input connections
            incoming_lines = self._get_input_connections(current.name, canvas)
            
            for line in incoming_lines:
                source_name = line.srcblock
                # Resolve block object
                # Canvas -> DSim -> Model -> get_block_by_name
                try:
                    source_block = canvas.dsim.model.get_block_by_name(source_name)
                except AttributeError:
                    logger.error("Could not access dsim.model from canvas")
                    source_block = None
                
                if source_block and source_name not in visited:
                    visited.add(source_name)
                    model = self._extract_system_model(source_block, canvas)
                    if model:
                        return model, source_block
                    queue.append((source_block, depth + 1))
                         
        return None, None

    def _setup_plot_widget(self, plot_window, title):
        """Configure standard white-bg plot widget."""
        plot_window.setWindowTitle(title)
        layout = QVBoxLayout()
        
        plot_widget = pg.PlotWidget()
        plot_widget.setBackground('w')
        plot_widget.getAxis('bottom').setPen('k')
        plot_widget.getAxis('left').setPen('k')
        plot_widget.getAxis('bottom').setTextPen('k')
        plot_widget.getAxis('left').setTextPen('k')
        
        layout.addWidget(plot_widget)
        plot_window.setLayout(layout)
        return plot_widget

    def _position_window(self, window):
        """Position window with cascading offset."""
        # We need a shared counter for cascading.
        # Since BaseAnalyzer is instantiated per facade, but the facade persists on the canvas,
        # we can check if 'parent' (ControlSystemAnalyzer facade or Canvas) has a window count.
        
        # Best approach: Use a static/class attribute or coordinate through the parent.
        # Let's try to access the facade's window list if possible, or just use a screen-relative system.
        
        base_x = 100
        base_y = 100
        offset = 30
        
        # If we can access existing windows, we calculate offset
        # But 'self.parent' is ModernCanvas (usually).
        # And ModernCanvas has self.analyzer (the facade).
        # The facade has self.plot_windows.
        
        # Try to find the facade to get window count for cascading.
        # self.parent is likely ModernCanvas (passed in init), and
        # ModernCanvas.analyzer is the facade holding plot_windows.
        # Compute the count defensively so a missing attribute simply
        # falls back to count=0 rather than masking unexpected errors.
        count = 0
        facade = getattr(self.parent, 'analyzer', None)
        plot_windows = getattr(facade, 'plot_windows', None)
        if plot_windows is not None:
            count = len(plot_windows)

        # Cascade
        x = base_x + (count * offset)
        y = base_y + (count * offset)

        # Keep within screen bounds (roughly): reset if too far
        if x > 800: x = base_x
        if y > 600: y = base_y

        window.move(x, y)
        window.resize(600, 400)
