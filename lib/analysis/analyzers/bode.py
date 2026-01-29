import logging
import numpy as np
import scipy.signal as signal
from PyQt5.QtCore import Qt
import pyqtgraph as pg
from .base_analyzer import BaseAnalyzer

logger = logging.getLogger(__name__)

class BodeAnalyzer(BaseAnalyzer):
    """Analyzer for generating Bode Magnitude and Phase plots."""
    
    def analyze(self, source_block, canvas, phase_only=False):
        """Generate Bode plot (Magnitude or Phase)."""
        logger.debug(f"BodeAnalyzer called for {source_block.name} (Phase={phase_only})")
        
        model, sys_block = self._find_connected_transfer_function(source_block, canvas)
        
        if not model:
            logger.warning("No valid linear system found for Bode analysis")
            return None
            
        try:
            num, den, dt = model
            
            # Create discrete or continuous system
            if dt > 0:
                sys = signal.TransferFunction(num, den, dt=dt)
                w_max = np.pi / dt
                w = np.logspace(np.log10(w_max/1000.0), np.log10(w_max), 500)
            else:
                sys = signal.TransferFunction(num, den)
                w = np.logspace(-2, 2, 500) # Default range
                
            w, mag, phase = sys.bode(w=w)
            
            from PyQt5.QtWidgets import QWidget # Lazy import
            plot_window = QWidget()
            
            if phase_only:
                title = f"Bode Phase Plot: {sys_block.name}"
                plot_widget = self._setup_plot_widget(plot_window, title)
                plot_widget.setLabel('left', 'Phase', units='deg', color='#000000')
                plot_widget.setLabel('bottom', 'Frequency', units='rad/s', color='#000000')
                plot_widget.setLogMode(x=True, y=False)
                
                # Filter out negative/zero freq if any (log mode safety)
                valid = w > 0
                plot_widget.plot(w[valid], phase[valid], pen=pg.mkPen('b', width=2))
                
                # Add -180 deg line
                plot_widget.addLine(y=-180, pen=pg.mkPen('r', width=1, style=Qt.DashLine))
                
            else: # Magnitude
                title = f"Bode Magnitude Plot: {sys_block.name}"
                plot_widget = self._setup_plot_widget(plot_window, title)
                plot_widget.setLabel('left', 'Magnitude', units='dB', color='#000000')
                plot_widget.setLabel('bottom', 'Frequency', units='rad/s', color='#000000')
                plot_widget.setLogMode(x=True, y=False)
                
                valid = w > 0
                plot_widget.plot(w[valid], mag[valid], pen=pg.mkPen('b', width=2))
                
                # Add 0 dB line
                plot_widget.addLine(y=0, pen=pg.mkPen('k', width=1, style=Qt.DashLine))
                
                # Add Nyquist line for discrete
                if dt > 0:
                     nyquist_freq = np.pi / dt
                     line = pg.InfiniteLine(pos=nyquist_freq, angle=90, pen=pg.mkPen('r', style=Qt.DashLine))
                     # We can't easily add text to InfiniteLine in some pg versions, use standard TextItem if needed?
                     # Simpler: Just the line
                     plot_widget.addItem(line)

            plot_widget.showGrid(x=True, y=True)
            self._position_window(plot_window)
            plot_window.show()
            return plot_window
            
        except Exception as e:
            logger.error(f"Error calculating Bode plot: {e}")
            return None
