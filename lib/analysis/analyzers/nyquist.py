import logging
import numpy as np
import scipy.signal as signal
from PyQt5.QtCore import Qt
import pyqtgraph as pg
from .base_analyzer import BaseAnalyzer

logger = logging.getLogger(__name__)

class NyquistAnalyzer(BaseAnalyzer):
    """Analyzer for generating Nyquist plots."""
    
    def analyze(self, source_block, canvas):
        """Generate Nyquist plot."""
        logger.debug(f"NyquistAnalyzer called for {source_block.name}")
        
        model, sys_block = self._find_connected_transfer_function(source_block, canvas)
        
        if not model:
            return None
            
        try:
            num, den, dt = model
            
            # Calculation
             # Create discrete or continuous system
            if dt > 0:
                sys = signal.TransferFunction(num, den, dt=dt)
                w_max = np.pi / dt
                w = np.logspace(np.log10(w_max/1000.0), np.log10(w_max), 1000)
            else:
                sys = signal.TransferFunction(num, den)
                w = np.logspace(-3, 3, 1000)
                
            w, H = signal.freqresp(sys, w=w)
            
            real = H.real
            imag = H.imag
            
            # Plotting
            from PyQt5.QtWidgets import QWidget
            plot_window = QWidget()
            t = f"Nyquist Plot: {sys_block.name}"
            plot_widget = self._setup_plot_widget(plot_window, t)
            
            plot_widget.setLabel('left', 'Imaginary', color='#000000')
            plot_widget.setLabel('bottom', 'Real', color='#000000')
            plot_widget.setAspectLocked(True)
            
            # Plot curve
            plot_widget.plot(real, imag, pen=pg.mkPen('b', width=2), name='System')
            # Mirror for full Nyquist
            plot_widget.plot(real, -imag, pen=pg.mkPen('b', width=2, style=Qt.DashLine))
            
            # Critical point -1+0j
            plot_widget.plot([-1], [0], symbol='+', symbolSize=12, symbolPen='r')
            
            # Axes lines
            plot_widget.addLine(x=0, pen=pg.mkPen('k', width=1))
            plot_widget.addLine(y=0, pen=pg.mkPen('k', width=1))
            
            plot_widget.showGrid(x=True, y=True)
            self._position_window(plot_window)
            plot_window.show()
            return plot_window
            
        except Exception as e:
            logger.error(f"Error calculating Nyquist plot: {e}")
            return None
