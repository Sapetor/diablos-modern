import logging
import numpy as np
import scipy.signal as signal
from PyQt5.QtCore import Qt
import pyqtgraph as pg
from .base_analyzer import BaseAnalyzer

logger = logging.getLogger(__name__)

class RootLocusAnalyzer(BaseAnalyzer):
    """Analyzer for generating Root Locus plots."""
    
    def analyze(self, source_block, canvas):
        """Generate Root Locus plot."""
        logger.debug(f"RootLocusAnalyzer called for {source_block.name}")
        
        model, sys_block = self._find_connected_transfer_function(source_block, canvas)
        
        if not model:
            return None
            
        try:
            num, den, dt = model
            
            # Root Locus Calculation
            # Gain range K from 0 to inf? 
            # Use logspace for K
            k_values = np.logspace(-2, 2, 500)
            k_values = np.insert(k_values, 0, 0) # Start at K=0
            
            all_roots = []
            
            # We need standard polynomial form strictly here
            # num/den are coefficients
            
            # Ensure they are 1D arrays
            num = np.atleast_1d(num)
            den = np.atleast_1d(den)
            
            if num.ndim > 1: num = num.flatten()
            if den.ndim > 1: den = den.flatten()
            
            # Pad to same length? No, np.roots handles it.
            # Denominator typically higher order.
            
            # Actually, char eq is Den(s) + K * Num(s) = 0
            # We need to pad coefficients to add them array-wise
            max_len = max(len(num), len(den))
            padded_num = np.pad(num, (max_len - len(num), 0), 'constant')
            padded_den = np.pad(den, (max_len - len(den), 0), 'constant')
            
            for k in k_values:
                char_poly = padded_den + k * padded_num
                roots = np.roots(char_poly)
                all_roots.append(roots)
                
            all_roots = np.array(all_roots)
            
            # Sorting/Threading branches
            # Simple nearest neighbor approach to keep lines smooth
            # (Copied from original implementation)
            num_poles = all_roots.shape[1]
            sorted_roots = [all_roots[0]]
            
            for i in range(1, len(all_roots)):
                current = all_roots[i]
                previous = sorted_roots[-1]
                matched = []
                available = list(range(len(current)))
                
                for prev_root in previous:
                    if not available: break
                    # Distance to previous
                    distances = [abs(current[j] - prev_root) for j in available]
                    closest_idx = available[np.argmin(distances)]
                    matched.append(current[closest_idx])
                    available.remove(closest_idx)
                    
                for idx in available:
                     matched.append(current[idx])
                     
                sorted_roots.append(np.array(matched))
                
            sorted_roots = np.array(sorted_roots)
            
            # Plotting
            from PyQt5.QtWidgets import QWidget
            plot_window = QWidget()
            t = f"Root Locus: {sys_block.name}"
            plot_widget = self._setup_plot_widget(plot_window, t)
            
            plot_widget.setLabel('left', 'Imaginary Axis', units='rad/s', color='#000000')
            plot_widget.setLabel('bottom', 'Real Axis', units='1/s', color='#000000')
            plot_widget.setTitle(f"Root Locus Plot: {sys_block.name}")
            
            colors = [(100, 149, 237), (237, 100, 100), (100, 237, 149),
                      (237, 149, 100), (149, 100, 237), (237, 237, 100)]
                      
            for pole_idx in range(num_poles):
                branch_real = sorted_roots[:, pole_idx].real
                branch_imag = sorted_roots[:, pole_idx].imag
                color = colors[pole_idx % len(colors)]
                plot_widget.plot(branch_real, branch_imag, pen=pg.mkPen(color, width=2))
                
            # Open Loop Poles (X)
            poles = np.roots(den) # Roots of pure denominator (K=0)
            plot_widget.plot(poles.real, poles.imag, pen=None, symbol='x', symbolSize=12, symbolPen=pg.mkPen('r', width=3))
            
            # Open Loop Zeros (O)
            if np.any(num):
                zeros = np.roots(num)
                plot_widget.plot(zeros.real, zeros.imag, pen=None, symbol='o', symbolSize=12, symbolPen=pg.mkPen('g', width=3))

            plot_widget.addLine(x=0, pen=pg.mkPen('k', width=1, style=Qt.DashLine))
            plot_widget.addLine(y=0, pen=pg.mkPen('k', width=1, style=Qt.DashLine))
            plot_widget.showGrid(x=True, y=True)
            
            self._position_window(plot_window)
            plot_window.show()
            return plot_window
            
        except Exception as e:
            logger.error(f"Error calculating Root Locus: {e}")
            return None
