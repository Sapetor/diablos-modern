import logging
import numpy as np
import scipy.signal as signal
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QMessageBox
import pyqtgraph as pg
from PyQt5.QtGui import QCursor
from PyQt5.QtCore import Qt

# Import modular analyzers
from lib.analysis.analyzers.bode import BodeAnalyzer
from lib.analysis.analyzers.nyquist import NyquistAnalyzer
from lib.analysis.analyzers.root_locus import RootLocusAnalyzer

logger = logging.getLogger(__name__)

class ControlSystemAnalyzer:
    """
    Facade class for control system analysis.
    Delegates actual analysis to specific analyzer classes.
    """
    
    def __init__(self, canvas, parent=None):
        self.canvas = canvas
        self.parent = parent
        self.plot_windows = []
        
        # Instantiate analyzers
        # We pass parent for error message boxes if needed
        self.bode_analyzer = BodeAnalyzer(parent)
        self.nyquist_analyzer = NyquistAnalyzer(parent)
        self.root_locus_analyzer = RootLocusAnalyzer(parent)

    def generate_bode_plot(self, block_name_or_obj=None):
        """Generate Bode Magnitude plot."""
        block = self._resolve_block(block_name_or_obj)
        if not block: return
        
        win = self.bode_analyzer.analyze(block, self.canvas, phase_only=False)
        if win: self.plot_windows.append(win)

    def generate_bode_phase_plot(self, block_name_or_obj=None):
        """Generate Bode Phase plot."""
        block = self._resolve_block(block_name_or_obj)
        if not block: return
        
        win = self.bode_analyzer.analyze(block, self.canvas, phase_only=True)
        if win: self.plot_windows.append(win)

    def generate_nyquist_plot(self, block_name_or_obj=None):
        """Generate Nyquist plot."""
        block = self._resolve_block(block_name_or_obj)
        if not block: return
        
        win = self.nyquist_analyzer.analyze(block, self.canvas)
        if win: self.plot_windows.append(win)

    def generate_root_locus(self, block_name_or_obj=None):
        """Generate Root Locus plot."""
        block = self._resolve_block(block_name_or_obj)
        if not block: return
        
        win = self.root_locus_analyzer.analyze(block, self.canvas)
        if win: self.plot_windows.append(win)

    def _resolve_block(self, block_name_or_obj):
        """Helper to get block object from name or object."""
        if not block_name_or_obj:
            logger.warning("No block specified for analysis")
            return None
            
        if isinstance(block_name_or_obj, str):
            try:
                return self.canvas.dsim.model.get_block_by_name(block_name_or_obj)
            except AttributeError:
                logger.error("Canvas does not have dsim.model attached")
                return None
        return block_name_or_obj
        
    def close_all_plots(self):
        """Close all open analysis windows."""
        for win in self.plot_windows:
            win.close()
        self.plot_windows.clear()
