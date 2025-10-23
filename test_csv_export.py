#!/usr/bin/env python
"""
Test script for CSV export functionality.
This creates a simple simulation with scope blocks and tests the export feature.
"""

import sys
import numpy as np
from PyQt5.QtWidgets import QApplication
from lib.lib import SignalPlot

def test_csv_export():
    """Test the CSV export functionality with sample data."""
    app = QApplication(sys.argv)

    # Create sample data
    dt = 0.01
    time = np.linspace(0, 10, 1000)

    # Create labels for multiple scopes
    labels = [
        'sine_wave',  # Single signal scope
        ['cos_wave', 'tan_wave'],  # Multi-signal scope
        'square_wave'  # Another single signal scope
    ]

    # Create sample data vectors
    data_vectors = [
        np.sin(time),  # Sine wave
        np.column_stack([np.cos(time), np.tan(time)]),  # Cos and tan waves
        np.sign(np.sin(time))  # Square wave
    ]

    # Create SignalPlot window
    plot = SignalPlot(dt, labels, len(time))

    # Update with data
    plot.loop(time, data_vectors)

    # Show the plot window
    plot.show()

    print("Test setup complete!")
    print(f"Timeline length: {len(plot.timeline)}")
    print(f"Number of scopes: {len(plot.labels)}")
    print(f"Labels: {plot.labels}")
    print("\nClick 'Export to CSV...' button to test the export functionality.")
    print("You should see a dialog to select which scopes to export.")

    sys.exit(app.exec_())

if __name__ == '__main__':
    test_csv_export()
