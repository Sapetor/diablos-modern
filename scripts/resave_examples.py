#!/usr/bin/env python3
"""
Batch re-save all .diablos example files.

Loads each file through FileService (which now applies theme-aware colors
and correct io_edit values), then re-saves so that stale hex colors and
auto-inferred io_edit values are replaced with current values.

Usage:
    python scripts/resave_examples.py
"""

import sys
import os
import json
import glob

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Must create QApplication before importing anything that uses Qt
from PyQt5.QtWidgets import QApplication
app = QApplication(sys.argv)

from lib.models.simulation_model import SimulationModel
from lib.services.file_service import FileService


def resave_file(filepath):
    """Load and re-save a single .diablos file."""
    model = SimulationModel()
    fs = FileService(model)

    # Load raw JSON
    with open(filepath, 'r') as f:
        data = json.load(f)

    # Extract sim params before apply_loaded_data clears them
    sim_data = data.get('sim_data', {})
    sim_params = {
        'sim_time': sim_data.get('sim_time', 1.0),
        'sim_dt': sim_data.get('sim_dt', 0.01),
        'plot_trange': sim_data.get('sim_trange', 100),
    }

    # Reconstruct blocks/lines with current theme colors and io_edit
    fs.apply_loaded_data(data)

    # Re-serialize and save
    modern_ui_data = data.get('modern_ui_data', None)
    new_data = fs.serialize(modern_ui_data=modern_ui_data, sim_params=sim_params)

    # Preserve any extra top-level keys from original file
    # (e.g. _verification_notes, version)
    standard_keys = {'sim_data', 'blocks_data', 'lines_data', 'version', 'modern_ui_data'}
    for key, value in data.items():
        if key not in standard_keys:
            new_data[key] = value
    new_data['version'] = data.get('version', '2.0')

    with open(filepath, 'w') as f:
        json.dump(new_data, f, indent=4)

    return True


def main():
    examples_dir = os.path.join(project_root, 'examples')
    pattern = os.path.join(examples_dir, '*.diablos')
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No .diablos files found in {examples_dir}")
        return

    print(f"Found {len(files)} .diablos files to re-save\n")

    updated = 0
    errors = 0
    for filepath in files:
        name = os.path.basename(filepath)
        try:
            resave_file(filepath)
            print(f"  OK  {name}")
            updated += 1
        except Exception as e:
            print(f"  ERR {name}: {e}")
            errors += 1

    print(f"\nDone: {updated} updated, {errors} errors")


if __name__ == '__main__':
    main()
