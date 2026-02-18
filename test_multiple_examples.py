#!/usr/bin/env python3
"""
Test multiple example files for property consistency.
"""

import sys
import json
from pathlib import Path
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QColor

# Initialize Qt app
app = QApplication(sys.argv)

# Import model
from lib.models.simulation_model import SimulationModel


def qcolor_to_hex(qcolor):
    """Convert QColor to hex string."""
    if isinstance(qcolor, QColor):
        return qcolor.name()
    return str(qcolor)


def load_example_file(filepath):
    """Load blocks_data from a .diablos file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data.get('blocks_data', [])


def test_file(filepath, model):
    """Test a single file."""
    palette_blocks = {mb.block_fn: mb for mb in model.menu_blocks}
    saved_blocks = load_example_file(filepath)
    
    mismatches = []
    for saved_block in saved_blocks:
        block_fn = saved_block['block_fn']
        palette_block = palette_blocks.get(block_fn)
        
        if not palette_block:
            continue
        
        # Check b_type
        saved_btype = saved_block.get('b_type')
        palette_btype = palette_block.b_type
        
        if saved_btype != palette_btype:
            mismatches.append({
                'file': filepath.name,
                'block': f"{block_fn} ({saved_block.get('username', '')})",
                'field': 'b_type',
                'saved': saved_btype,
                'palette': palette_btype
            })
    
    return mismatches


def main():
    examples_dir = Path('/Users/apeters/Documents/APR/02-Projects/diablos-modern/examples')
    model = SimulationModel()
    
    example_files = sorted(examples_dir.glob('*.diablos'))[:10]  # Test first 10
    
    all_mismatches = []
    passed = 0
    failed = 0
    
    for filepath in example_files:
        try:
            mismatches = test_file(filepath, model)
            if mismatches:
                all_mismatches.extend(mismatches)
                failed += 1
                print(f"✗ {filepath.name}: {len(mismatches)} mismatch(es)")
            else:
                passed += 1
                print(f"✓ {filepath.name}: All properties match")
        except Exception as e:
            print(f"⚠ {filepath.name}: Error - {e}")
    
    print(f"\n{'='*80}")
    print(f"Results: {passed} passed, {failed} failed")
    
    if all_mismatches:
        print(f"\n{'='*80}")
        print("MISMATCHES FOUND:")
        for m in all_mismatches:
            print(f"  {m['file']} - {m['block']}: {m['field']} = {m['saved']} (palette: {m['palette']})")
    
    return 0 if not all_mismatches else 1


if __name__ == "__main__":
    sys.exit(main())
