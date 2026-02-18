#!/usr/bin/env python3
"""
Compare block properties in saved .diablos files vs palette blocks.

This script:
1. Reads a .diablos file and extracts block properties
2. Instantiates a SimulationModel to get palette blocks with theme colors
3. Compares saved vs palette values for: b_color, io_edit, b_type, fn_name
4. Prints a detailed comparison table
"""

import sys
import json
from pathlib import Path
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QColor

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Initialize Qt app (required for QColor and theme_manager)
app = QApplication(sys.argv)

# Now import model classes
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


def main():
    example_file = project_root / "examples" / "c01_tank_feedback.diablos"
    
    if not example_file.exists():
        print(f"Error: {example_file} not found")
        sys.exit(1)
    
    print(f"\n{'='*120}")
    print(f"BLOCK PROPERTY COMPARISON: {example_file.name}")
    print(f"{'='*120}\n")
    
    # Load saved blocks from example file
    saved_blocks = load_example_file(example_file)
    print(f"Loaded {len(saved_blocks)} blocks from file\n")
    
    # Initialize simulation model (this loads palette blocks with theme colors)
    model = SimulationModel()
    palette_blocks = {mb.block_fn: mb for mb in model.menu_blocks}
    print(f"Loaded {len(palette_blocks)} block types from palette\n")
    
    # Comparison fields
    fields = ['b_color', 'io_edit', 'b_type', 'fn_name']
    
    # Build comparison table
    results = []
    all_match = True
    
    for saved_block in saved_blocks:
        block_fn = saved_block['block_fn']
        palette_block = palette_blocks.get(block_fn)
        
        if not palette_block:
            print(f"WARNING: {block_fn} not found in palette")
            continue
        
        block_results = {
            'block_fn': block_fn,
            'username': saved_block.get('username', ''),
            'comparisons': {}
        }
        
        for field in fields:
            saved_val = saved_block.get(field)
            
            # Get palette value with proper conversion
            if field == 'b_color':
                palette_val = qcolor_to_hex(palette_block.b_color)
            else:
                palette_val = getattr(palette_block, field, None)
            
            # Convert saved b_color to hex if it's not already
            if field == 'b_color' and saved_val:
                saved_val = saved_val.lower()  # Normalize hex
            
            # Normalize palette hex too
            if field == 'b_color' and palette_val:
                palette_val = palette_val.lower()
            
            match = saved_val == palette_val
            if not match:
                all_match = False
            
            block_results['comparisons'][field] = {
                'saved': saved_val,
                'palette': palette_val,
                'match': match
            }
        
        results.append(block_results)
    
    # Print table
    print(f"{'Block':<15} {'Field':<12} {'Saved Value':<25} {'Palette Value':<25} {'Match?':<8}")
    print("-" * 120)
    
    for i, block_result in enumerate(results):
        block_fn = block_result['block_fn']
        username = block_result['username']
        label = f"{block_fn} ({username})" if username else block_fn
        
        comparisons = block_result['comparisons']
        first_field = True
        
        for field in fields:
            comp = comparisons[field]
            saved = str(comp['saved'])[:25]
            palette = str(comp['palette'])[:25]
            match_str = "✓ YES" if comp['match'] else "✗ NO"
            match_color = "✓ YES" if comp['match'] else "✗ NO"
            
            if first_field:
                print(f"{label:<15} {field:<12} {saved:<25} {palette:<25} {match_color:<8}")
                first_field = False
            else:
                print(f"{'  ':<15} {field:<12} {saved:<25} {palette:<25} {match_color:<8}")
        
        print("-" * 120)
    
    # Summary
    print(f"\n{'='*120}")
    if all_match:
        print("✓ ALL PROPERTIES MATCH - Saved blocks match palette blocks perfectly!")
    else:
        print("✗ MISMATCHES FOUND - See details above")
    print(f"{'='*120}\n")
    
    return 0 if all_match else 1


if __name__ == "__main__":
    sys.exit(main())
