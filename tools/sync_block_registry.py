#!/usr/bin/env python3
"""
Scan the blocks/ directory and update _BLOCK_MODULES in lib/block_loader.py.

Run this before building with PyInstaller so the frozen app includes all blocks.
Called automatically by tools/build.sh.
"""

import os
import re
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLOCKS_DIR = os.path.join(PROJECT_ROOT, 'blocks')
BLOCK_LOADER = os.path.join(PROJECT_ROOT, 'lib', 'block_loader.py')


def scan_block_modules(blocks_dir=BLOCKS_DIR):
    """Scan blocks/ directory and return sorted list of dotted module names."""
    modules = []

    for filename in sorted(os.listdir(blocks_dir)):
        if filename.endswith('.py') and not filename.startswith('__') and filename != 'base_block.py':
            modules.append(f"blocks.{filename[:-3]}")

    for subdir in sorted(os.listdir(blocks_dir)):
        subdir_path = os.path.join(blocks_dir, subdir)
        if os.path.isdir(subdir_path) and not subdir.startswith('__'):
            init_file = os.path.join(subdir_path, '__init__.py')
            if os.path.exists(init_file):
                for filename in sorted(os.listdir(subdir_path)):
                    if filename.endswith('.py') and not filename.startswith('__'):
                        modules.append(f"blocks.{subdir}.{filename[:-3]}")

    return modules


def update_block_loader(modules, path=BLOCK_LOADER):
    """Rewrite _BLOCK_MODULES list in block_loader.py."""
    with open(path, 'r') as f:
        content = f.read()

    items = ',\n'.join(f"    '{m}'" for m in modules)
    new_list = f"_BLOCK_MODULES = [\n{items},\n]"

    updated = re.sub(
        r'_BLOCK_MODULES\s*=\s*\[.*?\]',
        new_list,
        content,
        flags=re.DOTALL,
    )

    if updated == content:
        print("block_loader.py: already up to date")
        return False

    with open(path, 'w') as f:
        f.write(updated)
    print(f"block_loader.py: updated _BLOCK_MODULES ({len(modules)} modules)")
    return True


if __name__ == '__main__':
    modules = scan_block_modules()
    changed = update_block_loader(modules)
    sys.exit(0)
