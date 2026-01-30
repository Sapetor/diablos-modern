
import os
import importlib
import inspect
from blocks.base_block import BaseBlock

def load_blocks():
    """
    Scans the 'blocks' directory and subdirectories, imports all block modules,
    and returns a list of all block classes.
    Skips abstract base classes that cannot be instantiated.
    """
    block_classes = []
    blocks_dir = os.path.join(os.path.dirname(__file__), '..', 'blocks')

    # Load from top-level blocks directory
    for filename in os.listdir(blocks_dir):
        if filename.endswith('.py') and not filename.startswith('__') and filename != 'base_block.py':
            module_name = f"blocks.{filename[:-3]}"
            try:
                module = importlib.import_module(module_name)
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    # Skip BaseBlock and any abstract base classes
                    if issubclass(obj, BaseBlock) and obj is not BaseBlock and not inspect.isabstract(obj):
                        block_classes.append(obj)
            except Exception as e:
                print(f"Error loading block from {filename}: {e}")

    # Load from subdirectories (pde, optimization, etc.)
    for subdir in os.listdir(blocks_dir):
        subdir_path = os.path.join(blocks_dir, subdir)
        if os.path.isdir(subdir_path) and not subdir.startswith('__'):
            # Check if it's a Python package (has __init__.py)
            init_file = os.path.join(subdir_path, '__init__.py')
            if os.path.exists(init_file):
                for filename in os.listdir(subdir_path):
                    if filename.endswith('.py') and not filename.startswith('__'):
                        module_name = f"blocks.{subdir}.{filename[:-3]}"
                        try:
                            module = importlib.import_module(module_name)
                            for name, obj in inspect.getmembers(module, inspect.isclass):
                                if issubclass(obj, BaseBlock) and obj is not BaseBlock and not inspect.isabstract(obj):
                                    block_classes.append(obj)
                        except Exception as e:
                            print(f"Error loading block from {subdir}/{filename}: {e}")

    return block_classes
