
import os
import importlib
import inspect
from blocks.base_block import BaseBlock

def load_blocks():
    """
    Scans the 'blocks' directory, imports all block modules, and returns a list of all block classes.
    Skips abstract base classes that cannot be instantiated.
    """
    block_classes = []
    blocks_dir = os.path.join(os.path.dirname(__file__), '..', 'blocks')

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

    return block_classes
