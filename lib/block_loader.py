
import os
import sys
import importlib
import inspect
from blocks.base_block import BaseBlock

# Static registry of all block modules — used when running as a frozen
# PyInstaller bundle (where filesystem scanning doesn't work).
_BLOCK_MODULES = [
    'blocks.abs_block',
    'blocks.assert_block',
    'blocks.bodemag',
    'blocks.bodephase',
    'blocks.constant',
    'blocks.deadband',
    'blocks.delay',
    'blocks.demux',
    'blocks.derivative',
    'blocks.discrete_statespace',
    'blocks.discrete_transfer_function',
    'blocks.display',
    'blocks.exponential',
    'blocks.export',
    'blocks.external',
    'blocks.fft',
    'blocks.first_order_hold',
    'blocks.from_block',
    'blocks.gain',
    'blocks.goto',
    'blocks.hysteresis',
    'blocks.inport',
    'blocks.input_helpers',
    'blocks.integrator',
    'blocks.math_function',
    'blocks.mux',
    'blocks.noise',
    'blocks.nyquist',
    'blocks.outport',
    'blocks.param_templates',
    'blocks.pid',
    'blocks.prbs',
    'blocks.product',
    'blocks.ramp',
    'blocks.rate_limiter',
    'blocks.rate_transition',
    'blocks.rootlocus',
    'blocks.saturation',
    'blocks.scope',
    'blocks.selector',
    'blocks.sigproduct',
    'blocks.sine',
    'blocks.statespace',
    'blocks.statespace_base',
    'blocks.step',
    'blocks.subsystem',
    'blocks.sum',
    'blocks.switch',
    'blocks.terminator',
    'blocks.transfer_function',
    'blocks.transport_delay',
    'blocks.wave_generator',
    'blocks.xygraph',
    'blocks.zero_order_hold',
    'blocks.optimization.constraint',
    'blocks.optimization.cost_function',
    'blocks.optimization.data_fit',
    'blocks.optimization.optimizer',
    'blocks.optimization.parameter',
    'blocks.optimization_primitives.adam',
    'blocks.optimization_primitives.linear_system_solver',
    'blocks.optimization_primitives.momentum',
    'blocks.optimization_primitives.numerical_gradient',
    'blocks.optimization_primitives.objective_function',
    'blocks.optimization_primitives.residual_norm',
    'blocks.optimization_primitives.root_finder',
    'blocks.optimization_primitives.state_variable',
    'blocks.optimization_primitives.vector_gain',
    'blocks.optimization_primitives.vector_perturb',
    'blocks.optimization_primitives.vector_sum',
    'blocks.pde.advection_equation_1d',
    'blocks.pde.advection_equation_2d',
    'blocks.pde.diffusion_reaction_1d',
    'blocks.pde.field_processing',
    'blocks.pde.field_processing_2d',
    'blocks.pde.heat_equation_1d',
    'blocks.pde.heat_equation_2d',
    'blocks.pde.wave_equation_1d',
    'blocks.pde.wave_equation_2d',
]


def _collect_block_classes(module_names):
    """Import modules by name and collect BaseBlock subclasses."""
    block_classes = []
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, BaseBlock) and obj is not BaseBlock and not inspect.isabstract(obj):
                    block_classes.append(obj)
        except Exception as e:
            print(f"Error loading block {module_name}: {e}")
    return block_classes


def load_blocks():
    """
    Imports all block modules and returns a list of all block classes.
    Skips abstract base classes that cannot be instantiated.

    In frozen (PyInstaller) mode, uses a static registry since filesystem
    scanning is not available. In development mode, scans the blocks directory.
    """
    if getattr(sys, 'frozen', False):
        return _collect_block_classes(_BLOCK_MODULES)

    # Development mode: scan the filesystem
    block_modules = []
    blocks_dir = os.path.join(os.path.dirname(__file__), '..', 'blocks')

    # Load from top-level blocks directory
    for filename in os.listdir(blocks_dir):
        if filename.endswith('.py') and not filename.startswith('__') and filename != 'base_block.py':
            block_modules.append(f"blocks.{filename[:-3]}")

    # Load from subdirectories (pde, optimization, etc.)
    for subdir in os.listdir(blocks_dir):
        subdir_path = os.path.join(blocks_dir, subdir)
        if os.path.isdir(subdir_path) and not subdir.startswith('__'):
            init_file = os.path.join(subdir_path, '__init__.py')
            if os.path.exists(init_file):
                for filename in os.listdir(subdir_path):
                    if filename.endswith('.py') and not filename.startswith('__'):
                        block_modules.append(f"blocks.{subdir}.{filename[:-3]}")

    return _collect_block_classes(block_modules)
