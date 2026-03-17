# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for DiaBloS Modern.

Build with:
    pyinstaller diablos.spec

Output:
    dist/DiaBloS.app  (macOS)
    dist/DiaBloS.exe  (Windows)
"""

import sys
from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

# --- Hidden imports ---
# Blocks are loaded dynamically via importlib (lib/block_loader.py),
# so PyInstaller can't discover them through static analysis.
hidden_imports_blocks = [
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
    'blocks.outport',
    'blocks.param_templates',
    'blocks.pde.advection_equation_1d',
    'blocks.pde.advection_equation_2d',
    'blocks.pde.diffusion_reaction_1d',
    'blocks.pde.field_processing',
    'blocks.pde.field_processing_2d',
    'blocks.pde.heat_equation_1d',
    'blocks.pde.heat_equation_2d',
    'blocks.pde.wave_equation_1d',
    'blocks.pde.wave_equation_2d',
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
]

# Collect all scipy/numpy submodules (they have complex lazy-loading)
hidden_imports_scipy = collect_submodules('scipy')
hidden_imports_numpy = collect_submodules('numpy')

hidden_imports = (
    hidden_imports_blocks
    + hidden_imports_scipy
    + hidden_imports_numpy
    + [
        'pyqtgraph',
        'PIL',
        'tqdm',
    ]
)

# --- Data files ---
# (source_path, destination_in_bundle)
datas = [
    ('config/default_config.json', 'config'),
    ('config/logging.json', 'config'),
    ('config/block_sizes.py', 'config'),
    ('examples', 'examples'),
    ('modern_ui/icons', 'modern_ui/icons'),
]

# --- Analysis ---
a = Analysis(
    ['diablos_modern.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tests',
        'tools',
        'docs',
        'tasks',
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# --- Platform-specific packaging ---
if sys.platform == 'darwin':
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name='DiaBloS',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=False,           # No terminal window
        disable_windowed_traceback=False,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=False,
        name='DiaBloS',
    )
    app = BUNDLE(
        coll,
        name='DiaBloS.app',
        icon=None,               # Add .icns file here if you have one
        bundle_identifier='com.diablos.modern',
        info_plist={
            'CFBundleDisplayName': 'DiaBloS',
            'CFBundleShortVersionString': '2.0.0',
            'NSHighResolutionCapable': True,
            'CFBundleDocumentTypes': [
                {
                    'CFBundleTypeName': 'DiaBloS Diagram',
                    'CFBundleTypeExtensions': ['diablos'],
                    'CFBundleTypeRole': 'Editor',
                }
            ],
        },
    )
else:
    # Windows / Linux: single-folder executable
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name='DiaBloS',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=False,
        disable_windowed_traceback=False,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=True,
        name='DiaBloS',
    )
