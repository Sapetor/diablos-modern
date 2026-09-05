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
import os
import platform
import re
from PyInstaller.utils.hooks import collect_submodules

block_cipher = None
ARCH = platform.machine()  # 'arm64' or 'x86_64'

PROJECT_ROOT = os.path.dirname(os.path.abspath(SPEC))


def _project_version(default='1.0.0'):
    """Read `[project] version` out of pyproject.toml (the single source of truth).

    Parsed with a regex rather than tomllib so the spec works on the 3.9 build
    env too (tomllib is 3.11+). tools/build.sh reads the same key, so the DMG
    name and CFBundleShortVersionString can never drift apart.
    """
    path = os.path.join(PROJECT_ROOT, 'pyproject.toml')
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            in_project = False
            for line in fh:
                stripped = line.strip()
                if stripped.startswith('['):
                    in_project = stripped == '[project]'
                    continue
                if in_project:
                    match = re.match(r'^version\s*=\s*["\'](.+?)["\']', stripped)
                    if match:
                        return match.group(1)
    except OSError:
        pass
    return default


VERSION = _project_version()

# Bundle the version as a plain-text data file. A frozen bundle ships no
# `.dist-info`, so `modern_ui/__init__.py` cannot read the version back through
# `importlib.metadata`; it reads `sys._MEIPASS/_version.txt` instead. Written
# into build/ (gitignored) so the source tree stays clean.
_BUILD_DIR = os.path.join(PROJECT_ROOT, 'build')
os.makedirs(_BUILD_DIR, exist_ok=True)
VERSION_FILE = os.path.join(_BUILD_DIR, '_version.txt')
with open(VERSION_FILE, 'w', encoding='utf-8') as _fh:
    _fh.write(VERSION + '\n')

# --- Hidden imports ---
# Blocks are loaded dynamically via importlib (lib/block_loader.py),
# so PyInstaller can't discover them through static analysis.
# Auto-scanned from blocks/ directory at build time.
sys.path.insert(0, PROJECT_ROOT)
from tools.sync_block_registry import scan_block_modules
hidden_imports_blocks = scan_block_modules()

# Collect scipy/numpy submodules, filtering out tests to save space
hidden_imports_scipy = [m for m in collect_submodules('scipy')
                        if '.tests' not in m and '.testing' not in m]
hidden_imports_numpy = [m for m in collect_submodules('numpy')
                        if '.tests' not in m and '.testing' not in m]

# SymPy powers the optional symbolic features (LaTeX/MathML export,
# `lib/engine/symbolic_engine.py`, the per-block `symbolic_execute`). It is an
# optional dependency (`pip install .[symbolic]`), NOT part of requirements.txt,
# so it is bundled only when the build environment has it -- see
# docs/building.md. It used to be listed in `excludes` below, which meant even a
# build env with sympy installed produced a bundle where LaTeX export silently
# degraded.
try:
    hidden_imports_sympy = [m for m in collect_submodules('sympy')
                            if '.tests' not in m and '.testing' not in m]
except Exception:
    hidden_imports_sympy = []

hidden_imports = (
    hidden_imports_blocks
    + hidden_imports_scipy
    + hidden_imports_numpy
    + hidden_imports_sympy
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
    (VERSION_FILE, '.'),
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
        # Project dirs
        'tests', 'tools', 'docs', 'tasks',
        # Heavy packages not used by DiaBloS
        'torch', 'torchvision', 'torchaudio',
        'pandas', 'pyarrow',
        'bokeh', 'selenium', 'playwright',
        'notebook', 'nbformat', 'nbconvert', 'nbclient',
        'sphinx', 'docutils', 'alabaster',
        'IPython', 'ipykernel', 'ipywidgets', 'jupyter',
        'jedi', 'parso',
        'distributed', 'dask',
        'h5py', 'tables',
        'numba', 'llvmlite',
        'sqlalchemy',
        'openpyxl',
        'lxml',
        'cryptography',
        'zmq', 'pyzmq',
        'pytest', 'py', '_pytest',
        'black', 'blib2to3',
        'astroid', 'pylint', 'isort',
        'cloudpickle', 'fsspec',
        'gevent',
        'babel',
        'pygments',
        'boto', 'botocore', 's3transfer',
        'sklearn', 'scikit-learn',
        'cv2', 'opencv',
        'tkinter', '_tkinter',
        'curses',
        'pygame',
        'tornado',
        'yaml', 'pyyaml',
        'debugpy',
        'setuptools', 'pkg_resources',
        'xmlrpc',
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
        name=f'DiaBloS-{ARCH}.app',
        icon=None,               # Add .icns file here if you have one
        bundle_identifier=f'com.diablos.modern.{ARCH}',
        info_plist={
            'CFBundleDisplayName': f'DiaBloS ({ARCH})',
            'CFBundleShortVersionString': VERSION,
            'CFBundleVersion': VERSION,
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
