#!/usr/bin/env python3
"""
Scan the blocks/ directory and update _BLOCK_MODULES in lib/block_loader.py.

Run this before building with PyInstaller so the frozen app includes all blocks.
Called automatically by tools/build.sh, and reused by diablos.spec (which imports
``scan_block_modules`` directly for its hidden-imports list) and by
``tests/test_block_registry_sync.py``.

Usage:
    python tools/sync_block_registry.py            # rewrite lib/block_loader.py
    python tools/sync_block_registry.py --check    # exit 1 if it would change
"""

import argparse
import os
import re
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLOCKS_DIR = os.path.join(PROJECT_ROOT, "blocks")
BLOCK_LOADER = os.path.join(PROJECT_ROOT, "lib", "block_loader.py")

#: Modules under blocks/ that define no block class. They are shared helpers
#: imported *by* blocks, so listing them in ``_BLOCK_MODULES`` only made the
#: frozen loader import them twice and log confusing "no block found" noise.
#: Names are relative to ``blocks/`` and use "/" for subdirectories.
EXCLUDED_MODULES = frozenset(
    {
        "base_block",  # the abstract base class every block inherits from
        "input_helpers",  # shared np.atleast_1d(inputs.get(...)) helpers
        "param_templates",  # shared parameter-spec fragments
        "statespace_base",  # shared A/B/C/D plumbing (also matches _BASE_SUFFIX)
        "pde/_compat",  # numpy/scipy compatibility shims (also matches "_" rule)
    }
)

#: Anything named ``*_base`` is a shared base class by convention, and anything
#: starting with "_" is private (``__init__``, ``_compat``, ...). Both are
#: skipped without needing an entry in ``EXCLUDED_MODULES``, so a new helper
#: dropped into blocks/ does not silently register itself as a block.
_BASE_SUFFIX = "_base"


def _is_block_module(rel_name):
    """Return True if ``rel_name`` (e.g. "gain" or "pde/heat_equation_1d") is a block."""
    if rel_name in EXCLUDED_MODULES:
        return False
    stem = rel_name.rsplit("/", 1)[-1]
    if stem.startswith("_"):
        return False
    if stem.endswith(_BASE_SUFFIX):
        return False
    return True


def scan_block_modules(blocks_dir=BLOCKS_DIR):
    """Scan blocks/ directory and return sorted list of dotted module names.

    Helper modules (see ``EXCLUDED_MODULES`` / ``_is_block_module``) are skipped.
    """
    modules = []

    for filename in sorted(os.listdir(blocks_dir)):
        if filename.endswith(".py") and _is_block_module(filename[:-3]):
            modules.append(f"blocks.{filename[:-3]}")

    for subdir in sorted(os.listdir(blocks_dir)):
        subdir_path = os.path.join(blocks_dir, subdir)
        if os.path.isdir(subdir_path) and not subdir.startswith("__"):
            init_file = os.path.join(subdir_path, "__init__.py")
            if os.path.exists(init_file):
                for filename in sorted(os.listdir(subdir_path)):
                    if filename.endswith(".py") and _is_block_module(f"{subdir}/{filename[:-3]}"):
                        modules.append(f"blocks.{subdir}.{filename[:-3]}")

    return modules


def render_block_loader(content, modules):
    """Return ``content`` with its ``_BLOCK_MODULES`` list replaced by ``modules``."""
    # Double quotes to match `ruff format`, which is a CI gate: emitting single
    # quotes here left block_loader.py dirty after every build with a diff that
    # was pure quote churn, and committing it failed `ruff format --check`.
    items = ",\n".join(f'    "{m}"' for m in modules)
    new_list = f"_BLOCK_MODULES = [\n{items},\n]"

    return re.sub(
        r"_BLOCK_MODULES\s*=\s*\[.*?\]",
        new_list,
        content,
        flags=re.DOTALL,
    )


def update_block_loader(modules, path=BLOCK_LOADER):
    """Rewrite _BLOCK_MODULES list in block_loader.py. Returns True if it changed."""
    with open(path, "r") as f:
        content = f.read()

    updated = render_block_loader(content, modules)

    if updated == content:
        print("block_loader.py: already up to date")
        return False

    with open(path, "w") as f:
        f.write(updated)
    print(f"block_loader.py: updated _BLOCK_MODULES ({len(modules)} modules)")
    return True


def check_block_loader(modules, path=BLOCK_LOADER):
    """Return True if ``path`` already matches ``modules`` (nothing to write)."""
    with open(path, "r") as f:
        content = f.read()
    return render_block_loader(content, modules) == content


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit 1 if lib/block_loader.py is out of date",
    )
    args = parser.parse_args(argv)

    modules = scan_block_modules()

    if args.check:
        if check_block_loader(modules):
            print(f"block_loader.py: up to date ({len(modules)} modules)")
            return 0
        print(
            "block_loader.py: OUT OF DATE — the frozen/PyInstaller build would "
            "ship the wrong set of blocks.\n"
            "Run:  python tools/sync_block_registry.py",
            file=sys.stderr,
        )
        return 1

    update_block_loader(modules)
    return 0


if __name__ == "__main__":
    sys.exit(main())
