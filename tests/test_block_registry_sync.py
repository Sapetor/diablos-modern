"""Guard against dual block-registry drift.

Blocks load two ways (see ``lib/block_loader.py``):

* **dev** — a live filesystem scan of ``blocks/`` (``load_blocks``);
* **frozen / PyInstaller** — the hardcoded ``_BLOCK_MODULES`` list.

``tools/sync_block_registry.py`` regenerates ``_BLOCK_MODULES`` from the same
scan. If a developer adds (or removes) a block file and forgets to run the
sync, everything works in dev but the block is *silently* missing from the
packaged app. This test recomputes the expected module list exactly the way
the sync script does and asserts it matches the checked-in ``_BLOCK_MODULES``.
"""

import importlib.util
import os

import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_scan_block_modules():
    """Import ``scan_block_modules`` from ``tools/sync_block_registry.py``.

    Source of truth for the scan logic — we deliberately reuse the sync
    tool's function rather than reimplement it, so the test and the tool can
    never disagree about which files count as blocks.
    """
    tool_path = os.path.join(_PROJECT_ROOT, "tools", "sync_block_registry.py")
    spec = importlib.util.spec_from_file_location("sync_block_registry", tool_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.scan_block_modules


@pytest.mark.unit
def test_block_modules_registry_is_in_sync():
    from lib.block_loader import _BLOCK_MODULES

    scan_block_modules = _load_scan_block_modules()
    expected = set(scan_block_modules())
    actual = set(_BLOCK_MODULES)

    missing = sorted(expected - actual)  # block files not yet in the registry
    stale = sorted(actual - expected)  # registry entries with no block file

    assert expected == actual, (
        "lib/block_loader._BLOCK_MODULES is out of sync with the blocks/ "
        "directory — the frozen/PyInstaller build would ship the wrong set "
        "of blocks.\n"
        f"  missing from _BLOCK_MODULES (present on disk): {missing}\n"
        f"  stale in _BLOCK_MODULES (no matching file):    {stale}\n"
        "Run:  python tools/sync_block_registry.py"
    )
