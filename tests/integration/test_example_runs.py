"""
Integration test: every examples/*.diablos must load and pass engine init.

This is the canary test that catches regressions in the
    load -> deserialize -> resolve_params -> engine.initialize_execution
pipeline without opening any GUI dialogs.

Commit 3e58bfb (safe_eval rollout) broke this pipeline for PDE examples
whose block params contain bare function-name strings like "sin"/"cos";
safe_expr(allow_numpy=True) silently coerced them to numpy ufuncs.
"""

from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"
EXAMPLE_FILES = sorted(EXAMPLES_DIR.glob("*.diablos"))

_KNOWN_BROKEN: set[str] = set()


@pytest.mark.integration
@pytest.mark.parametrize("example_file", EXAMPLE_FILES, ids=lambda f: f.name)
def test_example_loads_and_inits(example_file, qapp):
    """Load each example and run the engine init pass — no exceptions allowed."""
    if example_file.name in _KNOWN_BROKEN:
        pytest.xfail(
            f"{example_file.name} has a pre-existing diagram integrity issue "
            "unrelated to the safe_expr regression."
        )

    from lib.lib import DSim
    from lib.workspace import WorkspaceManager

    # Reset WorkspaceManager singleton so tests don't share state.
    prev_instance = WorkspaceManager._instance
    WorkspaceManager._instance = None

    try:
        dsim = DSim()

        # Load the file directly (bypasses QFileDialog).
        data = dsim.file_service.load(filepath=str(example_file))
        assert data is not None, f"file_service.load returned None for {example_file.name}"

        # Deserialize into the model (populates blocks_list, line_list, sim params).
        dsim.deserialize(data)

        # Mirror the resolve_params loop from DSim.execution_init (no dialog needed).
        workspace_manager = WorkspaceManager()

        def resolve_recursive(blocks):
            for block in blocks:
                block.exec_params = workspace_manager.resolve_params(block.params)
                block.exec_params.update(
                    {k: v for k, v in block.params.items() if k.startswith('_')}
                )
                block.exec_params['dtime'] = dsim.sim_dt
                dsim.engine.set_block_type(block)
                if getattr(block, 'block_type', '') == 'Subsystem':
                    resolve_recursive(block.sub_blocks)

        root_blocks, root_lines = dsim.get_root_context()
        resolve_recursive(root_blocks)

        # Engine init: resolves hierarchy, detects algebraic loops, validates connections.
        dsim.engine.update_sim_params(dsim.sim_time, dsim.sim_dt)
        ok = dsim.engine.initialize_execution(root_blocks, root_lines)

        assert ok, (
            f"{example_file.name}: engine.initialize_execution failed — "
            f"error_msg={dsim.error_msg!r}"
        )
        assert not dsim.error_msg, (
            f"{example_file.name}: init succeeded but error_msg set: {dsim.error_msg!r}"
        )

    finally:
        WorkspaceManager._instance = prev_instance
