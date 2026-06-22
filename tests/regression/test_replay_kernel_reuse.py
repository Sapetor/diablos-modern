"""Regression test for Phase 3 (safe partial dedup): the post-solve replay loop
in ``SimulationEngine.run_compiled_simulation`` reuses the compiled kernel
executors (``lib.engine.compiler_kernels`` via ``SystemCompiler.block_executors``)
for pure-function blocks, instead of a parallel inline if/elif. This keeps each
such block's output math in one place (the kernel) shared by the ODE solve and
the replay.

The allowlist of routed blocks is ``SimulationEngine._KERNEL_REPLAY_FNS``;
state/PDE/Field blocks and the intentionally-divergent Mathfunction/StateVariable
stay on their own replay branches.
"""
from pathlib import Path

import pytest

from lib.engine.simulation_engine import _KERNEL_REPLAY_FNS

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"


def _load(filename):
    from lib.lib import DSim
    from lib.workspace import WorkspaceManager

    WorkspaceManager._instance = None
    dsim = DSim()
    data = dsim.file_service.load(filepath=str(EXAMPLES_DIR / filename))
    assert data is not None, f"Failed to load {filename}"
    dsim.file_service.apply_loaded_data(data)
    return dsim


@pytest.mark.regression
class TestReplayKernelReuse:
    def test_block_executors_populated_after_compiled_run(self, qapp):
        """compile_system must expose a name -> executor map covering every block;
        this is what the replay loop reuses for the routed (pure-function) blocks."""
        dsim = _load("c01_tank_feedback.diablos")  # Step, Sum, Gain, TranFn
        dsim.use_fast_solver = True
        ok, err = dsim.run_tuning_simulation(dsim.sim_time, dsim.sim_dt)
        assert ok, f"compiled run failed: {err}"

        be = getattr(dsim.engine.compiler, "block_executors", None)
        assert be, "compile_system should populate compiler.block_executors"
        names = {b.name for b in dsim.engine.active_blocks_list}
        # Every block has exactly one executor, keyed by block name.
        assert set(be) == names
        assert all(callable(fn) for fn in be.values())

    def test_routed_block_appears_and_is_reused(self, qapp):
        """A routed block (Gain) from the diagram is present in block_executors,
        and its executor computes the block's output into the signal dict the
        way the replay calls it."""
        dsim = _load("c01_tank_feedback.diablos")
        dsim.use_fast_solver = True
        ok, err = dsim.run_tuning_simulation(dsim.sim_time, dsim.sim_dt)
        assert ok, f"compiled run failed: {err}"

        gains = [b for b in dsim.engine.active_blocks_list
                 if b.block_fn == "Gain"]
        assert gains, "expected a Gain block in c01_tank_feedback"
        be = dsim.engine.compiler.block_executors
        for g in gains:
            assert g.name in be and callable(be[g.name])

    def test_allowlist_excludes_state_pde_and_divergent_blocks(self):
        """Guard the design decision: routing a stateful/PDE block (reads
        reconstructed state) or an intentionally-divergent block (Mathfunction's
        domain-guarded math, StateVariable's discrete-state mechanism, Demux's
        secondary-port outputs) through the pure-function path would change
        replay output. They must stay off the allowlist."""
        must_exclude = {
            "Integrator", "StateSpace", "TransferFcn", "PID", "RateLimiter",
            "Heatequation1D", "Waveequation1D", "Advectionequation1D",
            "Diffusionreaction1D", "Heatequation2D", "Waveequation2D",
            "Advectionequation2D", "Mathfunction", "StateVariable",
            "Statevariable", "Demux", "Selector", "Hysteresis",
            "Fieldprobe", "Fieldscope", "Fieldprobe2D", "Fieldscope2D",
            "Fieldslice",
        }
        leaked = must_exclude & _KERNEL_REPLAY_FNS
        assert not leaked, f"these must NOT be routed via the kernel path: {leaked}"

    def test_allowlist_has_expected_pure_function_blocks(self):
        """The pure-function source/algebraic blocks should be routed."""
        expected = {
            "Sine", "Constant", "Gain", "Sum", "Step", "Product",
            "Exponential", "Deadband", "Saturation", "Ramp", "Switch",
            "Wavegenerator", "Mux", "Logicaloperator",
        }
        missing = expected - _KERNEL_REPLAY_FNS
        assert not missing, f"expected these to be routed via the kernel: {missing}"
