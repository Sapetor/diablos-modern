"""
Regression tests for the params / exec_params seam (lib/engine/block_params.py).

Two shipped bugs had the same shape, one per direction of the lossy copy
between a block's configured ``params`` and its runtime ``exec_params``:

  * RK45's ``_skip_`` -- written to ``params``, read from ``exec_params``, so
    the flag never reached sinks and every RK4 sub-step was recorded as a
    sample (21 scope samples against a 6-entry timeline).
  * The optimizer's ``_accumulated_cost_`` -- written to ``exec_params`` by the
    CostFunction block, read back from ``params``, so the objective was a
    constant 0.0 and scipy "converged" in two evaluations at the start point.

The read direction is closed by routing every reader through
``runtime_params``; the write direction by ``PUSH_DOWN_KEYS`` being refreshed
even on the ``_resolve_block_params`` cache-hit path.  These tests pin both,
plus the one key that must *never* join the push-down set.
"""

import pytest

from lib.engine.block_params import (
    PUSH_DOWN_KEYS,
    push_down_internal_params,
    runtime_params,
)


def _block(params=None, exec_params=None, step=0.01):
    """A stand-in with just the attributes the seam touches."""

    class _B:
        def __init__(self):
            self.params = dict(params or {})
            if exec_params is not None:
                self.exec_params = dict(exec_params)
            self.block_fn = "Scale"

        def execution_step(self, dt):
            return step

    return _B()


@pytest.mark.regression
class TestRuntimeParamsPicksTheDictTheWriterUsed:
    def test_prefers_exec_params(self):
        block = _block({"a": 1}, {"a": 2})
        assert runtime_params(block) is block.exec_params

    def test_falls_back_when_exec_params_is_absent(self):
        """exec_params does not exist until the diagram is initialised."""
        block = _block({"a": 1})
        assert runtime_params(block) is block.params

    def test_falls_back_when_exec_params_is_empty(self):
        block = _block({"a": 1}, {})
        assert runtime_params(block) is block.params

    def test_never_returns_none(self):
        """Callers index the result directly; a None would be a crash."""
        assert runtime_params(object()) == {}


@pytest.mark.regression
class TestEveryReaderSharesOneImplementation:
    """The idiom had independently evolved in three places before this."""

    def test_memory_blocks_delegates(self):
        from lib.engine import memory_blocks

        block = _block({"a": 1}, {"a": 2})
        assert memory_blocks._params_source(block) is block.exec_params

    def test_optimization_engine_delegates(self):
        from lib.engine.optimization_engine import OptimizationEngine

        block = _block({"a": 1}, {"a": 2})
        assert OptimizationEngine.runtime_params(block) is block.exec_params

    def test_resim_uses_the_helper(self):
        """harvest_scope_signals must read the buffer the run actually filled."""
        import numpy as np

        from lib.analysis.resim import harvest_scope_signals

        class _Scope:
            name = "scope0"
            block_fn = "Scope"
            params = {"vector": [0.0, 0.0, 0.0]}
            exec_params = {"vector": [1.0, 2.0, 3.0], "vec_dim": 1}

        class _Engine:
            active_blocks_list = [_Scope()]

        class _Dsim:
            timeline = np.array([0.0, 0.1, 0.2])
            engine = _Engine()
            blocks_list = _Engine.active_blocks_list

        harvested = harvest_scope_signals(_Dsim())
        assert list(harvested["signals"]["scope0"]) == [1.0, 2.0, 3.0]

    def test_no_reader_reintroduces_its_own_copy(self):
        """Guards against a fourth hand-rolled fallback appearing.

        Only flags the *fallback* idiom (exec_params or params).  Reading
        exec_params alone is a deliberate, different choice -- the compile
        cache fingerprints both dicts separately, and the post-run summary in
        simulation_controller wants runtime state only.
        """
        import re
        from pathlib import Path

        root = Path(__file__).parent.parent.parent
        idiom = re.compile(r'getattr\(\s*\w+\s*,\s*["\']exec_params["\'].*\bor\b.*params')
        offenders = []
        for path in list((root / "lib").rglob("*.py")) + list((root / "modern_ui").rglob("*.py")):
            if path.name == "block_params.py":
                continue  # the one legitimate implementation
            for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if idiom.search(line):
                    offenders.append(f"{path.relative_to(root)}:{n}")
        assert not offenders, (
            "hand-rolled exec_params fallback found; use "
            "lib.engine.block_params.runtime_params instead: " + ", ".join(offenders)
        )


@pytest.mark.regression
class TestPushDownReachesTheBlockOnACacheHit:
    def test_refreshes_the_push_down_keys(self):
        block = _block({"_skip_": True}, {"_skip_": False})
        push_down_internal_params(block)
        assert block.exec_params["_skip_"] is True

    def test_leaves_block_owned_state_alone(self):
        """Only the narrow set is copied; runtime state must survive."""
        block = _block({"mem": 0.0, "_int": 0.0}, {"mem": 7.0, "_int": 3.0})
        push_down_internal_params(block)
        assert block.exec_params["mem"] == 7.0
        assert block.exec_params["_int"] == 3.0

    def test_is_a_noop_before_initialisation(self):
        block = _block({"_skip_": True})
        push_down_internal_params(block)  # must not raise
        assert not hasattr(block, "exec_params")

    def test_init_start_is_not_pushed_down(self):
        """The clobber guard, and the reason the copy cannot be blanket.

        reset_memblocks sets _init_start_ True in *both* dicts; the block
        clears its own copy to False once initialised.  Pushing the stale True
        down again would re-initialise the block on every single step, so this
        key must stay out of the set no matter how convenient it looks.
        """
        assert "_init_start_" not in PUSH_DOWN_KEYS

        block = _block({"_init_start_": True}, {"_init_start_": False})
        push_down_internal_params(block)
        assert block.exec_params["_init_start_"] is False


@pytest.mark.regression
class TestResolveHonoursThePushDownOnItsFastPath:
    def test_mid_run_write_reaches_exec_params(self, qapp):
        """The exact shape of the _skip_ bug, at the seam that dropped it."""
        from lib.engine.simulation_engine import SimulationEngine

        engine = SimulationEngine(model=None)
        block = _block({"gain": 2.0}, None)

        engine._resolve_block_params(block, 0.01)
        assert "_skip_" not in block.exec_params

        # A mid-run write to params: '_' keys are excluded from the cache
        # fingerprint, so this must NOT invalidate the cache -- that is exactly
        # what made the old code drop it.
        block.params["_skip_"] = True
        engine._resolve_block_params(block, 0.01)
        assert block.exec_params["_skip_"] is True

        block.params["_skip_"] = False
        engine._resolve_block_params(block, 0.01)
        assert block.exec_params["_skip_"] is False

    def test_the_cache_hit_is_real(self, qapp):
        """If the fingerprint changed, the push-down would be untested."""
        from lib.engine.simulation_engine import SimulationEngine
        from lib.engine.compile_cache import source_params_fingerprint

        block = _block({"gain": 2.0}, None)
        before = source_params_fingerprint(block.params)
        block.params["_skip_"] = True
        assert source_params_fingerprint(block.params) == before

        engine = SimulationEngine(model=None)
        engine._resolve_block_params(block, 0.01)
        resolved = block.exec_params
        engine._resolve_block_params(block, 0.01)
        assert block.exec_params is resolved, "expected the cache-hit path"
