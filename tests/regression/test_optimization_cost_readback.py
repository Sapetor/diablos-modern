"""
Regression test: the optimizer must actually see the cost the run produced.

Blocks execute against ``block.exec_params`` (SimulationEngine.execute_block
passes that dict to execute()), so a CostFunction accumulates into
exec_params["_accumulated_cost_"].  OptimizationEngine.compute_cost read the
result back off ``block.params``, where nothing had been written, so
get_final_cost saw the default 0.0.

The objective was therefore a constant 0.0 for every parameter vector: scipy
found a zero gradient immediately, reported CONVERGENCE after two evaluations,
and returned the starting point as the optimum.  Measured on
examples/optimization_basic_demo.diablos:

    before: 2 evaluations, costs {0.0},          Kp 1.0 -> 1.0
    after: 22 evaluations, costs 0.388 -> 0.313, Kp 1.0 -> 20.0

This is the same params-vs-exec_params hazard as the RK45 ``_skip_`` flag: a
'_'-prefixed key is excluded from the exec_params cache fingerprint
(compile_cache.compile_param_items), so writes to one dict never reach the
other on their own.
"""

from pathlib import Path

import pytest

EXAMPLE = (
    Path(__file__).parent.parent.parent / "examples" / "optimization_basic_demo.diablos"
)


def _load(sim_time=2.0):
    from lib.lib import DSim

    dsim = DSim()
    dsim.buttons_list = [type("B", (), {"active": False})() for _ in range(20)]
    dsim.pyqtPlotScope = lambda: None
    data = dsim.file_service.load(filepath=str(EXAMPLE))
    assert data is not None
    dsim.deserialize(data)
    dsim.sim_time = sim_time
    dsim.plot_trange = sim_time
    dsim.execution_init_time = lambda: dsim.sim_time
    return dsim


@pytest.mark.regression
class TestRuntimeParamsSelection:
    def test_prefers_the_dict_blocks_write_to(self, qapp):
        from lib.engine.optimization_engine import OptimizationEngine

        block = type("B", (), {"params": {"a": 1}, "exec_params": {"a": 2}})()
        assert OptimizationEngine.runtime_params(block) is block.exec_params

    def test_falls_back_before_the_run(self, qapp):
        """exec_params does not exist until the diagram is initialised."""
        from lib.engine.optimization_engine import OptimizationEngine

        no_attr = type("B", (), {"params": {"a": 1}})()
        assert OptimizationEngine.runtime_params(no_attr) is no_attr.params

        empty = type("B", (), {"params": {"a": 1}, "exec_params": {}})()
        assert OptimizationEngine.runtime_params(empty) is empty.params


@pytest.mark.regression
@pytest.mark.slow
class TestOptimizerSeesARealObjective:
    def test_cost_is_not_identically_zero(self, qapp):
        from lib.engine.optimization_engine import OptimizationEngine

        dsim = _load()
        engine = OptimizationEngine(dsim=dsim)

        costs = []
        original = engine.compute_cost

        def record():
            value = original()
            costs.append(value)
            return value

        engine.compute_cost = record
        result = engine.run_optimization()

        assert result["success"] is True
        assert costs, "the objective was never evaluated"
        assert any(c > 0.0 for c in costs), (
            "every objective evaluation returned 0.0 -- the cost is being read "
            "from the dict the blocks did not write to"
        )
        assert len({round(c, 9) for c in costs}) > 1, (
            "the objective did not vary with the parameters"
        )

    def test_the_parameter_actually_moves(self, qapp):
        from lib.engine.optimization_engine import OptimizationEngine

        dsim = _load()
        engine = OptimizationEngine(dsim=dsim)
        param_block = next(b for b in dsim.blocks_list if b.block_fn == "Parameter")
        start = float(param_block.params.get("value", 1.0))

        result = engine.run_optimization()

        assert result["success"] is True
        assert engine.n_evaluations > 2, (
            f"only {engine.n_evaluations} evaluations -- scipy saw a flat objective"
        )
        assert float(param_block.params["value"]) != pytest.approx(start), (
            "the optimizer returned its starting point"
        )

    def test_accumulated_cost_lands_in_exec_params(self, qapp):
        """Pins the asymmetry the bug turned on, so a future refactor sees it."""
        dsim = _load()
        assert dsim.execution_init() is True
        dsim.execution_batch()

        cost_block = next(b for b in dsim.blocks_list if b.block_fn == "CostFunction")
        assert cost_block.exec_params.get("_accumulated_cost_", 0.0) > 0.0
        assert cost_block.params is not cost_block.exec_params
