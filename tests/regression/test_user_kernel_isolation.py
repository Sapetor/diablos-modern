"""Two projects' same-named user blocks keep their own compiled kernels.

A user block registers its compiled kernel with ``@kernel("<CanonicalName>")``
(docs/BLOCK_API.md, section 7). The registry used to be one process-global
dict keyed by that name only, so once two trusted projects each defining a
``ReviewTripler`` block were loaded in the same process, the *last* loaded
kernel served both: project A (factor 3) ran project B's kernel (factor 7) on
the compiled path while its interpreter path stayed correct. User kernels and
event builders are now bound to the module that defines the block class.

Each project's diagram is ``Constant(c) -> ReviewTripler -> Integrator -> Scope``
over 1 s, so the scope ends at ``factor * c``.
"""

import numpy as np
import pytest

import lib.user_blocks as ub

pytestmark = pytest.mark.regression

BLOCK_SOURCE = """\
import numpy as np

from blocks.base_block import BaseBlock
from lib.engine.compiler_kernels import EventSpec, events, kernel, signal_scalar

FACTOR = {factor}


class ReviewTriplerBlock(BaseBlock):
    @property
    def block_name(self):
        return "ReviewTripler"

    @property
    def category(self):
        return "Math"

    @property
    def params(self):
        return {{}}

    @property
    def inputs(self):
        return [{{"name": "in", "type": "any"}}]

    @property
    def outputs(self):
        return [{{"name": "out", "type": "any"}}]

    def execute(self, time, inputs, params, **kwargs):
        return {{0: FACTOR * np.atleast_1d(inputs.get(0, 0.0))}}


@kernel("Reviewtripler")
def build_review_tripler(ctx):
    b_name = ctx.b_name
    src = ctx.input_sources[0]

    def exec_review_tripler(t, y, dy_vec, signals):
        signals[b_name] = FACTOR * signals.get(src, 0.0)

    return exec_review_tripler


@events("Reviewtripler")
def events_review_tripler(ctx):
    src = ctx.input_sources[0]
    return [
        EventSpec(
            block=ctx.b_name,
            label="factor_{factor}",
            func=lambda t, y, signals: signal_scalar(signals, src) - 1e9,
        )
    ]
"""

FACTORS = {"a": 3.0, "b": 7.0}
ORDERS = pytest.mark.parametrize("order", [("a", "b"), ("b", "a")], ids=["a_then_b", "b_then_a"])


def write_project(root, factor):
    """A trusted project: ``blocks/review_tripler.py`` plus ``model.diablos``."""
    from lib.diagram_builder import DiagramBuilder

    blocks = root / "blocks"
    blocks.mkdir(parents=True)
    # The same file name in both projects, so the modules also share their
    # ``diablos_user_blocks.<stem>`` name in sys.modules.
    (blocks / "review_tripler.py").write_text(BLOCK_SOURCE.format(factor=factor), encoding="utf-8")
    builder = DiagramBuilder(sim_time=1.0, sim_dt=0.01)
    c = builder.add_block("Constant", 50, 100, name="c", params={"value": 2.0})
    t = builder.add_block("ReviewTripler", 200, 100, name="t")
    i = builder.add_block("Integrator", 350, 100, name="i", params={"init_conds": 0.0})
    s = builder.add_block("Scope", 500, 100, name="s", in_ports=1, out_ports=0)
    builder.connect(c, 0, t, 0)
    builder.connect(t, 0, i, 0)
    builder.connect(i, 0, s, 0)
    path = root / "model.diablos"
    builder.save(str(path))
    ub.trust_project_blocks_for_session(str(blocks))
    return str(path)


@pytest.fixture
def projects(tmp_path, monkeypatch):
    """Two trusted projects with identically named, different ReviewTripler blocks."""
    import lib.app_paths

    monkeypatch.delenv(ub.BLOCKS_ENV_VAR, raising=False)
    home = tmp_path / "userdata"
    home.mkdir()
    monkeypatch.setattr(ub, "get_user_data_dir", lambda: str(home))
    monkeypatch.setattr(lib.app_paths, "get_user_data_dir", lambda: str(home))
    monkeypatch.setattr(ub, "_SESSION_TRUSTED", set(), raising=False)
    ub.purge_user_modules()
    paths = {key: write_project(tmp_path / key, factor) for key, factor in FACTORS.items()}
    yield paths
    ub.purge_user_modules()


def load_in_order(paths, order):
    from lib import api

    return {key: api.load(paths[key]) for key in order}


def final_value(diagram, solver):
    result = diagram.run(t=1.0, dt=0.01, solver=solver)
    expected_engine = "compiled" if solver == "compiled" else "interpreter"
    assert result.engine == expected_engine, result.fallback_reason
    (trace,) = result.signals.values()
    return float(np.asarray(trace, dtype=float).ravel()[-1])


def scope_final(dsim):
    scope = next(b for b in dsim.engine.active_blocks_list if b.block_fn == "Scope")
    trace = np.asarray(getattr(scope, "exec_params", scope.params)["vector"], dtype=float)
    return float(trace.ravel()[-1])


@pytest.mark.qt
@ORDERS
@pytest.mark.parametrize("solver", ["compiled", "interpreter"])
def test_each_project_keeps_its_own_block(qapp, projects, order, solver):
    diagrams = load_in_order(projects, order)
    for key in order:
        got = final_value(diagrams[key], solver)
        assert np.isclose(got, 2.0 * FACTORS[key], rtol=1e-3), (key, got)


@pytest.mark.qt
@ORDERS
def test_recompile_after_param_change_keeps_own_kernel(qapp, projects, order):
    diagrams = load_in_order(projects, order)
    for key in order:
        diagram = diagrams[key]
        assert np.isclose(final_value(diagram, "compiled"), 2.0 * FACTORS[key], rtol=1e-3)
        misses = diagram.dsim.engine.compile_cache_misses
        diagram.set_param("constant0", "value", 5.0)
        assert np.isclose(final_value(diagram, "compiled"), 5.0 * FACTORS[key], rtol=1e-3)
        assert diagram.dsim.engine.compile_cache_misses == misses + 1, "did not recompile"
        # A repeat run is a cache hit and still answers with this project's kernel.
        assert np.isclose(final_value(diagram, "compiled"), 5.0 * FACTORS[key], rtol=1e-3)


@pytest.mark.qt
@ORDERS
def test_analysis_clone_keeps_own_kernel(qapp, projects, order):
    diagrams = load_in_order(projects, order)
    for key in order:
        clone = diagrams[key].dsim.clone_for_analysis()
        clone.use_fast_solver = True
        ok, err = clone.run_tuning_simulation(1.0, 0.01)
        assert ok, err
        assert clone.engine.get_compile_fallback_reason() is None
        assert np.isclose(scope_final(clone), 2.0 * FACTORS[key], rtol=1e-3), key


@pytest.mark.qt
def test_events_are_bound_to_the_owning_block(qapp, projects):
    from lib.engine.compiler_kernels import BuildContext, build_events

    diagrams = load_in_order(projects, ("a", "b"))
    for key, diagram in diagrams.items():
        block = next(b for b in diagram.dsim.blocks_list if b.block_fn == "ReviewTripler")
        ctx = BuildContext(
            block=block,
            b_name=block.name,
            fn="Reviewtripler",
            params={},
            input_sources=["constant0"],
            deps={},
            state_map={},
            block_matrices={},
        )
        (spec,) = build_events(ctx)
        assert spec.label == "factor_{}".format(FACTORS[key])


@pytest.mark.qt
def test_user_kernel_is_not_reachable_by_name_alone(qapp, projects):
    """Only the owning block class reaches a user kernel; a bare name does not."""
    from lib.engine.compiler_kernels import get_event_builder, get_kernel_builder

    load_in_order(projects, ("a",))
    assert get_kernel_builder("Reviewtripler") is None
    assert get_event_builder("Reviewtripler") is None


@pytest.mark.qt
@pytest.mark.slow
@pytest.mark.parametrize("workers", [1, 2])
def test_experiment_workers_keep_own_kernel(qapp, projects, workers):
    diagrams = load_in_order(projects, ("a", "b"))
    for key, diagram in diagrams.items():
        diagram.dsim.use_fast_solver = True
        ensemble = diagram.monte_carlo(2, t=1.0, dt=0.01, workers=workers)
        assert ensemble["n_ok"] == 2
        (stats,) = ensemble["signals"].values()
        final = np.asarray(stats["mean"], dtype=float).ravel()[-1]
        assert np.isclose(final, 2.0 * FACTORS[key], rtol=1e-3), (key, final)


@pytest.mark.qt
def test_lookup_by_block_instance_and_class(qapp, projects):
    from lib.engine.compiler_kernels import get_kernel_builder

    diagrams = load_in_order(projects, ("a", "b"))
    builders = set()
    for diagram in diagrams.values():
        block = next(b for b in diagram.dsim.blocks_list if b.block_fn == "ReviewTripler")
        instance = block.block_instance
        builder = get_kernel_builder("Reviewtripler", block)
        assert builder is not None
        assert get_kernel_builder("Reviewtripler", instance) is builder
        assert get_kernel_builder("Reviewtripler", type(instance)) is builder
        builders.add(builder)
    assert len(builders) == 2
