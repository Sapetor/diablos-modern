"""The Function (expression) block runs on the compiled fast path
(lib/engine/compiler_kernels/function.py) with the interpreter's semantics."""

import numpy as np
import pytest

from blocks.function import FunctionBlock
from lib.engine.compiler_kernels import BuildContext, get_kernel_builder
from lib.safe_eval import SafeEvalError


def _kernel(expr, input_sources, n_inputs=None):
    params = {"expression": expr}
    if n_inputs is not None:
        params["_inputs_"] = n_inputs
    ctx = BuildContext(
        block=None,
        b_name="f",
        fn="Function",
        params=params,
        input_sources=input_sources,
        deps={},
        state_map={},
        block_matrices={},
    )
    return get_kernel_builder("Function")(ctx)


@pytest.mark.unit
class TestKernelMatchesExecute:
    @pytest.mark.parametrize(
        "expr, inputs",
        [
            ("u[0]**2 + 0.5*t", {0: 1.5}),
            ("sin(u[0]) + np.tanh(u1)", {0: 0.3}),
            ("u1*u2 - u[1]/(1 + u[0]**2)", {0: 2.0, 1: -0.7}),
            ("np.arctan2(u[1], u[0]) + pi", {0: -1.0, 1: 0.5}),
            ("u[0] if u[0] > 0 else -2*u[0]", {0: -0.25}),
            ("u[0][0]", {0: 4.0}),  # a scalar input is still indexable, as in execute()
            ("[u[0][0] + u[0][2], u[0][1]*t]", {0: [1.0, 2.0, 3.0]}),
            ("u[0] * u[1]", {0: [1.0, 2.0], 1: 3.0}),
            ("sum(u[0]) + max(u[0])", {0: [1.0, -4.0, 2.5]}),
            ("u[1] + 1", {0: 1.0}),  # port 1 unconnected reads 0
        ],
    )
    def test_same_output(self, expr, inputs):
        t = 0.8
        n = 2 if expr == "u[1] + 1" else max(inputs) + 1
        ref = FunctionBlock().execute(
            t,
            {i: np.atleast_1d(np.asarray(v, dtype=float)) for i, v in inputs.items()},
            {"expression": expr, "_inputs_": n},
        )
        assert "E" not in ref, ref
        sources = ["s{}".format(i) if i in inputs else None for i in range(n)]
        signals = {"s{}".format(i): v for i, v in inputs.items()}
        # Compiled signals are floats for scalars and arrays for vectors.
        signals = {
            k: (np.asarray(v, dtype=float) if isinstance(v, list) else v)
            for k, v in signals.items()
        }
        _kernel(expr, sources, n)(t, None, None, signals)
        out = signals["f"]
        np.testing.assert_allclose(np.atleast_1d(out), ref[0], rtol=0, atol=0)
        if ref[0].size == 1:
            assert isinstance(out, float)

    def test_port_count_comes_from_the_block(self):
        """u has one entry per port even when trailing ports are unconnected."""
        signals = {"a": 2.0}
        _kernel("len(u)", ["a"], n_inputs=3)(0.0, None, None, signals)
        assert signals["f"] == 3.0

    def test_forbidden_construct_fails_at_build_time(self):
        with pytest.raises(SafeEvalError):
            _kernel("__import__('os').getcwd()", ["a"])

    def test_runtime_error_raises_instead_of_outputting_zero(self):
        run = _kernel("u[3]", ["a"])
        with pytest.raises(SafeEvalError, match="Function block 'f'"):
            run(0.0, None, None, {"a": 1.0})


def _diagram(tmp_path, expr, sources, integrate=False, n_inputs=None):
    """sources -> Function(expr) -> Scope; with ``integrate`` the Function also
    closes a loop through an Integrator: x' = f(x, sources...), Scope on x."""
    from lib.diagram_builder import DiagramBuilder

    b = DiagramBuilder(sim_time=2.0, sim_dt=0.01)
    n = n_inputs or (len(sources) + (1 if integrate else 0))
    f = b.add_block("Function", 0, 0, name="f", params={"expression": expr}, in_ports=n)
    sc = b.add_block("Scope", 0, 0, name="sc", params={"labels": "y"}, in_ports=1, out_ports=0)
    port = 0
    if integrate:
        x = b.add_block("Integrator", 0, 0, name="x", params={"init_conds": 0.5})
        b.connect(x, 0, f, 0)
        b.connect(f, 0, x, 0)
        b.connect(x, 0, sc, 0)
        port = 1
    else:
        b.connect(f, 0, sc, 0)
    for i, (fn, params) in enumerate(sources):
        src = b.add_block(fn, 0, 0, name="src{}".format(i), params=params)
        b.connect(src, 0, f, port + i)
    path = tmp_path / "function.diablos"
    b.save(str(path))
    return path


def _run(path, fast):
    from lib.analysis.resim import harvest_scope_signals
    from lib.cli import load_diagram
    from lib.engine.system_compiler import SystemCompiler

    calls = []
    original = SystemCompiler.compile_system

    def spy(self, *args, **kwargs):
        calls.append(True)
        return original(self, *args, **kwargs)

    SystemCompiler.compile_system = spy
    try:
        dsim, _ = load_diagram(str(path))
        dsim.use_fast_solver = fast
        ok, err = dsim.run_tuning_simulation(2.0, 0.01)
    finally:
        SystemCompiler.compile_system = original
    data = harvest_scope_signals(dsim) if ok else None
    return ok, err, dsim, data, bool(calls)


SINE = ("Sine", {"amplitude": 1.0, "omega": 3.0, "init_angle": 0.0})
CONST = ("Constant", {"value": 0.7})
CONST2 = ("Constant", {"value": -1.3})


@pytest.mark.regression
class TestCompiledMatchesInterpreter:
    @pytest.mark.parametrize(
        "expr, sources",
        [
            ("u[0]**2 + 0.5*t", [SINE]),
            ("np.tanh(2*u1) * u2", [SINE, CONST]),
            ("[u[0][0], u[0][0]*u[1][0], sin(t)]", [CONST, CONST2]),  # vector output
        ],
    )
    def test_algebraic(self, qapp, tmp_path, expr, sources):
        path = _diagram(tmp_path, expr, sources)
        ok_c, err_c, dsim, fast, compiled = _run(path, True)
        ok_i, err_i, _, slow, _ = _run(path, False)
        assert ok_c and ok_i, (err_c, err_i)
        assert compiled and dsim.engine.get_compile_fallback_reason() is None
        assert dsim.engine.last_solver_diagnostics["success"]
        y_c = np.asarray(list(fast["signals"].values()), dtype=float)
        y_i = np.asarray(list(slow["signals"].values()), dtype=float)
        n = min(y_c.shape[-1], y_i.shape[-1])
        np.testing.assert_allclose(y_c[..., :n], y_i[..., :n], atol=1e-9)

    @pytest.mark.parametrize(
        "expr, sources",
        [
            ("-sin(u[0]) + u[1]", [CONST]),  # pendulum-like nonlinearity
            ("-np.tanh(3*u1) + 0.2*u2", [SINE]),
        ],
    )
    def test_dynamic_loop_through_an_integrator(self, qapp, tmp_path, expr, sources):
        path = _diagram(tmp_path, expr, sources, integrate=True)
        ok_c, err_c, dsim, fast, compiled = _run(path, True)
        ok_i, err_i, _, slow, _ = _run(path, False)
        assert ok_c and ok_i, (err_c, err_i)
        assert compiled and dsim.engine.last_solver_diagnostics["success"]
        y_c = np.ravel(list(fast["signals"].values())[0])
        y_i = np.ravel(list(slow["signals"].values())[0])
        n = min(y_c.size, y_i.size)
        # Adaptive RK45 vs the interpreter's fixed step: agree to integration error.
        np.testing.assert_allclose(y_c[:n], y_i[:n], atol=2e-3)

    def test_bad_expression_falls_back_and_reports_like_the_interpreter(self, qapp, tmp_path):
        path = _diagram(tmp_path, "u[0] + nosuchname", [SINE])
        ok_c, err_c, _, _, _ = _run(path, True)
        ok_i, err_i, _, _, _ = _run(path, False)
        assert not ok_c and not ok_i
        assert "Function expression error" in err_c and "Function expression error" in err_i


@pytest.mark.unit
def test_python_export_still_refuses_function(tmp_path):
    from lib.cli import export_python
    from lib.export.python_codegen import CodegenUnsupportedError

    path = _diagram(tmp_path, "u[0]**2", [SINE])
    with pytest.raises(CodegenUnsupportedError, match="Function"):
        export_python(str(path), str(tmp_path / "out.py"))
