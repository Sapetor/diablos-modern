"""Standalone Python script export (``lib/export/python_codegen.py``).

Three levels of checking:

* the emitted text is valid Python and only imports numpy/scipy/stdlib;
* the emitted script *runs* and reproduces what the headless compiled path
  produces for the same diagram (examples + an in-memory PID/tank loop);
* diagrams the exporter cannot express raise ``CodegenUnsupportedError`` naming
  the offending blocks instead of emitting broken code.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from lib.export.python_codegen import (
    CodegenError,
    CodegenUnsupportedError,
    PythonCodeGenerator,
    generate_python_script,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXAMPLES = os.path.join(REPO_ROOT, "examples")


# ---------------------------------------------------------------------------
# Lightweight stand-ins for DBlock / DLine (the generator only duck-types).
# ---------------------------------------------------------------------------


class FakeBlock:
    def __init__(self, name, block_fn, params=None, in_ports=1, out_ports=1, username=""):
        self.name = name
        self.block_fn = block_fn
        self.params = params or {}
        self.in_ports = in_ports
        self.out_ports = out_ports
        self.username = username or name


class FakeLine:
    def __init__(self, srcblock, srcport, dstblock, dstport):
        self.srcblock = srcblock
        self.srcport = srcport
        self.dstblock = dstblock
        self.dstport = dstport


def _first_order_loop():
    """Step -> Sum(+-) -> Gain -> TranFn 1/(s+0.5) -> Scope, with feedback."""
    blocks = [
        FakeBlock("step0", "Step", {"value": 2.0, "delay": 1.0, "type": "up"}, 0, 1, "ref"),
        FakeBlock("sum1", "Sum", {"sign": "+-"}, 2, 1, "error"),
        FakeBlock("gain2", "Gain", {"gain": 3.0}, 1, 1, "Kp"),
        FakeBlock(
            "tranfn3",
            "TranFn",
            {"numerator": [1.0], "denominator": [1.0, 0.5]},
            1,
            1,
            "plant",
        ),
        FakeBlock("scope4", "Scope", {"labels": "ref,y"}, 2, 0, "scope"),
    ]
    lines = [
        FakeLine("step0", 0, "sum1", 0),
        FakeLine("tranfn3", 0, "sum1", 1),
        FakeLine("sum1", 0, "gain2", 0),
        FakeLine("gain2", 0, "tranfn3", 0),
        FakeLine("step0", 0, "scope4", 0),
        FakeLine("tranfn3", 0, "scope4", 1),
    ]
    return blocks, lines


def _block_data(fn, sid, username, params, ins, outs, b_type=2, io_edit="none"):
    """One entry of a ``.diablos`` file's ``blocks_data``."""
    return {
        "block_fn": fn,
        "sid": sid,
        "username": username,
        "coords_left": 50 + 90 * sid,
        "coords_top": 100 + 40 * sid,
        "coords_width": 60,
        "coords_height": 50,
        "coords_height_base": 50,
        "in_ports": ins,
        "out_ports": outs,
        "dragging": False,
        "selected": False,
        "b_color": "#1e3a8a",
        "b_type": b_type,
        "io_edit": io_edit,
        "fn_name": fn.lower(),
        "params": params,
        "external": False,
        "flipped": False,
    }


def _line_data(sid, src, srcport, dst, dstport):
    """One entry of a ``.diablos`` file's ``lines_data``."""
    return {
        "name": "L%d" % sid,
        "sid": sid,
        "srcblock": src,
        "srcport": srcport,
        "dstblock": dst,
        "dstport": dstport,
        "points": [[0, 0], [1, 1]],
        "cptr": 0,
        "selected": False,
    }


def _diagram(blocks_data, lines_data, sim_time=10.0, sim_dt=0.01):
    return {
        "sim_data": {
            "wind_width": 1280,
            "wind_height": 770,
            "fps": 60,
            "sim_time": sim_time,
            "sim_dt": sim_dt,
            "sim_trange": 100,
        },
        "blocks_data": blocks_data,
        "lines_data": lines_data,
        "version": "2.0",
    }


def _pid_tank_diagram():
    """PID/tank feedback loop as a ``.diablos``-format dict.

    Wired the way the compiled PID kernel expects: port 0 = setpoint,
    port 1 = measurement.
    """
    return _diagram(
        [
            _block_data("Step", 0, "ref", {"value": 1.0, "delay": 1.0, "type": "up"}, 0, 1, 0),
            _block_data("PID", 1, "ctrl", {"Kp": 2.0, "Ki": 1.0, "Kd": 0.1, "N": 20.0}, 2, 1),
            _block_data("TranFn", 2, "tank", {"numerator": [1.0], "denominator": [1.0, 0.5]}, 1, 1),
            _block_data("Scope", 3, "scope", {"labels": "ref,level"}, 2, 0, 3, "in"),
        ],
        [
            _line_data(0, "step0", 0, "pid1", 0),
            _line_data(1, "tranfn2", 0, "pid1", 1),
            _line_data(2, "pid1", 0, "tranfn2", 0),
            _line_data(3, "step0", 0, "scope3", 0),
            _line_data(4, "tranfn2", 0, "scope3", 1),
        ],
    )


def _kitchen_sink_diagram():
    """One diagram touching every emitter the example diagrams don't.

    Constant, Sine, Ramp, Product (with a division), Abs, Saturation, Mux,
    Demux (secondary output port), Integrator and SgProd, all landing on one
    seven-port Scope.
    """
    return _diagram(
        [
            _block_data("Constant", 0, "c", {"value": 2.0}, 0, 1, 0),
            _block_data(
                "Sine", 1, "s", {"amplitude": 1.0, "omega": 2.0, "init_angle": 0.3}, 0, 1, 0
            ),
            _block_data("Ramp", 2, "r", {"slope": 0.5, "delay": 0.2}, 0, 1, 0),
            _block_data("Product", 3, "div", {"ops": "*/"}, 2, 1),
            _block_data("Abs", 4, "a", {}, 1, 1),
            _block_data("Saturation", 5, "sat", {"min": -0.5, "max": 0.5}, 1, 1),
            _block_data("Mux", 6, "mx", {}, 2, 1),
            _block_data("Demux", 7, "dm", {"output_shape": 1, "_outputs_": 2}, 1, 2),
            _block_data("Integrator", 8, "itg", {"init_conds": 0.0, "method": "SOLVE_IVP"}, 1, 1),
            _block_data("SgProd", 9, "prod", {}, 2, 1),
            _block_data(
                "Scope", 10, "scope", {"labels": "div,abs,sat,d0,d1,itg,prod"}, 7, 0, 3, "in"
            ),
        ],
        [
            _line_data(0, "sine1", 0, "product3", 0),
            _line_data(1, "constant0", 0, "product3", 1),
            _line_data(2, "ramp2", 0, "abs4", 0),
            _line_data(3, "sine1", 0, "saturation5", 0),
            _line_data(4, "sine1", 0, "mux6", 0),
            _line_data(5, "constant0", 0, "mux6", 1),
            _line_data(6, "mux6", 0, "demux7", 0),
            _line_data(7, "demux7", 1, "integrator8", 0),
            _line_data(8, "sine1", 0, "sgprod9", 0),
            _line_data(9, "ramp2", 0, "sgprod9", 1),
            _line_data(10, "product3", 0, "scope10", 0),
            _line_data(11, "abs4", 0, "scope10", 1),
            _line_data(12, "saturation5", 0, "scope10", 2),
            _line_data(13, "demux7", 0, "scope10", 3),
            _line_data(14, "demux7", 1, "scope10", 4),
            _line_data(15, "integrator8", 0, "scope10", 5),
            _line_data(16, "sgprod9", 0, "scope10", 6),
        ],
        sim_time=5.0,
    )


# ---------------------------------------------------------------------------
# Generated source
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGeneratedSource:
    def test_source_compiles(self):
        blocks, lines = _first_order_loop()
        source = generate_python_script(blocks, lines, sim_time=5.0, sim_dt=0.01)
        compile(source, "<generated>", "exec")

    def test_has_the_documented_structure(self):
        blocks, lines = _first_order_loop()
        source = generate_python_script(blocks, lines, diagram_name="loop.diablos")
        assert "def evaluate(t, x):" in source
        assert "def rhs(t, x):" in source
        assert "solve_ivp(" in source
        assert "X0 = np.array(" in source
        # State-space matrices for the transfer function, named after the block.
        assert "A_plant = np.array(" in source
        assert "B_plant = np.array(" in source
        assert "C_plant = np.array(" in source
        # Parameters are hoisted into named constants, not inlined magic numbers.
        assert "KP_GAIN = 3.0" in source
        assert "REF_VALUE = 2.0" in source
        assert "REF_DELAY = 1.0" in source
        # The feedback sum keeps its sign convention.
        assert "error = ref - plant" in source

    def test_only_needs_numpy_and_scipy(self):
        blocks, lines = _first_order_loop()
        source = generate_python_script(blocks, lines)
        imports = [ln.strip() for ln in source.splitlines() if ln.startswith(("import ", "from "))]
        assert imports == [
            "import argparse",
            "import os",
            "import sys",
            "import numpy as np",
            "from scipy.integrate import solve_ivp",
        ]
        assert "import lib" not in source and "from lib" not in source

    def test_stateless_diagram_and_sinks(self):
        blocks = [
            FakeBlock("step0", "Step", {"value": 1.0, "delay": 0.0, "type": "up"}, 0, 1, "u"),
            FakeBlock("gain1", "Gain", {"gain": 2.0}, 1, 1, "K"),
            FakeBlock("term2", "Terminator", {}, 1, 0, "term"),
            FakeBlock("disp3", "Display", {}, 1, 0, "disp"),
            FakeBlock("scope4", "Scope", {"labels": "y"}, 1, 0, "scope"),
        ]
        lines = [
            FakeLine("step0", 0, "gain1", 0),
            FakeLine("gain1", 0, "term2", 0),
            FakeLine("gain1", 0, "disp3", 0),
            FakeLine("gain1", 0, "scope4", 0),
        ]
        source = generate_python_script(blocks, lines)
        compile(source, "<generated>", "exec")
        assert "X0 = np.array([])" in source  # purely algebraic: no ODE state
        # Sinks consume signals but contribute none.
        signals = source.split("signals = {")[1].split("}")[0]
        assert '"u": u,' in signals and '"K": K,' in signals
        assert "term" not in signals and "disp" not in signals

    def test_fixed_step_solver_emits_its_own_loop(self):
        blocks, lines = _first_order_loop()
        source = generate_python_script(blocks, lines, solver="RK4")
        compile(source, "<generated>", "exec")
        assert "solve_ivp(" not in source.split("def simulate")[1].split("def ")[0]
        assert "k1 = np.asarray(rhs(" in source


# ---------------------------------------------------------------------------
# Unsupported diagrams
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestUnsupportedDiagrams:
    def test_unsupported_block_is_named(self):
        blocks, lines = _first_order_loop()
        blocks.append(FakeBlock("noise5", "Noise", {"mu": 0.0, "sigma": 1.0}, 0, 1))
        blocks.append(FakeBlock("hyst6", "Hysteresis", {}, 1, 1, "relay"))
        with pytest.raises(CodegenUnsupportedError) as excinfo:
            generate_python_script(blocks, lines)
        names = {name for name, _fn, _reason in excinfo.value.blocks}
        assert names == {"noise5", "hyst6"}
        message = str(excinfo.value)
        assert "noise5" in message and "Noise" in message
        assert "hyst6" in message and "Hysteresis" in message
        # The message tells the user what *is* supported.
        assert "Supported blocks:" in message

    def test_unsupported_error_is_a_codegen_error(self):
        blocks = [FakeBlock("noise0", "Noise", {}, 0, 1)]
        with pytest.raises(CodegenError):
            generate_python_script(blocks, [])

    def test_impulse_step_is_rejected(self):
        # Step(type='impulse') runs on the interpreter, not the compiled path,
        # so exporting it would silently change the answer.
        blocks, lines = _first_order_loop()
        blocks[0].params["type"] = "impulse"
        with pytest.raises(CodegenUnsupportedError) as excinfo:
            generate_python_script(blocks, lines)
        assert "step0" in str(excinfo.value)

    def test_discrete_sample_time_is_rejected(self):
        blocks, lines = _first_order_loop()
        blocks[2].params["sampling_time"] = 0.1
        with pytest.raises(CodegenUnsupportedError) as excinfo:
            generate_python_script(blocks, lines)
        assert "gain2" in str(excinfo.value)

    def test_algebraic_loop_is_rejected(self):
        blocks = [
            FakeBlock("sum0", "Sum", {"sign": "+-"}, 2, 1),
            FakeBlock("gain1", "Gain", {"gain": 2.0}, 1, 1),
        ]
        lines = [FakeLine("sum0", 0, "gain1", 0), FakeLine("gain1", 0, "sum0", 1)]
        with pytest.raises(CodegenError) as excinfo:
            generate_python_script(blocks, lines)
        assert "Algebraic loop" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Numeric parity with the headless compiled path
# ---------------------------------------------------------------------------


def _run_generated(script_path, out_path, extra_args=(), env_extra=None):
    """Execute a generated script headlessly; return the loaded .npz."""
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    env.update(env_extra or {})
    proc = subprocess.run(
        [sys.executable, script_path, "--out", out_path] + list(extra_args),
        capture_output=True,
        text=True,
        env=env,
        cwd=os.path.dirname(script_path),
        timeout=300,
    )
    assert proc.returncode == 0, "generated script failed:\n{}\n{}".format(proc.stdout, proc.stderr)
    return np.load(out_path)


def _reference_signals(diagram_path):
    """Run the diagram on the existing headless compiled path."""
    from lib import cli
    from lib.analysis.resim import harvest_scope_signals

    dsim = cli.run_diagram(diagram_path)
    result = harvest_scope_signals(dsim)
    assert result is not None and result["signals"], "reference run produced no Scope data"
    return result


def _assert_matches(generated, reference, rtol=1e-3, atol=1e-6):
    """Compare a generated .npz against harvest_scope_signals output."""
    t_ref = np.asarray(reference["timeline"], dtype=float).ravel()
    t_gen = np.asarray(generated["t"], dtype=float).ravel()
    assert t_gen[0] == pytest.approx(t_ref[0])
    assert t_gen[-1] == pytest.approx(t_ref[-1])

    for label, ref_trace in reference["signals"].items():
        assert label in generated.files, "missing signal {!r} (have {})".format(
            label, generated.files
        )
        gen_trace = np.asarray(generated[label], dtype=float).ravel()
        ref_trace = np.asarray(ref_trace, dtype=float).ravel()
        if len(gen_trace) != len(ref_trace):
            # Align on the reference grid before comparing (RK45 output grids
            # agree in practice; this keeps the check honest if they ever don't).
            gen_trace = np.interp(t_ref, t_gen[: len(gen_trace)], gen_trace)
        np.testing.assert_allclose(gen_trace, ref_trace, rtol=rtol, atol=atol)


@pytest.mark.unit
@pytest.mark.slow
class TestMatchesHeadlessRun:
    """The exported script must reproduce the app's own compiled solver."""

    def _export(self, diagram_path, tmp_path):
        from lib import cli

        script = str(tmp_path / "model.py")
        cli.export_python(diagram_path, out_path=script)
        return script

    def test_tank_feedback_example(self, qapp, tmp_path):
        # Closed loop: Step -> Sum -> Gain -> TranFn -> Scope (+ feedback).
        diagram = os.path.join(EXAMPLES, "c01_tank_feedback.diablos")
        script = self._export(diagram, tmp_path)
        data = _run_generated(script, str(tmp_path / "gen.npz"), ["--no-plot"])
        _assert_matches(data, _reference_signals(diagram))

    def test_first_order_step_response_example(self, qapp, tmp_path):
        # Step -> TranFn 1/(s+1) -> Integrator, both traces on one Scope.
        diagram = os.path.join(EXAMPLES, "c02_vehicle_single_agent.diablos")
        script = self._export(diagram, tmp_path)
        data = _run_generated(script, str(tmp_path / "gen.npz"), ["--no-plot"])
        _assert_matches(data, _reference_signals(diagram))

    def test_pid_tank_loop(self, qapp, tmp_path):
        diagram = str(tmp_path / "pid_tank.diablos")
        with open(diagram, "w") as fp:
            json.dump(_pid_tank_diagram(), fp)
        script = self._export(diagram, tmp_path)
        data = _run_generated(script, str(tmp_path / "gen.npz"), ["--no-plot"])
        reference = _reference_signals(diagram)
        _assert_matches(data, reference)
        # Sanity: the loop actually tracks the unit setpoint.
        assert data["level"][-1] == pytest.approx(1.0, abs=1e-2)

    def test_every_other_emitter(self, qapp, tmp_path):
        # Constant/Sine/Ramp/Product/Abs/Saturation/Mux/Demux/Integrator/SgProd.
        diagram = str(tmp_path / "kitchen_sink.diablos")
        with open(diagram, "w") as fp:
            json.dump(_kitchen_sink_diagram(), fp)
        script = self._export(diagram, tmp_path)
        data = _run_generated(script, str(tmp_path / "gen.npz"), ["--no-plot"])
        _assert_matches(data, _reference_signals(diagram))


@pytest.mark.unit
@pytest.mark.slow
class TestGeneratedScriptCli:
    def _write(self, tmp_path):
        blocks, lines = _first_order_loop()
        source = PythonCodeGenerator(
            blocks, lines, sim_time=2.0, sim_dt=0.01, diagram_name="loop.diablos"
        ).generate()
        script = str(tmp_path / "loop.py")
        with open(script, "w") as fp:
            fp.write(source)
        return script

    def test_env_var_disables_plotting(self, tmp_path):
        # No --no-plot flag: DIABLOS_NO_PLOT=1 must be enough.
        script = self._write(tmp_path)
        data = _run_generated(script, str(tmp_path / "out.npz"), env_extra={"DIABLOS_NO_PLOT": "1"})
        assert set(data.files) == {"t", "ref", "y"}
        assert len(data["t"]) == 201  # 0 .. 2 s at dt = 0.01

    def test_csv_output_and_time_override(self, tmp_path):
        script = self._write(tmp_path)
        out_csv = str(tmp_path / "out.csv")
        env = dict(os.environ, MPLBACKEND="Agg")
        proc = subprocess.run(
            [sys.executable, script, "--no-plot", "--time", "1", "--dt", "0.1", "--out", out_csv],
            capture_output=True,
            text=True,
            env=env,
            timeout=300,
        )
        assert proc.returncode == 0, proc.stderr
        rows = open(out_csv).read().strip().splitlines()
        assert rows[0] == "t,ref,y"
        assert len(rows) - 1 == 11  # header + t = 0.0 .. 1.0 step 0.1
