"""Regression tests for three numerical bugs reported at 4dcd717.

1. Inherited sample rates depended on the order blocks are stored in.  The
   engine's propagation loop turned an inheriting block whose predecessor was
   not resolved *yet* into continuous (-1) for good, and never revisited a
   block that had settled on a slower input before a faster inherited one
   resolved.  Constant -> ZOH(0.1) -> DTF(inherit) -> DTF(inherit) -> Scope
   gave rates 0.1/0.1/0.1 and a final value of 45 (compiled) in source->sink
   order, but 0.1/0.1/-1 and 450 (interpreter fallback) reversed.  The
   exporter had a copy of the same loop.  Both now share
   ``lib.engine.sample_times.propagate_sample_times``.

2. A sample period written as a workspace variable (``sampling_time="Ts"``)
   went through ``float()`` on the raw string and silently became continuous
   (final value 4950 instead of 45).  It is now read from the resolved params,
   and an unresolvable period is a validation error.

3. The exec_params cache keyed on the raw params + dt only, so after changing a
   workspace variable a linearization (which never re-runs DSim's resolve)
   kept using the old value: K=2 -> K=9 still gave B = [[2]].  The workspace
   content is now part of the cache key.
"""

import itertools
import logging
import os
import subprocess
import sys

import numpy as np
import pytest

INTEGRATING_DTF = {"numerator": [1.0], "denominator": [1.0, -1.0], "sampling_time": 0.0}


def _diagram(specs, connections, order, variables=None):
    """A headless ``lib.api.Diagram`` with its blocks stored in ``order``."""
    from lib.api import from_builder
    from lib.diagram_builder import DiagramBuilder

    builder = DiagramBuilder(sim_time=1.0, sim_dt=0.01)
    for i, name in enumerate(order):
        block_type, params, *extra = specs[name]
        ports = extra[0] if extra else {}
        builder.add_block(block_type, x=120 * i, y=100, name=name, params=dict(params), **ports)
    for src, sport, dst, dport in connections:
        builder.connect(src, sport, dst, dport)
    diagram = from_builder(builder)
    for key, value in (variables or {}).items():
        diagram.set_variable(key, value)
    return diagram


def _chain(zoh_ts=0.1):
    specs = {
        "c": ("Constant", {"value": 1.0}),
        "zoh": ("ZeroOrderHold", {"sampling_time": zoh_ts}),
        "d1": ("DiscreteTranFn", INTEGRATING_DTF),
        "d2": ("DiscreteTranFn", INTEGRATING_DTF),
        "sc": ("Scope", {}),
    }
    conns = [("c", 0, "zoh", 0), ("zoh", 0, "d1", 0), ("d1", 0, "d2", 0), ("d2", 0, "sc", 0)]
    return specs, conns


def _branched():
    """A PID inheriting from two inherited inputs at different rates.

    The fast (0.1) input resolves through a longer inherited chain than the
    slow (0.2) one, so a single in-order pass sees the slow rate first.
    """
    specs = {
        "c": ("Constant", {"value": 1.0}),
        "zs": ("ZeroOrderHold", {"sampling_time": 0.2}),
        "ds": ("DiscreteTranFn", INTEGRATING_DTF),
        "zf": ("ZeroOrderHold", {"sampling_time": 0.1}),
        "df1": ("DiscreteTranFn", INTEGRATING_DTF),
        "df2": ("DiscreteTranFn", INTEGRATING_DTF),
        "pid": (
            "PID",
            {"Kp": 1.0, "Ki": 1.0, "Kd": 0.0, "sampling_time": 0.0},
            {"in_ports": 2, "out_ports": 1},
        ),
        "sc": ("Scope", {}),
    }
    conns = [
        ("c", 0, "zs", 0),
        ("zs", 0, "ds", 0),
        ("ds", 0, "pid", 0),
        ("c", 0, "zf", 0),
        ("zf", 0, "df1", 0),
        ("df1", 0, "df2", 0),
        ("df2", 0, "pid", 1),
        ("pid", 0, "sc", 0),
    ]
    return specs, conns


def _run(diagram):
    """(effective rates, final Scope value, engine) of one run."""
    res = diagram.run(t=1.0, dt=0.01)
    rates = {b.name: b.effective_sample_time for b in diagram.dsim.engine.active_blocks_list}
    trace = next(iter(res.signals.values()))
    return rates, trace, res.engine


def _export_rates(diagram):
    from lib.export.python_codegen import PythonCodeGenerator

    gen = PythonCodeGenerator.from_dsim(diagram.dsim)
    source = gen.generate()
    return {b.name: b.rate for b in gen._blocks}, source


def _orders(names, limit=None):
    """Forward, reversed and (a bounded sample of) other permutations."""
    perms = [list(names), list(reversed(names))]
    for p in itertools.permutations(names):
        if list(p) not in perms:
            perms.append(list(p))
        if limit and len(perms) >= limit:
            break
    return perms


# ---------------------------------------------------------------------------
# 1. Order-independent inherited rates (engine + exporter)
# ---------------------------------------------------------------------------


@pytest.mark.regression
class TestInheritedRatesAreOrderIndependent:
    def test_chain_every_permutation(self, qapp):
        specs, conns = _chain()
        expected_rates = {"zoh": 0.1, "d1": 0.1, "d2": 0.1}
        reference = None
        for order in _orders(list(specs), limit=24):
            rates, trace, engine = _run(_diagram(specs, conns, order))
            assert {k: rates[k] for k in expected_rates} == pytest.approx(expected_rates), order
            assert engine == "compiled", order
            # y2[k] = sum_{j<k} y1[j], y1[j] = j  ->  y2(t=1) = 45
            assert trace[-1] == pytest.approx(45.0), order
            if reference is None:
                reference = trace
            np.testing.assert_allclose(trace, reference, err_msg=str(order))

    def test_branched_two_inherited_inputs_take_the_fastest(self, qapp):
        specs, conns = _branched()
        reference = None
        for order in _orders(list(specs), limit=30):
            rates, trace, _ = _run(_diagram(specs, conns, order))
            assert rates["ds"] == pytest.approx(0.2), order
            assert rates["df1"] == rates["df2"] == pytest.approx(0.1), order
            assert rates["pid"] == pytest.approx(0.1), order
            if reference is None:
                reference = trace
            np.testing.assert_allclose(trace, reference, err_msg=str(order))

    @pytest.mark.parametrize("make", [_chain, _branched])
    def test_exporter_matches_engine_in_every_order(self, qapp, make):
        specs, conns = make()
        for order in _orders(list(specs), limit=12):
            diagram = _diagram(specs, conns, order)
            exported, _ = _export_rates(diagram)
            engine_rates, _, _ = _run(diagram)
            for name in specs:
                assert exported[name] == pytest.approx(engine_rates[name]), (order, name)

    @pytest.mark.slow
    def test_exported_script_reproduces_reversed_chain(self, qapp, tmp_path):
        specs, conns = _chain()
        diagram = _diagram(specs, conns, list(reversed(list(specs))))
        _, source = _export_rates(diagram)
        script = tmp_path / "model.py"
        script.write_text(source)
        out = tmp_path / "out.npz"
        env = dict(os.environ, MPLBACKEND="Agg")
        proc = subprocess.run(
            [sys.executable, str(script), "--out", str(out), "--no-plot"],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(tmp_path),
            timeout=300,
        )
        assert proc.returncode == 0, proc.stderr
        data = np.load(out)
        signal = [k for k in data.files if k != "t"][0]
        assert np.ravel(data[signal])[-1] == pytest.approx(45.0)


@pytest.mark.regression
class TestPropagationFunction:
    def test_cycle_of_inheriting_blocks_reaches_the_discrete_rate(self):
        from lib.engine.sample_times import propagate_sample_times

        declared = {"z": 0.1, "a": 0.0, "b": 0.0, "u": 0.0}
        sources = {"a": ["z", "b"], "b": ["a"], "u": []}
        for keys in itertools.permutations(declared):
            rates = propagate_sample_times({k: declared[k] for k in keys}, sources)
            assert rates == {"z": 0.1, "a": 0.1, "b": 0.1, "u": -1.0}

    def test_continuous_inputs_do_not_count(self):
        from lib.engine.sample_times import propagate_sample_times

        rates = propagate_sample_times({"c": -1.0, "z": 0.3, "x": 0.0}, {"x": ["c", "z"]})
        assert rates["x"] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# 2. Workspace expressions in sample periods
# ---------------------------------------------------------------------------


@pytest.mark.regression
class TestWorkspaceSamplePeriods:
    @pytest.mark.parametrize(
        "zoh_ts, variables",
        [(0.1, None), ("Ts", {"Ts": 0.1}), ("Ts * 2 / 2", {"Ts": 0.1}), ("0.1", None)],
    )
    def test_literal_variable_and_expression_agree(self, qapp, zoh_ts, variables):
        specs, conns = _chain(zoh_ts)
        literal_rates, literal_trace, literal_engine = _run(_diagram(*_chain(0.1), list(specs)))
        rates, trace, engine = _run(_diagram(specs, conns, list(specs), variables))
        for name in ("zoh", "d1", "d2"):
            assert rates[name] == pytest.approx(literal_rates[name]) == 0.1
        assert engine == literal_engine == "compiled"
        np.testing.assert_allclose(trace, literal_trace)
        assert trace[-1] == pytest.approx(45.0)

    def test_variable_change_takes_effect_on_the_next_run(self, qapp):
        specs, conns = _chain("Ts")
        diagram = _diagram(specs, conns, list(specs), {"Ts": 0.1})
        assert _run(diagram)[1][-1] == pytest.approx(45.0)
        diagram.set_variable("Ts", 0.2)
        rates, trace, _ = _run(diagram)
        assert rates["d2"] == pytest.approx(0.2)
        # Samples at 0, 0.2, ..., 1.0: y2 = sum_{j<5} j = 10
        assert trace[-1] == pytest.approx(10.0)

    def test_exporter_reads_the_variable(self, qapp):
        specs, conns = _chain("Ts")
        exported, _ = _export_rates(_diagram(specs, conns, list(specs), {"Ts": 0.1}))
        assert exported["zoh"] == exported["d2"] == pytest.approx(0.1)

    @pytest.mark.parametrize("bad", ["Tq", "[0.1, 0.2]", "float('nan')"])
    def test_unresolvable_period_is_a_validation_error(self, qapp, bad, caplog):
        from lib.api import SimulationError

        specs, conns = _chain(bad)
        diagram = _diagram(specs, conns, list(specs), {"Ts": 0.1})
        with caplog.at_level(logging.CRITICAL):
            with pytest.raises(SimulationError, match=r"Block 'zoh': sample period"):
                diagram.run(t=1.0, dt=0.01)
        assert diagram.dsim.engine.error_block == "zoh"

    def test_unresolvable_period_is_refused_by_the_exporter(self, qapp):
        from lib.export.python_codegen import CodegenUnsupportedError

        specs, conns = _chain("Tq")
        with pytest.raises(CodegenUnsupportedError, match="sample period"):
            _export_rates(_diagram(specs, conns, list(specs)))

    def test_block_resolver_literal_beats_stale_resolution(self, qapp):
        """A literal typed after a run wins over the run's exec_params."""
        from lib.engine.sample_times import SampleTimeError, declared_sample_time

        assert declared_sample_time({"sampling_time": 0.05}, {"sampling_time": 0.2}) == 0.05
        assert declared_sample_time({"sampling_time": "Ts"}, {"sampling_time": 0.2}) == 0.2
        assert declared_sample_time({}, None) == -1.0
        with pytest.raises(SampleTimeError):
            declared_sample_time({"sampling_time": "Ts"}, {"sampling_time": "Ts"})
        with pytest.raises(SampleTimeError):
            declared_sample_time({"sampling_time": float("inf")})
