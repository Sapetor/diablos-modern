"""Scope labels that collide with generated ``#k`` names keep every channel.

Three Constants (1, 2, 3) into Scopes labelled ``x``, ``x`` and ``x#1`` came
back as ``{"x": 1, "x#1": 3}`` from both solvers and from the exported Python
script: the generated ``x#1`` for the second ``x`` was overwritten by the
literal label.
"""

import numpy as np
import pytest

from tests.unit.test_python_codegen import _run_generated

LABELS = ["x", "x", "x#1"]
EXPECTED = {"x": 1.0, "x#1": 2.0, "x#1#1": 3.0}


def _builder(labels=LABELS, vector=False):
    from lib.diagram_builder import DiagramBuilder

    b = DiagramBuilder(sim_time=0.5, sim_dt=0.1)
    consts = [
        b.add_block("Constant", 0, 0, name=f"c{k}", params={"value": float(k + 1)})
        for k in range(len(labels))
    ]
    if vector:
        sc = b.add_block(
            "Scope",
            0,
            0,
            name="sc",
            params={"labels": ", ".join(labels)},
            in_ports=len(labels),
            out_ports=0,
        )
        for k, c in enumerate(consts):
            b.connect(c, 0, sc, k)
    else:
        for k, (c, lab) in enumerate(zip(consts, labels)):
            sc = b.add_block(
                "Scope", 0, 0, name=f"sc{k}", params={"labels": lab}, in_ports=1, out_ports=0
            )
            b.connect(c, 0, sc, 0)
    return b


def _last(signals):
    return {k: float(np.ravel(v)[-1]) for k, v in signals.items()}


@pytest.mark.regression
class TestScopeLabelCollisions:
    @pytest.mark.parametrize("solver", ["compiled", "interpreter"])
    @pytest.mark.parametrize("vector", [False, True])
    def test_api_keeps_every_channel(self, qapp, solver, vector):
        from lib import api

        result = api.from_builder(_builder(vector=vector)).run(solver=solver)
        assert _last(result.signals) == EXPECTED

    @pytest.mark.parametrize("vector", [False, True])
    def test_exported_script_names_match_the_app(self, qapp, tmp_path, vector):
        from lib import cli

        path = str(tmp_path / "labels.diablos")
        _builder(vector=vector).save(path)
        script = str(tmp_path / "labels.py")
        cli.export_python(path, out_path=script)
        generated = _run_generated(script, str(tmp_path / "out.npz"), ["--no-plot"])
        assert _last({k: generated[k] for k in generated.files if k != "t"}) == EXPECTED
