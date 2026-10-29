"""Bode / Nyquist / root locus resolve workspace variables.

``BaseAnalyzer`` built its LTI model from raw ``block.params``, so a transfer
function written as ``numerator="[K]"`` (or a Gain ``"K"`` inside a subsystem
chain) was never resolved against the diagram's workspace -- the analysis
either failed or used the wrong numbers, while the Linearizer path was right.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from lib.analysis.analyzers.base_analyzer import BaseAnalyzer
from lib.workspace import Workspace
from tests.unit.test_analyzer_subsystem_model import _io_ports, _primitive, _subsystem


@pytest.fixture(autouse=True)
def _require_qapp(qapp):
    return qapp


def _canvas(**variables):
    return SimpleNamespace(dsim=SimpleNamespace(workspace=Workspace(variables)))


@pytest.mark.unit
class TestWorkspaceParamsInAnalysis:
    def test_transfer_function_coefficients_from_variables(self):
        tf = _primitive("TranFn", "Tf1", 1, 1, 1, {"numerator": "[K]", "denominator": "[1, a]"})
        num, den, dt = BaseAnalyzer()._extract_system_model(tf, _canvas(K=2.0, a=3.0))
        np.testing.assert_allclose(num, [2.0])
        np.testing.assert_allclose(den, [1.0, 3.0])
        assert dt == 0.0

    def test_changing_the_variable_changes_the_model(self):
        tf = _primitive("TranFn", "Tf1", 1, 1, 1, {"numerator": "[K]", "denominator": [1.0, 1.0]})
        a = BaseAnalyzer()._extract_system_model(tf, _canvas(K=2.0))
        b = BaseAnalyzer()._extract_system_model(tf, _canvas(K=9.0))
        np.testing.assert_allclose(a[0], [2.0])
        np.testing.assert_allclose(b[0], [9.0])

    def test_gain_variable_inside_a_subsystem_chain(self):
        from lib.simulation.connection import DLine

        inport, outport = _io_ports()
        gain = _primitive("Gain", "Gain1", 3, 1, 1, {"gain": "K"})
        tf = _primitive("TranFn", "Tf1", 4, 1, 1, {"numerator": [1.0], "denominator": [1.0, 2.0]})
        subsys = _subsystem(
            "Sub1",
            [inport, gain, tf, outport],
            [
                DLine(1, "In1", 0, "Gain1", 0, [(0, 0), (10, 10)]),
                DLine(2, "Gain1", 0, "Tf1", 0, [(0, 0), (10, 10)]),
                DLine(3, "Tf1", 0, "Out1", 0, [(0, 0), (10, 10)]),
            ],
        )
        analyzer = BaseAnalyzer()
        model = analyzer._extract_system_model(subsys, _canvas(K=4.0))
        assert model is not None, analyzer.last_error
        num, den, _ = model
        np.testing.assert_allclose(num / den[0], [4.0])
        np.testing.assert_allclose(den / den[0], [1.0, 2.0])

    def test_without_a_workspace_literals_still_work(self):
        tf = _primitive("TranFn", "Tf1", 1, 1, 1, {"numerator": [1.0], "denominator": [1.0, 1.0]})
        num, den, _ = BaseAnalyzer()._extract_system_model(tf, None)
        np.testing.assert_allclose(num, [1.0])
