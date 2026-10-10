"""System Identification source block (blocks/system_identification.py)."""

import numpy as np
import pytest
from scipy import signal

from blocks.system_identification import SystemIdentificationBlock


@pytest.fixture
def experiment(tmp_path):
    """Step test of G(s) = 4 / (s^2 + 2 s + 4) from an operating point."""
    t = np.linspace(0, 10, 1001)
    u = 1.0 + np.where(t >= 1.0, 0.5, 0.0)
    _, dy, _ = signal.lsim(signal.lti([4.0], [1.0, 2.0, 4.0]), u - 1.0, t)
    y = 1.0 + dy  # at equilibrium (G(0) = 1) before the step
    path = tmp_path / "step_test.csv"
    np.savetxt(path, np.column_stack([t, u, y]), delimiter=",", header="t,u,y", comments="")
    return str(path)


def _params(path, **over):
    p = {"data_file": path, "time_col": "t", "input_col": "u", "output_col": "y"}
    p.update(num_order=0, den_order=2, _init_start_=True, _name_="sysid0")
    p.update(over)
    return p


@pytest.mark.unit
class TestSystemIdentificationBlock:
    def test_identifies_and_replays(self, experiment):
        block = SystemIdentificationBlock()
        params = _params(experiment)
        out = block.execute(time=0.0, inputs={}, params=params)
        assert out["E"] is False
        np.testing.assert_allclose(params["_fit_den_"], [1.0, 2.0, 4.0], rtol=3e-2)
        np.testing.assert_allclose(params["_fit_num_"], [4.0], rtol=3e-2)
        assert params["_fit_percent_"] > 99
        late = block.execute(time=8.0, inputs={}, params=params)
        assert late[0][0] == pytest.approx(late[1][0], abs=0.02)
        assert late[1][0] == pytest.approx(1.5, abs=0.02)

    def test_fits_once_per_run(self, experiment, monkeypatch):
        import lib.analysis.sysid as sysid

        calls = []
        real = sysid.identify_tf
        monkeypatch.setattr(sysid, "identify_tf", lambda *a, **k: calls.append(1) or real(*a, **k))
        block = SystemIdentificationBlock()
        params = _params(experiment)
        for t in (0.0, 0.1, 0.2):
            block.execute(time=t, inputs={}, params=params)
        assert len(calls) == 1
        params["_init_start_"] = True  # engine reset for the next run
        block.execute(time=0.0, inputs={}, params=params)
        assert len(calls) == 2

    def test_errors_are_reported(self, experiment, tmp_path):
        block = SystemIdentificationBlock()
        assert block.execute(0.0, {}, _params(""))["E"] is True
        missing = block.execute(0.0, {}, _params(str(tmp_path / "nope.csv")))
        assert missing["E"] is True and "System identification" in missing["error"]
        bad = block.execute(0.0, {}, _params(experiment, num_order=3))
        assert bad["E"] is True and "numerator order" in bad["error"]

    def test_is_a_source(self):
        block = SystemIdentificationBlock()
        assert block.inputs == [] and not block.requires_inputs
        assert [o["name"] for o in block.outputs] == ["model", "measured"]


@pytest.mark.integration
def test_headless_run_overlays_model_and_measurement(experiment, tmp_path):
    from lib.analysis.resim import harvest_scope_signals
    from lib.cli import run_diagram
    from lib.diagram_builder import DiagramBuilder

    b = DiagramBuilder(sim_time=8.0, sim_dt=0.01)
    sid = b.add_block(
        "SystemIdentification",
        x=100,
        y=100,
        name="sysid",
        params=_params(experiment),
        in_ports=0,
        out_ports=2,
    )
    scope = b.add_block("Scope", x=300, y=100, name="scope", in_ports=2)
    b.connect(sid, 0, scope, 0)
    b.connect(sid, 1, scope, 1)
    path = tmp_path / "sysid.diablos"
    b.save(str(path))

    result = harvest_scope_signals(run_diagram(str(path)))
    model, measured = (np.ravel(v) for v in list(result["signals"].values())[:2])
    assert np.max(np.abs(model - measured)) < 0.02
