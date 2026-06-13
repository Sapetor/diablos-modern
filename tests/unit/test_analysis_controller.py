"""
Tests for AnalysisController (modern_ui/controllers/analysis_controller.py).

The controller is HEADLESS (no Qt). It wraps Linearizer and produces the shared
result-dict contract consumed by the results window. Diagrams are built with
DiagramBuilder, saved, and reloaded into a fresh DSim (same pattern as
tests/unit/test_linearizer.py).
"""

import math

import numpy as np
import pytest

from lib.diagram_builder import DiagramBuilder
from modern_ui.controllers.analysis_controller import AnalysisController


_BLOCK_INSTANCES = None


def _params(block_type, **overrides):
    """Full flat params for a block (class defaults + overrides)."""
    global _BLOCK_INSTANCES
    if _BLOCK_INSTANCES is None:
        from lib.block_loader import load_blocks
        _BLOCK_INSTANCES = {}
        for cls in load_blocks():
            try:
                inst = cls()
                _BLOCK_INSTANCES[inst.block_name] = inst
            except Exception:
                pass
    inst = _BLOCK_INSTANCES.get(block_type)
    out = {}
    if inst is not None:
        for k, v in inst.params.items():
            out[k] = v['default'] if isinstance(v, dict) and 'default' in v else v
    out.update(overrides)
    return out


def _load(builder, tmp_path, name):
    """Save a built diagram and load it into a fresh DSim."""
    from lib.lib import DSim
    from lib.workspace import WorkspaceManager

    path = tmp_path / name
    builder.save(str(path))

    WorkspaceManager._instance = None
    dsim = DSim()
    data = dsim.file_service.load(filepath=str(path))
    assert data is not None
    dsim.file_service.apply_loaded_data(data)
    return dsim


def _name_of(dsim, block_fn):
    """Return the internal name of the first block with the given block_fn."""
    for b in dsim.blocks_list:
        if b.block_fn == block_fn:
            return b.name
    raise AssertionError(f"No block with block_fn={block_fn!r} in diagram")


@pytest.mark.unit
class TestAnalysisController:
    def test_gain_feedback_one_pole_near_minus_three(self, qapp, tmp_path):
        """Integrator <- Gain(-3) <- Integrator: x' = -3x.

        A-only path (no I/O designated): ok=True, one pole near -3, stable.
        controllable/observable are None (no I/O), bode/margins None, zeros [].
        """
        b = DiagramBuilder()
        i = b.add_block("Integrator", 200, 100, params=_params("Integrator", init_conds=1.0))
        g = b.add_block("Gain", 350, 100, params=_params("Gain", gain=-3.0))
        b.connect(i, 0, g, 0)
        b.connect(g, 0, i, 0)
        dsim = _load(b, tmp_path, "gainfb.diablos")

        ctrl = AnalysisController(dsim)
        res = ctrl.analyze()

        assert res["ok"] is True
        assert res["error"] == ""
        assert res["n_states"] == 1
        assert len(res["poles"]) == 1
        re, im = res["poles"][0]
        assert abs(re - (-3.0)) < 1e-5
        assert abs(im) < 1e-9
        assert res["is_stable"] is True
        assert np.allclose(sorted(res["time_constants"]), [1.0 / 3.0], atol=1e-6)
        # No I/O designated.
        assert res["bode"] is None
        assert res["gain_margin_db"] is None
        assert res["phase_margin_deg"] is None
        assert res["zeros"] == []
        assert res["controllable"] is None
        assert res["observable"] is None
        assert res["tf_num"] is None
        assert isinstance(res["summary"], str) and res["summary"].strip()
        # state_names carried through.
        assert len(res["state_names"]) == 1

    def test_siso_with_io_yields_bode_and_margins_and_tf(self, qapp, tmp_path):
        """Constant -> Integrator -> Scope, input=Constant, output=Integrator.

        TF = 1/s. With a designated SISO I/O pair we get B/C/D, a transfer
        function, a non-empty bode dict, and finite-or-inf margins.
        """
        b = DiagramBuilder()
        c = b.add_block("Constant", 50, 100, params=_params("Constant", value=1.0))
        i = b.add_block("Integrator", 200, 100, params=_params("Integrator", init_conds=0.0))
        sc = b.add_block("Scope", 350, 100, params=_params("Scope"))
        b.connect(c, 0, i, 0)
        b.connect(i, 0, sc, 0)
        dsim = _load(b, tmp_path, "siso.diablos")

        src = _name_of(dsim, "Constant")
        integ = _name_of(dsim, "Integrator")

        ctrl = AnalysisController(dsim)
        res = ctrl.analyze(input_blocks=[src], output_blocks=[integ])

        assert res["ok"] is True
        assert res["error"] == ""
        # A/B/C/D present and shaped.
        assert np.allclose(res["A"], [[0.0]], atol=1e-6)
        assert np.allclose(res["B"], [[1.0]], atol=1e-6)
        assert np.allclose(res["C"], [[1.0]], atol=1e-6)
        assert np.allclose(res["D"], [[0.0]], atol=1e-6)
        # Transfer function 1/s.
        assert res["tf_num"] is not None
        assert res["tf_den"] is not None
        # Bode is a non-empty dict with equal-length vectors.
        bode = res["bode"]
        assert isinstance(bode, dict)
        assert len(bode["w"]) > 0
        assert len(bode["w"]) == len(bode["mag_db"]) == len(bode["phase_deg"])
        for arr in (bode["w"], bode["mag_db"], bode["phase_deg"]):
            assert all(math.isfinite(v) for v in arr)
        # Margins are finite-or-inf floats (or None), never NaN.
        for key in ("gain_margin_db", "phase_margin_deg"):
            val = res[key]
            assert val is None or (isinstance(val, float) and not math.isnan(val))
        # controllable/observable now known (single integrator -> both True).
        assert res["controllable"] is True
        assert res["observable"] is True
        # zeros: 1/s has no finite zeros.
        assert res["zeros"] == []

    def test_uncompilable_noise_returns_not_ok_without_raising(self, qapp, tmp_path):
        """A Noise diagram is interpreter-only -> not compilable.

        analyze() must return ok=False with a non-empty error, not raise.
        """
        b = DiagramBuilder()
        n = b.add_block("Noise", 50, 100, params=_params("Noise"))
        s = b.add_block("Scope", 200, 100, params=_params("Scope"))
        b.connect(n, 0, s, 0)
        dsim = _load(b, tmp_path, "noisy.diablos")

        ctrl = AnalysisController(dsim)
        res = ctrl.analyze()  # must not raise

        assert res["ok"] is False
        assert isinstance(res["error"], str) and res["error"].strip()
        # Minimal keys still present for the consuming window.
        assert "summary" in res
        assert res["n_states"] == 0


@pytest.mark.unit
class TestStepImpulseResponse:
    def test_siso_has_step_and_impulse(self, qapp, tmp_path):
        """Constant -> Integrator -> Scope (TF = 1/s).

        Step response of 1/s is a ramp (y == t); impulse response is constant 1.
        """
        b = DiagramBuilder()
        c = b.add_block("Constant", 50, 100, params=_params("Constant", value=1.0))
        i = b.add_block("Integrator", 200, 100, params=_params("Integrator", init_conds=0.0))
        sc = b.add_block("Scope", 350, 100, params=_params("Scope"))
        b.connect(c, 0, i, 0)
        b.connect(i, 0, sc, 0)
        dsim = _load(b, tmp_path, "siso_si.diablos")

        src = _name_of(dsim, "Constant")
        integ = _name_of(dsim, "Integrator")
        res = AnalysisController(dsim).analyze(input_blocks=[src], output_blocks=[integ])

        step = res["step_response"]
        impulse = res["impulse_response"]
        assert isinstance(step, dict) and isinstance(impulse, dict)
        assert len(step["t"]) == len(step["y"]) > 0
        assert len(impulse["t"]) == len(impulse["y"]) > 0
        # Step of 1/s is a ramp: final y ~= final t.
        assert np.isclose(step["y"][-1], step["t"][-1], rtol=1e-3, atol=1e-6)
        # Impulse of 1/s is the constant 1.
        assert np.allclose(impulse["y"], 1.0, atol=1e-6)
        # All finite.
        assert all(np.isfinite(step["y"])) and all(np.isfinite(impulse["y"]))

    def test_no_io_has_no_step_impulse(self, qapp, tmp_path):
        """Without a designated SISO I/O pair there is no TF -> no time response."""
        b = DiagramBuilder()
        i = b.add_block("Integrator", 200, 100, params=_params("Integrator", init_conds=1.0))
        g = b.add_block("Gain", 350, 100, params=_params("Gain", gain=-3.0))
        b.connect(i, 0, g, 0)
        b.connect(g, 0, i, 0)
        dsim = _load(b, tmp_path, "noio_si.diablos")

        res = AnalysisController(dsim).analyze()
        assert res["step_response"] is None
        assert res["impulse_response"] is None


@pytest.mark.unit
class TestFindTrim:
    def test_first_order_equilibrium(self, qapp, tmp_path):
        """x' = 6 - 3x has the equilibrium x* = 2.

        Constant(6) + Gain(-3)*x summed into an Integrator.
        """
        b = DiagramBuilder()
        c = b.add_block("Constant", 50, 100, params=_params("Constant", value=6.0))
        s = b.add_block("Sum", 150, 100, params=_params("Sum"))
        i = b.add_block("Integrator", 300, 100, params=_params("Integrator", init_conds=0.0))
        g = b.add_block("Gain", 300, 220, params=_params("Gain", gain=-3.0))
        b.connect(c, 0, s, 0)
        b.connect(g, 0, s, 1)
        b.connect(s, 0, i, 0)
        b.connect(i, 0, g, 0)
        dsim = _load(b, tmp_path, "trim_first.diablos")

        res = AnalysisController(dsim).find_trim()
        assert res["ok"] is True
        assert res["success"] is True
        assert len(res["states"]) == 1
        assert np.isclose(res["states"][0]["value"], 2.0, atol=1e-6)
        # Residual at the equilibrium is ~0.
        assert res["residual_norm"] is not None
        assert res["residual_norm"] < 1e-6
        # operating_point is keyed by block name and reusable.
        assert any(np.isclose(v, 2.0, atol=1e-6)
                   for v in res["operating_point"].values())
        assert isinstance(res["summary"], str) and res["summary"].strip()

    def test_no_states_reports_cleanly(self, qapp, tmp_path):
        """A purely algebraic diagram has no equilibrium to solve for."""
        b = DiagramBuilder()
        c = b.add_block("Constant", 50, 100, params=_params("Constant", value=1.0))
        g = b.add_block("Gain", 200, 100, params=_params("Gain", gain=2.0))
        sc = b.add_block("Scope", 350, 100, params=_params("Scope"))
        b.connect(c, 0, g, 0)
        b.connect(g, 0, sc, 0)
        dsim = _load(b, tmp_path, "trim_nostate.diablos")

        res = AnalysisController(dsim).find_trim()
        # No continuous states: ok=False with a clean message, no raise.
        assert res["ok"] is False
        assert res["states"] == []
        assert isinstance(res["summary"], str) and res["summary"].strip()

    def test_uncompilable_returns_not_ok(self, qapp, tmp_path):
        """An interpreter-only diagram (Noise) cannot be trimmed; no raise."""
        b = DiagramBuilder()
        n = b.add_block("Noise", 50, 100, params=_params("Noise"))
        i = b.add_block("Integrator", 200, 100, params=_params("Integrator"))
        sc = b.add_block("Scope", 350, 100, params=_params("Scope"))
        b.connect(n, 0, i, 0)
        b.connect(i, 0, sc, 0)
        dsim = _load(b, tmp_path, "trim_noise.diablos")

        res = AnalysisController(dsim).find_trim()
        assert res["ok"] is False
        assert isinstance(res["error"], str) and res["error"].strip()
