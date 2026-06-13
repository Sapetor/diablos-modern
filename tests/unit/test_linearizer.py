"""
Tests for numerical linearization (lib/analysis/linearizer.py).

Builds small diagrams with known dynamics via DiagramBuilder, loads them into a
DSim, and checks that Linearizer.linearize_at_point recovers the correct A/B/C/D
(for linear systems) and the correct local Jacobian (for a nonlinear system),
by finite-differencing the compiled ODE right-hand side.
"""

import numpy as np
import pytest

from lib.diagram_builder import DiagramBuilder
from lib.analysis.linearizer import Linearizer


_BLOCK_INSTANCES = None


def _params(block_type, **overrides):
    """Full flat params for a block (class defaults + overrides).

    DiagramBuilder stores only the params passed in, but the engine's
    initialize_execution runs every block once, so blocks need their complete
    default param set (e.g. Noise needs 'sigma'). Pull defaults from the block
    class's params property and apply overrides.
    """
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
class TestLinearizer:
    def test_integrator_recovers_A0_B1_C1_D0(self, qapp, tmp_path):
        """Constant -> Integrator: dx/dt = u, y = x  =>  A=[[0]], B=[[1]], C=[[1]], D=[[0]]."""
        b = DiagramBuilder()
        c = b.add_block("Constant", 50, 100, params=_params("Constant", value=1.0))
        i = b.add_block("Integrator", 200, 100, params=_params("Integrator", init_conds=0.0))
        sc = b.add_block("Scope", 350, 100, params=_params("Scope"))
        b.connect(c, 0, i, 0)
        b.connect(i, 0, sc, 0)  # consume the output (avoid dangling-port integrity error)
        dsim = _load(b, tmp_path, "integ.diablos")

        lin = Linearizer(dsim)
        src = _name_of(dsim, "Constant")
        integ = _name_of(dsim, "Integrator")
        res = lin.linearize_at_point(input_blocks=[src], output_blocks=[integ])

        assert res is not None
        assert np.allclose(res["A"], [[0.0]], atol=1e-6)
        assert np.allclose(res["B"], [[1.0]], atol=1e-6)
        assert np.allclose(res["C"], [[1.0]], atol=1e-6)
        assert np.allclose(res["D"], [[0.0]], atol=1e-6)
        assert res["n_states"] == 1

    def test_gain_feedback_recovers_negative_gain(self, qapp, tmp_path):
        """Integrator <- Gain(-3) <- Integrator: dx/dt = -3x  =>  A=[[-3]], eig=-3, stable."""
        b = DiagramBuilder()
        i = b.add_block("Integrator", 200, 100, params=_params("Integrator", init_conds=1.0))
        g = b.add_block("Gain", 350, 100, params=_params("Gain", gain=-3.0))
        b.connect(i, 0, g, 0)
        b.connect(g, 0, i, 0)
        dsim = _load(b, tmp_path, "gainfb.diablos")

        lin = Linearizer(dsim)
        res = lin.linearize_at_point()  # A-only path (no I/O designated)

        assert res is not None
        assert np.allclose(res["A"], [[-3.0]], atol=1e-6)
        assert np.allclose(np.sort(np.real(res["eigenvalues"])), [-3.0], atol=1e-6)
        assert bool(res["is_stable"])
        # No I/O designated -> B/C/D absent.
        assert "B" not in res
        # Time constant tau = 1/3.
        assert np.allclose(sorted(res["time_constants"]), [1.0 / 3.0], atol=1e-6)

    def test_nonlinear_xdot_minus_x_squared(self, qapp, tmp_path):
        """Integrator <- Gain(-1) <- Product(x,x): dx/dt = -x^2  =>  A = -2*x0.

        At the operating point x0 = 2, the local Jacobian is A = -4. This proves
        linearization works for a genuinely NONLINEAR diagram, not just linear blocks.
        """
        b = DiagramBuilder()
        i = b.add_block("Integrator", 300, 100, params=_params("Integrator", init_conds=2.0))
        p = b.add_block("Product", 200, 100, params=_params("Product", ops="**"), in_ports=2)
        g = b.add_block("Gain", 100, 100, params=_params("Gain", gain=-1.0))
        # x -> both product inputs; product (x*x) -> gain(-1) -> integrator
        b.connect(i, 0, p, 0)
        b.connect(i, 0, p, 1)
        b.connect(p, 0, g, 0)
        b.connect(g, 0, i, 0)
        dsim = _load(b, tmp_path, "nonlin.diablos")

        lin = Linearizer(dsim)
        res = lin.linearize_at_point()  # operating point = compiled y0 = [2.0]

        assert res is not None
        assert np.allclose(res["operating_point"], [2.0], atol=1e-9)
        # d(-x^2)/dx at x=2 is -4.
        assert np.allclose(res["A"], [[-4.0]], atol=1e-4)
        assert bool(res["is_stable"])  # eigenvalue -4 < 0

    def test_operating_point_override_changes_jacobian(self, qapp, tmp_path):
        """The same nonlinear system linearized at x0=3 gives A = -2*3 = -6."""
        b = DiagramBuilder()
        i = b.add_block("Integrator", 300, 100, params=_params("Integrator", init_conds=2.0))
        p = b.add_block("Product", 200, 100, params=_params("Product", ops="**"), in_ports=2)
        g = b.add_block("Gain", 100, 100, params=_params("Gain", gain=-1.0))
        b.connect(i, 0, p, 0)
        b.connect(i, 0, p, 1)
        b.connect(p, 0, g, 0)
        b.connect(g, 0, i, 0)
        dsim = _load(b, tmp_path, "nonlin2.diablos")

        lin = Linearizer(dsim)
        integ = _name_of(dsim, "Integrator")
        res = lin.linearize_at_point(operating_point={integ: 3.0})
        assert np.allclose(res["A"], [[-6.0]], atol=1e-4)

    def test_find_operating_point_equilibrium(self, qapp, tmp_path):
        """x' = -3x has its equilibrium at x = 0; the trim finder should locate it,
        and linearizing there reproduces A = [[-3]]."""
        b = DiagramBuilder()
        i = b.add_block("Integrator", 200, 100, params=_params("Integrator", init_conds=5.0))
        g = b.add_block("Gain", 350, 100, params=_params("Gain", gain=-3.0))
        b.connect(i, 0, g, 0)
        b.connect(g, 0, i, 0)
        dsim = _load(b, tmp_path, "trim.diablos")

        lin = Linearizer(dsim)
        trim = lin.find_operating_point()
        assert trim["success"]
        assert np.allclose(trim["y"], [0.0], atol=1e-6)
        # The returned operating_point plugs straight into linearize_at_point.
        res = lin.linearize_at_point(operating_point=trim["operating_point"])
        assert np.allclose(res["A"], [[-3.0]], atol=1e-6)

    def test_uncompilable_diagram_raises(self, qapp, tmp_path):
        """A diagram with an interpreter-only block (Noise) cannot be linearized."""
        b = DiagramBuilder()
        n = b.add_block("Noise", 50, 100, params=_params("Noise"))
        s = b.add_block("Scope", 200, 100, params=_params("Scope"))
        b.connect(n, 0, s, 0)
        dsim = _load(b, tmp_path, "noisy.diablos")

        lin = Linearizer(dsim)
        with pytest.raises(ValueError, match="not compilable"):
            lin.linearize_at_point()
