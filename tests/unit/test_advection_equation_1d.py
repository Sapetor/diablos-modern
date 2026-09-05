"""Unit tests for the 1D advection block (blocks/pde/advection_equation_1d.py).

Solves dc/dt + v dc/dx = 0 by method of lines with upwind differencing and an
explicit RK4 step. The block had no test at all, so the properties pinned here
are the ones a numerical regression would break first:

* the initial-condition builder (scalar / list / resampled list / the named
  'gaussian', 'step', 'sine' profiles);
* mass conservation under a periodic BC -- the defining property of pure
  transport, and the thing an upwind sign error destroys;
* transport *direction*: a positive velocity moves the profile to larger x,
  a negative velocity to smaller x;
* the Dirichlet inlet being pinned to the connected input, on whichever end the
  flow enters;
* ``output_only`` reporting the current field without advancing time.
"""

import numpy as np
import pytest

from blocks.pde.advection_equation_1d import AdvectionEquation1DBlock


def _defaults(**overrides):
    params = {}
    for name, spec in AdvectionEquation1DBlock().params.items():
        params[name] = spec["default"]
    params.update(overrides)
    return params


def _step(block, params, steps, c_inlet=0.0, dtime=None, t0=0.0):
    """Advance ``steps`` timesteps; return the final (field, total) pair."""
    if dtime is not None:
        params["dtime"] = dtime
    dt = float(params.get("dtime", 0.01))
    result = None
    for i in range(steps):
        result = block.execute(
            time=t0 + i * dt,
            inputs={0: np.array([float(c_inlet)])},
            params=params,
            dtime=dt,
        )
        assert result["E"] is False
    return np.asarray(result[0], dtype=float), float(result[1])


@pytest.mark.unit
class TestAdvection1DContract:
    def test_identity(self):
        block = AdvectionEquation1DBlock()
        assert block.block_name == "AdvectionEquation1D"
        assert block.category == "PDE"

    def test_ports(self):
        block = AdvectionEquation1DBlock()
        assert [p["name"] for p in block.inputs] == ["c_inlet"]
        assert [p["name"] for p in block.outputs] == ["c_field", "c_total"]
        # The inlet is unused under a periodic BC; c_total is diagnostic only.
        assert 0 in block.optional_inputs
        assert 1 in block.optional_outputs

    def test_params(self):
        params = AdvectionEquation1DBlock().params
        assert {"velocity", "L", "N", "bc_type", "init_conds", "_init_start_"} <= set(params)
        assert params["bc_type"]["default"] == "Dirichlet"
        assert params["_init_start_"]["default"] is True

    def test_num_states_is_the_node_count(self):
        block = AdvectionEquation1DBlock()
        assert block.get_num_states(_defaults(N=37)) == 37


@pytest.mark.unit
class TestAdvection1DInitialConditions:
    def test_scalar_initial_condition_fills_the_domain(self):
        c0 = AdvectionEquation1DBlock().get_initial_conditions(_defaults(N=11, init_conds=2.5))
        assert c0.shape == (11,)
        assert np.allclose(c0, 2.5)

    def test_single_element_list_is_broadcast(self):
        c0 = AdvectionEquation1DBlock().get_initial_conditions(_defaults(N=5, init_conds=[3.0]))
        assert np.allclose(c0, 3.0)

    def test_matching_length_list_is_used_verbatim(self):
        values = [0.0, 1.0, 2.0, 3.0]
        c0 = AdvectionEquation1DBlock().get_initial_conditions(_defaults(N=4, init_conds=values))
        assert c0 == pytest.approx(values)

    def test_mismatched_list_is_resampled_onto_the_grid(self):
        """A saved diagram whose N was changed must still load."""
        c0 = AdvectionEquation1DBlock().get_initial_conditions(
            _defaults(N=5, init_conds=[0.0, 1.0, 2.0])
        )
        assert c0.shape == (5,)
        assert c0 == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.0])

    def test_gaussian_profile_peaks_at_a_quarter_of_the_domain(self):
        L, N = 4.0, 201
        c0 = AdvectionEquation1DBlock().get_initial_conditions(
            _defaults(N=N, L=L, init_conds="gaussian")
        )
        x = np.linspace(0, L, N)
        assert x[int(np.argmax(c0))] == pytest.approx(L / 4, abs=L / N)
        assert c0.max() == pytest.approx(1.0)

    def test_step_profile_is_one_before_the_quarter_point_and_zero_after(self):
        L, N = 4.0, 101
        c0 = AdvectionEquation1DBlock().get_initial_conditions(
            _defaults(N=N, L=L, init_conds="step")
        )
        x = np.linspace(0, L, N)
        assert np.all(c0[x < L / 4] == 1.0)
        assert np.all(c0[x >= L / 4] == 0.0)

    def test_sine_profile_stays_within_zero_and_one(self):
        c0 = AdvectionEquation1DBlock().get_initial_conditions(
            _defaults(N=101, L=1.0, init_conds="sine")
        )
        assert c0.min() >= -1e-12
        assert c0.max() <= 1.0 + 1e-12

    def test_unknown_profile_name_yields_a_zero_field(self):
        c0 = AdvectionEquation1DBlock().get_initial_conditions(
            _defaults(N=8, init_conds="not-a-profile")
        )
        assert np.allclose(c0, 0.0)


@pytest.mark.unit
class TestAdvection1DExecution:
    def test_first_call_seeds_the_state_and_grid_spacing(self):
        block = AdvectionEquation1DBlock()
        params = _defaults(N=21, L=2.0, init_conds="gaussian", bc_type="Periodic")

        field, _ = _step(block, params, steps=1, dtime=1e-4)

        assert params["_init_start_"] is False
        assert params["dx"] == pytest.approx(2.0 / 20)
        assert field.shape == (21,)

    def test_a_degenerate_node_count_is_clamped(self):
        """N=1 would make dx = L/0; the block clamps to two nodes."""
        block = AdvectionEquation1DBlock()
        params = _defaults(N=1, bc_type="Periodic", init_conds=1.0)

        field, _ = _step(block, params, steps=1, dtime=1e-4)

        assert params["N"] == 2
        assert field.shape == (2,)
        assert np.all(np.isfinite(field))

    def test_c_total_is_the_discrete_integral_of_the_field(self):
        block = AdvectionEquation1DBlock()
        params = _defaults(N=11, L=1.0, init_conds=2.0, bc_type="Periodic")

        field, total = _step(block, params, steps=1, dtime=1e-4)

        assert total == pytest.approx(np.sum(field) * params["dx"])

    def test_output_only_reports_the_field_without_advancing_it(self):
        block = AdvectionEquation1DBlock()
        params = _defaults(N=41, L=1.0, init_conds="gaussian", bc_type="Periodic")
        field, _ = _step(block, params, steps=5, dtime=1e-3)

        peeked = block.execute(
            time=1.0, inputs={0: np.array([0.0])}, params=params, output_only=True
        )

        assert np.asarray(peeked[0]) == pytest.approx(field)
        assert np.asarray(params["c"]) == pytest.approx(field), "output_only advanced the state"

    def test_a_uniform_field_is_a_steady_state_under_a_periodic_bc(self):
        """dc/dx = 0 everywhere, so pure transport must leave it untouched."""
        block = AdvectionEquation1DBlock()
        params = _defaults(N=21, L=1.0, velocity=1.0, init_conds=3.0, bc_type="Periodic")

        field, _ = _step(block, params, steps=50, dtime=1e-3)

        assert field == pytest.approx(np.full(21, 3.0), abs=1e-9)

    def test_periodic_transport_conserves_mass(self):
        """The defining property of advection; an upwind sign error breaks it."""
        block = AdvectionEquation1DBlock()
        params = _defaults(N=101, L=2.0, velocity=1.0, init_conds="gaussian", bc_type="Periodic")

        _, total_first = _step(block, params, steps=1, dtime=1e-3)
        _, total_last = _step(block, params, steps=200, dtime=1e-3, t0=1e-3)

        assert total_last == pytest.approx(total_first, rel=1e-3)

    def test_positive_velocity_moves_the_profile_downstream(self):
        block = AdvectionEquation1DBlock()
        params = _defaults(N=201, L=4.0, velocity=1.0, init_conds="gaussian", bc_type="Periodic")
        x = np.linspace(0, 4.0, 201)

        field, _ = _step(block, params, steps=200, dtime=2e-3)  # t = 0.4 s

        centroid = float(np.sum(x * field) / np.sum(field))
        assert centroid == pytest.approx(4.0 / 4 + 0.4, abs=0.1)

    def test_negative_velocity_moves_the_profile_upstream(self):
        block = AdvectionEquation1DBlock()
        params = _defaults(N=201, L=4.0, velocity=-1.0, init_conds="gaussian", bc_type="Periodic")
        x = np.linspace(0, 4.0, 201)

        field, _ = _step(block, params, steps=200, dtime=2e-3)

        centroid = float(np.sum(x * field) / np.sum(field))
        assert centroid == pytest.approx(4.0 / 4 - 0.4, abs=0.1)


@pytest.mark.unit
class TestAdvection1DBoundaryConditions:
    def test_dirichlet_pins_the_left_node_for_a_rightward_flow(self):
        block = AdvectionEquation1DBlock()
        params = _defaults(N=41, L=1.0, velocity=1.0, init_conds=0.0, bc_type="Dirichlet")

        field, _ = _step(block, params, steps=20, c_inlet=5.0, dtime=1e-3)

        assert field[0] == pytest.approx(5.0)

    def test_dirichlet_pins_the_right_node_for_a_leftward_flow(self):
        block = AdvectionEquation1DBlock()
        params = _defaults(N=41, L=1.0, velocity=-1.0, init_conds=0.0, bc_type="Dirichlet")

        field, _ = _step(block, params, steps=20, c_inlet=5.0, dtime=1e-3)

        assert field[-1] == pytest.approx(5.0)

    def test_an_inlet_signal_propagates_into_the_domain(self):
        block = AdvectionEquation1DBlock()
        params = _defaults(N=41, L=1.0, velocity=1.0, init_conds=0.0, bc_type="Dirichlet")

        field, total = _step(block, params, steps=100, c_inlet=1.0, dtime=1e-3)

        assert field[0] == pytest.approx(1.0), "the inlet node is pinned to the input"
        assert total > 0.0, "inlet material never entered the domain"
        # v = 1 m/s for 0.1 s over a 1 m domain: the front is ~10% in.
        assert field[-1] == pytest.approx(0.0, abs=1e-6), "the front crossed too fast"
        # The scheme is not monotone (it overshoots slightly at the front), but
        # it must stay bounded -- a blow-up here means the RK4 step went unstable.
        assert field.max() <= 1.2
        assert field.min() >= -0.2

    def test_apply_boundary_conditions_returns_a_modified_copy(self):
        block = AdvectionEquation1DBlock()
        params = _defaults(N=5, velocity=1.0, bc_type="Dirichlet")
        c = np.zeros(5)

        out = block.apply_boundary_conditions(c, params, {"c_inlet": 7.0})

        assert out[0] == pytest.approx(7.0)
        assert c[0] == pytest.approx(0.0), "the input field must not be mutated"

    def test_periodic_bc_ignores_the_inlet_port(self):
        block = AdvectionEquation1DBlock()
        params = _defaults(N=5, velocity=1.0, bc_type="Periodic")
        c = np.arange(5, dtype=float)

        out = block.apply_boundary_conditions(c, params, {"c_inlet": 99.0})

        assert out == pytest.approx(c)

    def test_compute_derivatives_vanishes_on_a_uniform_field(self):
        block = AdvectionEquation1DBlock()
        params = _defaults(N=21, L=1.0, velocity=1.0, bc_type="Periodic")

        dcdt = block.compute_derivatives(0.0, np.full(21, 2.0), {}, params)

        assert np.asarray(dcdt) == pytest.approx(np.zeros(21), abs=1e-12)

    def test_draw_icon_returns_a_painter_path(self, qapp):
        from PyQt5.QtCore import QRect
        from PyQt5.QtGui import QPainterPath

        path = AdvectionEquation1DBlock().draw_icon(QRect(0, 0, 100, 60))
        assert isinstance(path, QPainterPath)
        assert not path.isEmpty()
