"""Unit tests for the PDE Phase 1 features: periodic BCs, 2D Robin BCs,
dynamic (input-driven) Robin coefficients, and the extra initial conditions.

These exercise the shared operators in ``lib/engine/pde_ops.py`` and the block
wrappers directly. The compiled-vs-interpreted equivalence of the same features
is covered separately in ``tests/regression/test_equiv_pde_phase1.py``.
"""

import numpy as np
import pytest

from lib.engine.pde_helpers import (
    companion_seed,
    parse_pde_2d_initial_condition,
    parse_pde_initial_condition,
)
from lib.engine.pde_ops import (
    heat_rhs_1d,
    heat_rhs_2d,
    is_periodic,
    wave_rhs_1d,
    wave_rhs_2d,
)

pytestmark = pytest.mark.unit


def _heat1d(T, bc_l, bc_r, alpha=1.0, dx=0.1, q=None, h_l=0.0, h_r=0.0, k=1.0, mode="hold"):
    q_src = np.zeros(len(T)) if q is None else q
    return heat_rhs_1d(T, alpha, dx, q_src, bc_l, 0.0, bc_r, 0.0, h_l, h_r, k, boundary_mode=mode)


def _heat2d(T, bc, *, alpha=1.0, dx=0.2, dy=0.2, q=0.0, vals=(0.0,) * 4, h=(0.0,) * 4, k=1.0):
    """heat_rhs_2d with the edge arguments grouped, in edge order l/r/b/t.

    The operator takes 19 positional arguments; naming the groups here keeps the
    call sites readable and makes an accidentally transposed edge obvious.
    """
    return heat_rhs_2d(T, alpha, dx, dy, q, *bc, *vals, *h, k)


def _wave2d(u, v, bc, *, c=1.0, damping=0.0, dx=0.1, dy=0.1, f=0.0, vals=(0.0,) * 4):
    """wave_rhs_2d with the edge arguments grouped, in edge order l/r/b/t."""
    return wave_rhs_2d(u, v, c, damping, dx, dy, f, *bc, *vals)


class TestPeriodicAxisSelection:
    """'Periodic' on either end of an axis wraps that axis."""

    @pytest.mark.parametrize(
        "a,b,expected",
        [
            ("Periodic", "Periodic", True),
            ("Periodic", "Dirichlet", True),
            ("Neumann", "Periodic", True),
            ("Dirichlet", "Neumann", False),
            ("Robin", "Robin", False),
        ],
    )
    def test_either_end_wraps_the_axis(self, a, b, expected):
        assert is_periodic(a, b) is expected

    def test_one_ended_periodic_matches_two_ended(self):
        """Setting only one end to 'Periodic' gives the same RHS as setting both,
        i.e. the opposite edge's BC type is genuinely ignored."""
        T = np.array([1.0, 2.0, 5.0, 3.0, 0.5])
        both = _heat1d(T, "Periodic", "Periodic")
        left_only = _heat1d(T, "Periodic", "Dirichlet")
        right_only = _heat1d(T, "Neumann", "Periodic")
        assert np.array_equal(both, left_only)
        assert np.array_equal(both, right_only)


class TestPeriodicHeat1D:
    def test_wraps_end_nodes_to_each_other(self):
        """Node 0's left neighbour is node N-1 and vice versa."""
        T = np.array([1.0, 2.0, 5.0, 3.0, 0.5])
        dx, alpha = 0.1, 1.0
        dT = _heat1d(T, "Periodic", "Periodic", alpha=alpha, dx=dx)
        assert np.isclose(dT[0], alpha * (T[1] - 2 * T[0] + T[-1]) / dx**2)
        assert np.isclose(dT[-1], alpha * (T[0] - 2 * T[-1] + T[-2]) / dx**2)

    def test_total_heat_is_conserved(self):
        """A periodic ring with no source has an exactly conservative RHS: the
        circulant Laplacian's columns sum to zero, so sum(dT/dt) == 0 for ANY
        field. This is the invariant a Dirichlet or Neumann rod does not have.
        """
        rng = np.random.default_rng(1234)
        for _ in range(5):
            T = rng.normal(size=17)
            dT = _heat1d(T, "Periodic", "Periodic", alpha=0.7, dx=0.05)
            assert abs(np.sum(dT)) < 1e-9

    def test_uniform_field_is_a_steady_state(self):
        T = np.full(9, 3.5)
        assert np.allclose(_heat1d(T, "Periodic", "Periodic"), 0.0)

    def test_both_boundary_modes_agree(self):
        """Periodic needs no penalty/hold distinction, so both execution paths'
        boundary modes must produce identical derivatives."""
        T = np.array([1.0, 2.0, 5.0, 3.0, 0.5])
        hold = _heat1d(T, "Periodic", "Periodic", mode="hold")
        penalty = _heat1d(T, "Periodic", "Periodic", mode="penalty")
        assert np.array_equal(hold, penalty)


class TestPeriodicWave1D:
    def test_wraps_end_nodes(self):
        u = np.array([0.0, 1.0, 3.0, 2.0, -1.0])
        v = np.zeros(5)
        c, dx = 2.0, 0.25
        du, dv = wave_rhs_1d(u, v, c, 0.0, dx, np.zeros(5), "Periodic", 0.0, "Periodic", 0.0)
        assert np.allclose(du, v)
        assert np.isclose(dv[0], c**2 * (u[1] - 2 * u[0] + u[-1]) / dx**2)
        assert np.isclose(dv[-1], c**2 * (u[0] - 2 * u[-1] + u[-2]) / dx**2)

    def test_momentum_is_conserved(self):
        """sum(dv/dt) == 0 on an undamped, unforced ring (wrapped Laplacian)."""
        rng = np.random.default_rng(7)
        u = rng.normal(size=13)
        _, dv = wave_rhs_1d(
            u, np.zeros(13), 1.5, 0.0, 0.08, np.zeros(13), "Periodic", 0.0, "Periodic", 0.0
        )
        assert abs(np.sum(dv)) < 1e-9

    def test_pulse_travels_around_the_ring(self):
        """Integrate a d'Alembert-style right-moving pulse; after wrapping past
        the right end it must reappear on the left rather than reflect."""
        from scipy.integrate import solve_ivp

        N, L, c = 200, 1.0, 1.0
        dx = L / (N - 1)
        x = np.linspace(0, L, N)
        # Right-moving pulse: u(x,0) = f(x), v(x,0) = -c f'(x)
        f = np.exp(-400 * (x - 0.3) ** 2)
        df = -800 * (x - 0.3) * f
        y0 = np.concatenate([f, -c * df])
        force = np.zeros(N)

        def rhs(t, y):
            du, dv = wave_rhs_1d(y[:N], y[N:], c, 0.0, dx, force, "Periodic", 0.0, "Periodic", 0.0)
            return np.concatenate([du, dv])

        # Travel 0.9 of the ring circumference (N*dx, not L).
        t_end = 0.9 * (N * dx) / c
        sol = solve_ivp(rhs, (0, t_end), y0, rtol=1e-8, atol=1e-10)
        u_end = sol.y[:N, -1]

        peak_start = int(np.argmax(f))
        peak_end = int(np.argmax(u_end))
        expected = (peak_start + round(0.9 * N)) % N
        assert min(abs(peak_end - expected), N - abs(peak_end - expected)) <= 3
        # A reflecting boundary would have inverted the pulse; a periodic one
        # keeps its sign and roughly its height.
        assert u_end[peak_end] > 0.8 * f[peak_start]


class TestPeriodicHeat2D:
    def test_fully_periodic_conserves_total_heat(self):
        rng = np.random.default_rng(99)
        T = rng.normal(size=(7, 9))
        dT = _heat2d(T, ("Periodic",) * 4, alpha=0.3, dx=0.11, dy=0.07)
        assert abs(np.sum(dT)) < 1e-9

    def test_axes_are_independent(self):
        """x-periodic with Dirichlet top/bottom (a channel) keeps the Dirichlet
        penalty on the y edges while wrapping x."""
        T = np.zeros((5, 6)) + 2.0
        dT = _heat2d(T, ("Periodic", "Periodic", "Dirichlet", "Dirichlet"))
        # Bottom/top rows driven toward bc=0 by the penalty.
        assert np.all(dT[0, :] < -1000.0)
        assert np.all(dT[-1, :] < -1000.0)
        # Interior rows are unaffected in x (uniform field wraps to zero) but
        # feel the cold rows above/below only at rows adjacent to them.
        assert np.allclose(dT[2, :], 0.0)

    def test_x_periodic_column_uses_wrapped_neighbour(self):
        T = np.arange(12, dtype=float).reshape(3, 4)
        dx = 0.5
        dT = _heat2d(T, ("Periodic", "Periodic", "Neumann", "Neumann"), dx=dx, dy=1.0)
        j = 1  # an interior row, so the y-stencil is the plain central one
        expected_xx = (T[j, 1] - 2 * T[j, 0] + T[j, -1]) / dx**2
        expected_yy = T[j + 1, 0] - 2 * T[j, 0] + T[j - 1, 0]
        assert np.isclose(dT[j, 0], expected_xx + expected_yy)


class TestPeriodicWave2D:
    def test_fully_periodic_conserves_momentum(self):
        rng = np.random.default_rng(5)
        u = rng.normal(size=(6, 8))
        v = np.zeros((6, 8))
        _, dv = _wave2d(u, v, ("Periodic",) * 4, dy=0.2)
        assert abs(np.sum(dv)) < 1e-9

    def test_du_dt_is_still_velocity(self):
        u = np.ones((4, 4))
        v = np.arange(16, dtype=float).reshape(4, 4)
        du, _ = _wave2d(u, v, ("Periodic", "Periodic", "Neumann", "Neumann"))
        assert np.array_equal(du, v)


class TestRobin2D:
    """Per-edge Robin BCs on HeatEquation2D, extended from the 1D block."""

    @staticmethod
    def _steady_state(bc_types, ambient, h, alpha=0.5, n=7, t_end=400.0):
        """Relax a uniformly hot plate to steady state.

        Each edge's ``bc_*`` value is the ambient temperature on a Robin edge
        and a zero normal flux (insulated) on a Neumann one -- passing ``ambient``
        to a Neumann edge would prescribe a non-zero flux and drive a permanent
        gradient instead.
        """
        from scipy.integrate import solve_ivp

        T0 = np.full((n, n), 10.0)
        dx = dy = 1.0 / (n - 1)
        bc_vals = [ambient if bc == "Robin" else 0.0 for bc in bc_types]

        def rhs(t, y):
            return _heat2d(
                y.reshape((n, n)),
                bc_types,
                alpha=alpha,
                dx=dx,
                dy=dy,
                vals=bc_vals,
                h=(h,) * 4,
            ).flatten()

        sol = solve_ivp(rhs, (0, t_end), T0.flatten(), rtol=1e-8, atol=1e-10)
        return sol.y[:, -1].reshape((n, n))

    def test_steady_state_approaches_ambient(self):
        """A plate cooled by Robin BCs on all four edges relaxes to the ambient
        temperature, with no source. This is the defining Robin behaviour and it
        also pins the outward-normal sign convention: a wrong sign on any edge
        would heat that edge away instead of toward ambient."""
        ambient = 2.0
        T = self._steady_state(("Robin",) * 4, ambient, h=5.0)
        assert np.allclose(T, ambient, atol=1e-2), f"max dev {np.max(np.abs(T - ambient))}"

    def test_all_four_edges_cool(self):
        """Every edge must move toward ambient -- catches a sign flip on any one
        of them, which a plate-average assertion alone would hide."""
        T0 = np.full((6, 6), 10.0)
        dT = _heat2d(T0, ("Robin",) * 4, h=(4.0,) * 4)
        # Interior stays put (uniform field), all four edge lines cool.
        assert np.all(dT[1:-1, 0] < 0), "left edge is not cooling"
        assert np.all(dT[1:-1, -1] < 0), "right edge is not cooling"
        assert np.all(dT[0, 1:-1] < 0), "bottom edge is not cooling"
        assert np.all(dT[-1, 1:-1] < 0), "top edge is not cooling"

    def test_per_edge_h_is_independent(self):
        """A large h on one edge cools it faster than a small h on another."""
        T0 = np.full((6, 6), 10.0)
        dT = _heat2d(T0, ("Robin", "Robin", "Neumann", "Neumann"), h=(50.0, 0.5, 0.0, 0.0))
        assert dT[3, 0] < dT[3, -1] < 0

    def test_h_zero_degenerates_to_insulated(self):
        """h = 0 removes the convective flux, so a Robin edge becomes the same
        zero-flux Neumann edge."""
        rng = np.random.default_rng(3)
        T = rng.normal(size=(6, 7))
        robin = _heat2d(T, ("Robin",) * 4, dy=0.3, vals=(5.0,) * 4, h=(0.0,) * 4)
        neumann = _heat2d(T, ("Neumann",) * 4, dy=0.3)
        assert np.allclose(robin, neumann)

    def test_robin_matches_1d_steady_state(self):
        """A 2D plate with Robin left/right and insulated top/bottom must reach
        the same steady state as the 1D block's Robin formulation (which uses
        the penalty form), confirming the two agree where it matters."""
        ambient = 3.0
        T = self._steady_state(
            ("Robin", "Robin", "Neumann", "Neumann"), ambient, h=5.0, t_end=600.0
        )
        assert np.allclose(T, ambient, atol=1e-2)


class TestDynamicRobinCoefficients:
    """h fed from an input port instead of a static param."""

    def test_block_reads_h_from_input_port(self):
        from blocks.pde.heat_equation_1d import HeatEquation1DBlock

        block = HeatEquation1DBlock()
        params = {"h_left": 1.0, "h_right": 1.0, "k_thermal": 1.0}
        # Port unconnected (None) -> param value.
        assert block._robin_coeffs(params, None, None) == (1.0, 1.0)
        # Port connected -> port value wins, arrays coerced to scalars.
        assert block._robin_coeffs(params, np.array([42.0]), 7.0) == (42.0, 7.0)

    def test_changing_h_changes_the_derivative(self):
        """A larger convective coefficient must pull the boundary node harder
        toward ambient -- the whole point of a time-varying h."""
        T = np.full(6, 10.0)
        weak = _heat1d(T, "Robin", "Neumann", h_l=0.1, mode="penalty")
        strong = _heat1d(T, "Robin", "Neumann", h_l=100.0, mode="penalty")
        assert strong[0] < weak[0] < 0

    def test_2d_block_reads_h_from_input_ports(self):
        from blocks.pde.heat_equation_2d import HeatEquation2DBlock

        block = HeatEquation2DBlock()
        params = {"h_left": 1.0, "h_right": 2.0, "h_bottom": 3.0, "h_top": 4.0}
        assert block._robin_coeffs(params, {}) == [1.0, 2.0, 3.0, 4.0]
        # Ports 5..8 are h_left, h_right, h_bottom, h_top.
        assert block._robin_coeffs(params, {5: np.array([9.0]), 7: 8.0}) == [9.0, 2.0, 8.0, 4.0]

    def test_2d_dynamic_ambient_moves_the_field(self):
        """Raising the ambient temperature mid-run reverses the sign of the edge
        derivative: the plate stops cooling and starts warming."""
        from blocks.pde.heat_equation_2d import HeatEquation2DBlock

        block = HeatEquation2DBlock()
        params = {
            "Nx": 6,
            "Ny": 6,
            "Lx": 1.0,
            "Ly": 1.0,
            "alpha": 1.0,
            "bc_type_left": "Robin",
            "bc_type_right": "Robin",
            "bc_type_bottom": "Robin",
            "bc_type_top": "Robin",
            "h_left": 5.0,
            "h_right": 5.0,
            "h_bottom": 5.0,
            "h_top": 5.0,
            "k_thermal": 1.0,
        }
        state = np.full(36, 10.0)
        cold = block.compute_derivatives(0.0, state, {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}, params)
        hot = block.compute_derivatives(1.0, state, {1: 50.0, 2: 50.0, 3: 50.0, 4: 50.0}, params)
        assert cold.reshape(6, 6)[3, 0] < 0 < hot.reshape(6, 6)[3, 0]


class TestNewInitialConditions:
    @pytest.mark.parametrize("name", ["linear", "step", "random", "uniform", "sine", "gaussian"])
    def test_1d_patterns_produce_finite_fields(self, name):
        ic = parse_pde_initial_condition(name, 21, 2.0, seed=5)
        assert ic.shape == (21,)
        assert np.all(np.isfinite(ic))

    def test_1d_linear_ramps_down(self):
        ic = parse_pde_initial_condition("linear", 11, 1.0)
        assert np.isclose(ic[0], 1.0) and np.isclose(ic[-1], 0.0)
        assert np.all(np.diff(ic) < 0)

    def test_1d_step_is_binary(self):
        ic = parse_pde_initial_condition("step", 21, 1.0)
        assert set(np.unique(ic)) == {0.0, 1.0}
        assert ic[0] == 1.0 and ic[-1] == 0.0

    def test_1d_random_is_reproducible_from_seed(self):
        a = parse_pde_initial_condition("random", 25, 1.0, seed=1234)
        b = parse_pde_initial_condition("random", 25, 1.0, seed=1234)
        c = parse_pde_initial_condition("random", 25, 1.0, seed=4321)
        assert np.array_equal(a, b), "same seed must give the same field"
        assert not np.array_equal(a, c), "different seeds must differ"
        assert np.all((a >= 0.0) & (a < 1.0))

    def test_1d_random_seed_zero_is_not_reproducible(self):
        """seed == 0 means entropy, matching the blocks/noise.py convention."""
        a = parse_pde_initial_condition("random", 64, 1.0, seed=0)
        b = parse_pde_initial_condition("random", 64, 1.0, seed=0)
        assert not np.array_equal(a, b)

    @pytest.mark.parametrize(
        "name", ["linear", "step", "random", "checkerboard", "radial", "gaussian", "sinusoidal"]
    )
    def test_2d_patterns_produce_finite_fields(self, name):
        ic = parse_pde_2d_initial_condition(name, 7, 5, seed=9)
        assert ic.shape == (5, 7)
        assert np.all(np.isfinite(ic))

    def test_2d_checkerboard_alternates(self):
        ic = parse_pde_2d_initial_condition("checkerboard", 4, 4, amplitude=2.0)
        assert set(np.unique(ic)) == {-2.0, 2.0}
        # Every 4-neighbour has the opposite sign.
        assert np.all(ic[:, :-1] == -ic[:, 1:])
        assert np.all(ic[:-1, :] == -ic[1:, :])

    def test_2d_random_is_reproducible_from_seed(self):
        a = parse_pde_2d_initial_condition("random", 6, 5, seed=77)
        b = parse_pde_2d_initial_condition("random", 6, 5, seed=77)
        c = parse_pde_2d_initial_condition("random", 6, 5, seed=78)
        assert np.array_equal(a, b)
        assert not np.array_equal(a, c)

    def test_2d_random_scales_with_amplitude(self):
        a = parse_pde_2d_initial_condition("random", 8, 8, amplitude=1.0, seed=3)
        b = parse_pde_2d_initial_condition("random", 8, 8, amplitude=10.0, seed=3)
        assert np.allclose(b, 10.0 * a)


class TestBlockInitialConditionDelegation:
    """The blocks must build ICs through the shared parsers, so the interpreter
    and the compiled path (which call the same helpers) cannot drift apart."""

    def test_heat_1d_block_matches_parser(self):
        from blocks.pde.heat_equation_1d import HeatEquation1DBlock

        params = {"N": 17, "L": 2.0, "init_conds": "random", "seed": 42}
        block_ic = HeatEquation1DBlock().get_initial_conditions(params)
        parser_ic = parse_pde_initial_condition("random", 17, 2.0, pde_type="heat", seed=42)
        assert np.array_equal(block_ic, parser_ic)

    def test_heat_2d_block_matches_parser(self):
        from blocks.pde.heat_equation_2d import HeatEquation2DBlock

        params = {"Nx": 6, "Ny": 4, "init_temp": "random", "seed": 11, "init_amplitude": 3.0}
        block_ic = HeatEquation2DBlock().get_initial_state(params)
        parser_ic = parse_pde_2d_initial_condition("random", 6, 4, 1.0, 1.0, 3.0, seed=11)
        assert np.array_equal(block_ic, parser_ic.flatten())

    def test_wave_1d_random_u_and_v_are_independent(self):
        """One seed drives both fields; the companion offset keeps them from
        being the identical array."""
        from blocks.pde.wave_equation_1d import WaveEquation1DBlock

        params = {
            "N": 20,
            "L": 1.0,
            "init_displacement": "random",
            "init_velocity": "random",
            "seed": 8,
        }
        state = WaveEquation1DBlock().get_initial_conditions(params)
        u0, v0 = state[:20], state[20:]
        assert not np.array_equal(u0, v0)
        assert np.array_equal(u0, parse_pde_initial_condition("random", 20, 1.0, "wave", seed=8))
        assert np.array_equal(
            v0, parse_pde_initial_condition("random", 20, 1.0, "wave", seed=companion_seed(8))
        )

    def test_wave_2d_checkerboard_reaches_the_block(self):
        from blocks.pde.wave_equation_2d import WaveEquation2DBlock

        params = {"Nx": 5, "Ny": 5, "init_displacement": "checkerboard", "init_amplitude": 1.0}
        state = WaveEquation2DBlock().get_initial_state(params)
        u0 = state[:25].reshape(5, 5)
        assert set(np.unique(u0)) == {-1.0, 1.0}

    def test_companion_seed_preserves_the_entropy_convention(self):
        assert companion_seed(0) == 0
        assert companion_seed(5) == 6
        assert companion_seed("not a number") == 0
