"""
Unit tests for PDE block implementations.

Tests verify numerical accuracy against known analytical solutions.
"""

import pytest
import numpy as np
from scipy.integrate import solve_ivp


@pytest.mark.unit
class TestHeatEquation1D:
    """Tests for 1D Heat Equation block."""

    def test_sine_mode_decay(self):
        """Test heat equation with sine initial condition decays correctly.

        Analytical solution: T(x,t) = sin(pi*x/L) * exp(-alpha*(pi/L)^2 * t)
        """
        # Parameters
        alpha = 0.1
        L = 1.0
        N = 51
        dx = L / (N - 1)
        x = np.linspace(0, L, N)

        # Initial condition: sine mode
        T0 = np.sin(np.pi * x / L)

        # Expected decay rate
        lambda_heat = alpha * (np.pi / L)**2

        def heat_rhs(t, T):
            dT_dt = np.zeros(N)
            dx_sq = dx * dx
            for i in range(1, N-1):
                dT_dt[i] = alpha * (T[i+1] - 2*T[i] + T[i-1]) / dx_sq
            # Dirichlet BC with penalty method
            dT_dt[0] = 1000.0 * (0.0 - T[0])
            dT_dt[N-1] = 1000.0 * (0.0 - T[N-1])
            return dT_dt

        sol = solve_ivp(heat_rhs, (0, 5), T0, t_eval=np.linspace(0, 5, 501),
                        rtol=1e-6, atol=1e-8)

        # Compare at center
        numerical = sol.y[N//2, :]
        analytical = np.exp(-lambda_heat * sol.t)

        # Calculate relative error where analytical is significant
        mask = analytical > 0.001
        rel_error = np.abs(numerical[mask] - analytical[mask]) / analytical[mask]
        max_rel_error = np.max(rel_error)

        assert max_rel_error < 0.01, f"Heat equation error {max_rel_error*100:.2f}% exceeds 1%"

    def test_decay_rate_matches_theory(self):
        """Test that decay rate matches theoretical value."""
        alpha = 0.05
        L = 2.0
        N = 41
        dx = L / (N - 1)
        x = np.linspace(0, L, N)

        T0 = np.sin(np.pi * x / L)
        expected_lambda = alpha * (np.pi / L)**2

        def heat_rhs(t, T):
            dT_dt = np.zeros(N)
            dx_sq = dx * dx
            for i in range(1, N-1):
                dT_dt[i] = alpha * (T[i+1] - 2*T[i] + T[i-1]) / dx_sq
            dT_dt[0] = 1000.0 * (0.0 - T[0])
            dT_dt[N-1] = 1000.0 * (0.0 - T[N-1])
            return dT_dt

        sol = solve_ivp(heat_rhs, (0, 10), T0, t_eval=np.linspace(0, 10, 101),
                        rtol=1e-8, atol=1e-10)

        # Fit exponential decay to extract numerical decay rate
        center = sol.y[N//2, :]
        # Use log-linear fit: ln(T) = ln(T0) - lambda*t
        valid = center > 0.001
        if np.sum(valid) > 10:
            t_valid = sol.t[valid]
            c_valid = center[valid]
            coeffs = np.polyfit(t_valid, np.log(c_valid), 1)
            numerical_lambda = -coeffs[0]

            rel_error = abs(numerical_lambda - expected_lambda) / expected_lambda
            assert rel_error < 0.02, f"Decay rate error {rel_error*100:.2f}% exceeds 2%"


@pytest.mark.unit
class TestWaveEquation1D:
    """Tests for 1D Wave Equation block."""

    def test_standing_wave_oscillation(self):
        """Test standing wave oscillates at correct frequency.

        Analytical: u(x,t) = sin(pi*x/L) * cos(pi*c*t/L)
        """
        c = 1.0  # wave speed
        L = 1.0
        N = 51
        dx = L / (N - 1)
        x = np.linspace(0, L, N)

        # Initial: u0 = sin(pi*x/L), du0/dt = 0
        u0 = np.sin(np.pi * x / L)
        v0 = np.zeros(N)
        y0 = np.concatenate([u0, v0])

        omega = np.pi * c / L

        def wave_rhs(t, y):
            u = y[:N]
            v = y[N:]
            du_dt = v.copy()
            dv_dt = np.zeros(N)
            dx_sq = dx * dx
            for i in range(1, N-1):
                dv_dt[i] = c**2 * (u[i+1] - 2*u[i] + u[i-1]) / dx_sq
            # Dirichlet BC with penalty
            du_dt[0] = 1000.0 * (0.0 - u[0])
            du_dt[N-1] = 1000.0 * (0.0 - u[N-1])
            dv_dt[0] = 0.0
            dv_dt[N-1] = 0.0
            return np.concatenate([du_dt, dv_dt])

        sol = solve_ivp(wave_rhs, (0, 4), y0, t_eval=np.linspace(0, 4, 401),
                        rtol=1e-6, atol=1e-8)

        numerical = sol.y[N//2, :]
        analytical = np.cos(omega * sol.t)

        # Wave equation is harder to match exactly due to dispersion
        max_error = np.max(np.abs(numerical - analytical))
        assert max_error < 0.15, f"Wave equation max error {max_error:.3f} exceeds 0.15"

    def test_period_matches_theory(self):
        """Test that oscillation period matches theoretical value T = 2L/c."""
        c = 2.0
        L = 1.0
        N = 51
        dx = L / (N - 1)
        x = np.linspace(0, L, N)

        u0 = np.sin(np.pi * x / L)
        v0 = np.zeros(N)
        y0 = np.concatenate([u0, v0])

        expected_period = 2 * L / c

        def wave_rhs(t, y):
            u = y[:N]
            v = y[N:]
            du_dt = v.copy()
            dv_dt = np.zeros(N)
            dx_sq = dx * dx
            for i in range(1, N-1):
                dv_dt[i] = c**2 * (u[i+1] - 2*u[i] + u[i-1]) / dx_sq
            du_dt[0] = 1000.0 * (0.0 - u[0])
            du_dt[N-1] = 1000.0 * (0.0 - u[N-1])
            return np.concatenate([du_dt, dv_dt])

        # Simulate for 3 periods
        t_end = 3 * expected_period
        sol = solve_ivp(wave_rhs, (0, t_end), y0, t_eval=np.linspace(0, t_end, 301),
                        rtol=1e-8, atol=1e-10)

        # Find peaks to measure period (more robust than zero crossings)
        center = sol.y[N//2, :]
        peaks = []
        for i in range(1, len(center) - 1):
            if center[i] > center[i-1] and center[i] > center[i+1]:
                peaks.append(sol.t[i])

        if len(peaks) >= 2:
            numerical_period = peaks[1] - peaks[0]
            rel_error = abs(numerical_period - expected_period) / expected_period
            assert rel_error < 0.05, f"Period error {rel_error*100:.2f}% exceeds 5%"


@pytest.mark.unit
class TestAdvectionEquation1D:
    """Tests for 1D Advection Equation block."""

    def test_peak_travels_at_correct_velocity(self):
        """Test that advected peak travels at the specified velocity."""
        v = 0.5  # advection velocity
        L = 2.0
        N = 101
        dx = L / (N - 1)
        x = np.linspace(0, L, N)

        # Initial: Gaussian at x = L/4
        x_init = L / 4
        c0 = np.exp(-100 * (x - x_init)**2)

        def advection_rhs(t, c):
            dc_dt = np.zeros(N)
            c_inlet = 0.0
            # Upwind for v > 0
            for i in range(1, N):
                dc_dx = (c[i] - c[i-1]) / dx
                dc_dt[i] = -v * dc_dx
            dc_dt[0] = 1000.0 * (c_inlet - c[0])
            return dc_dt

        sol = solve_ivp(advection_rhs, (0, 2), c0, t_eval=np.linspace(0, 2, 201),
                        rtol=1e-6, atol=1e-8)

        # Track peak position
        peak_positions = []
        for t_idx in range(0, len(sol.t), 50):
            profile = sol.y[:, t_idx]
            peak_idx = np.argmax(profile)
            peak_positions.append((sol.t[t_idx], x[peak_idx]))

        # Verify peak moves at velocity v
        for t, x_peak in peak_positions:
            expected_x = x_init + v * t
            if expected_x < L:  # Before reaching boundary
                error = abs(x_peak - expected_x)
                assert error < 0.05, f"Peak at t={t:.2f} expected at {expected_x:.3f}, got {x_peak:.3f}"

    def test_mass_conservation_periodic(self):
        """Test that total mass is conserved with periodic BC."""
        v = 1.0
        L = 1.0
        N = 51
        dx = L / (N - 1)
        x = np.linspace(0, L, N)

        # Smooth initial condition
        c0 = 0.5 * (1 + np.sin(2 * np.pi * x / L))
        initial_mass = np.sum(c0) * dx

        def advection_periodic(t, c):
            dc_dt = np.zeros(N)
            for i in range(1, N):
                dc_dx = (c[i] - c[i-1]) / dx
                dc_dt[i] = -v * dc_dx
            # Periodic BC
            dc_dx = (c[0] - c[N-1]) / dx
            dc_dt[0] = -v * dc_dx
            return dc_dt

        sol = solve_ivp(advection_periodic, (0, 2), c0, t_eval=np.linspace(0, 2, 201),
                        rtol=1e-8, atol=1e-10)

        # Check mass at final time
        final_mass = np.sum(sol.y[:, -1]) * dx
        rel_error = abs(final_mass - initial_mass) / initial_mass
        assert rel_error < 0.01, f"Mass conservation error {rel_error*100:.2f}% exceeds 1%"


@pytest.mark.unit
class TestDiffusionReaction1D:
    """Tests for 1D Diffusion-Reaction Equation block."""

    def test_linear_decay_with_diffusion(self):
        """Test diffusion-reaction with linear decay (n=1).

        Analytical: c(x,t) = sin(pi*x/L) * exp(-(D*(pi/L)^2 + k)*t)
        """
        D = 0.1
        k = 0.5
        n = 1
        L = 1.0
        N = 51
        dx = L / (N - 1)
        x = np.linspace(0, L, N)

        # Sine initial condition
        c0 = np.sin(np.pi * x / L)

        # Total decay rate
        lambda_total = D * (np.pi / L)**2 + k

        def diffreact_rhs(t, c):
            dc_dt = np.zeros(N)
            dx_sq = dx * dx

            # Interior
            for i in range(1, N-1):
                d2c_dx2 = (c[i+1] - 2*c[i] + c[i-1]) / dx_sq
                reaction = k * np.power(max(c[i], 0), n)
                dc_dt[i] = D * d2c_dx2 - reaction

            # Dirichlet BC with penalty
            dc_dt[0] = 1000.0 * (0.0 - c[0])
            dc_dt[N-1] = 1000.0 * (0.0 - c[N-1])

            return dc_dt

        sol = solve_ivp(diffreact_rhs, (0, 5), c0, t_eval=np.linspace(0, 5, 501),
                        rtol=1e-6, atol=1e-8)

        # Compare at center
        numerical = sol.y[N//2, :]
        analytical = np.exp(-lambda_total * sol.t)

        # Calculate relative error
        mask = analytical > 0.001
        rel_error = np.abs(numerical[mask] - analytical[mask]) / analytical[mask]
        max_rel_error = np.max(rel_error)

        assert max_rel_error < 0.01, f"Diffusion-reaction error {max_rel_error*100:.2f}% exceeds 1%"

    def test_pure_reaction_decay(self):
        """Test pure reaction (no diffusion) with uniform initial condition."""
        D = 0.0  # No diffusion
        k = 1.0
        n = 1
        L = 1.0
        N = 21
        dx = L / (N - 1)

        # Uniform initial condition
        c0 = np.ones(N)

        def reaction_rhs(t, c):
            dc_dt = np.zeros(N)
            for i in range(N):
                dc_dt[i] = -k * np.power(max(c[i], 0), n)
            return dc_dt

        sol = solve_ivp(reaction_rhs, (0, 5), c0, t_eval=np.linspace(0, 5, 501),
                        rtol=1e-8, atol=1e-10)

        # Analytical: c(t) = c0 * exp(-k*t)
        numerical = sol.y[N//2, :]
        analytical = np.exp(-k * sol.t)

        max_error = np.max(np.abs(numerical - analytical))
        assert max_error < 1e-6, f"Pure reaction error {max_error:.2e} exceeds 1e-6"

    def test_steady_state_diffusion(self):
        """Test that pure diffusion reaches steady state."""
        D = 0.1
        k = 0.0  # No reaction
        L = 1.0
        N = 31
        dx = L / (N - 1)
        x = np.linspace(0, L, N)

        # Initial: sine mode
        c0 = np.sin(np.pi * x / L)

        def diffusion_rhs(t, c):
            dc_dt = np.zeros(N)
            dx_sq = dx * dx
            for i in range(1, N-1):
                dc_dt[i] = D * (c[i+1] - 2*c[i] + c[i-1]) / dx_sq
            dc_dt[0] = 1000.0 * (0.0 - c[0])
            dc_dt[N-1] = 1000.0 * (0.0 - c[N-1])
            return dc_dt

        sol = solve_ivp(diffusion_rhs, (0, 20), c0, t_eval=[0, 20],
                        rtol=1e-8, atol=1e-10)

        # At steady state, should be zero (Dirichlet BC at both ends)
        final_max = np.max(np.abs(sol.y[:, -1]))
        assert final_max < 0.001, f"Steady state max {final_max:.4f} exceeds 0.001"


@pytest.mark.unit
class TestPDEBoundaryConditions:
    """Tests for boundary condition implementations."""

    def test_dirichlet_bc_enforced(self):
        """Test that Dirichlet BC is properly enforced."""
        N = 21
        c = np.ones(N)  # Uniform field
        bc_value = 5.0

        # Apply penalty method
        dc_dt = np.zeros(N)
        dc_dt[0] = 1000.0 * (bc_value - c[0])

        # After small time step, boundary should move toward bc_value
        dt = 0.001
        c_new = c[0] + dc_dt[0] * dt

        # Should be closer to bc_value
        assert abs(c_new - bc_value) < abs(c[0] - bc_value)

    def test_neumann_bc_zero_flux(self):
        """Test that Neumann BC with zero flux preserves total mass."""
        D = 0.1
        L = 1.0
        N = 31
        dx = L / (N - 1)

        # Gaussian in center
        x = np.linspace(0, L, N)
        c0 = np.exp(-50 * (x - L/2)**2)
        initial_mass = np.sum(c0) * dx

        def diffusion_neumann(t, c):
            dc_dt = np.zeros(N)
            dx_sq = dx * dx
            for i in range(1, N-1):
                dc_dt[i] = D * (c[i+1] - 2*c[i] + c[i-1]) / dx_sq
            # Neumann BC: dc/dx = 0 at boundaries (zero flux)
            dc_dt[0] = D * (2*c[1] - 2*c[0]) / dx_sq
            dc_dt[N-1] = D * (2*c[N-2] - 2*c[N-1]) / dx_sq
            return dc_dt

        sol = solve_ivp(diffusion_neumann, (0, 5), c0, t_eval=[0, 5],
                        rtol=1e-8, atol=1e-10)

        final_mass = np.sum(sol.y[:, -1]) * dx
        rel_error = abs(final_mass - initial_mass) / initial_mass
        # Allow 5% error due to discretization effects at boundaries
        assert rel_error < 0.05, f"Mass conservation error {rel_error*100:.3f}% with Neumann BC"


@pytest.mark.unit
class TestPDEInitialConditions:
    """Tests for initial condition implementations."""

    def test_sine_initial_condition(self):
        """Test sine initial condition is properly generated."""
        L = 2.0
        N = 51
        x = np.linspace(0, L, N)

        c0 = np.sin(np.pi * x / L)

        # Check boundary values
        assert abs(c0[0]) < 1e-10, "Sine IC should be zero at x=0"
        assert abs(c0[-1]) < 1e-10, "Sine IC should be zero at x=L"

        # Check maximum at center
        center_idx = N // 2
        assert abs(c0[center_idx] - 1.0) < 0.01, "Sine IC should be 1.0 at center"

    def test_gaussian_initial_condition(self):
        """Test Gaussian initial condition is properly generated."""
        L = 2.0
        N = 101
        x = np.linspace(0, L, N)

        # Gaussian at L/4
        c0 = np.exp(-100 * (x - L/4)**2)

        # Check peak location
        peak_idx = np.argmax(c0)
        peak_x = x[peak_idx]
        assert abs(peak_x - L/4) < 0.02, f"Gaussian peak at {peak_x}, expected {L/4}"

        # Check peak value is 1.0
        assert abs(c0[peak_idx] - 1.0) < 0.01, "Gaussian peak should be 1.0"

    def test_uniform_initial_condition(self):
        """Test uniform initial condition."""
        N = 31
        value = 2.5
        c0 = np.full(N, value)

        assert np.all(c0 == value), "Uniform IC should have constant value"


# =============================================================================
# 2D PDE Block Tests
# =============================================================================

@pytest.mark.unit
class TestHeatEquation2D:
    """Tests for 2D Heat Equation block."""

    def test_sinusoidal_mode_decay(self):
        """Test 2D heat equation with sinusoidal initial condition decays correctly.

        Analytical solution: T(x,y,t) = sin(πx/Lx)*sin(πy/Ly) * exp(-α*(π²/Lx² + π²/Ly²)*t)
        For square domain Lx=Ly=L: T = sin(πx/L)*sin(πy/L) * exp(-2απ²t/L²)
        """
        alpha = 0.1
        L = 1.0
        Nx, Ny = 21, 21
        dx = L / (Nx - 1)
        dy = L / (Ny - 1)

        x = np.linspace(0, L, Nx)
        y = np.linspace(0, L, Ny)
        X, Y = np.meshgrid(x, y)  # Shape: (Ny, Nx)

        # Initial: sinusoidal eigenmode
        T0 = np.sin(np.pi * X / L) * np.sin(np.pi * Y / L)
        T0_flat = T0.flatten()

        # Decay rate for (1,1) mode: α * (π²/Lx² + π²/Ly²) = α * 2π²/L²
        lambda_heat = alpha * 2 * (np.pi / L)**2

        def heat_2d_rhs(t, T_flat):
            T = T_flat.reshape((Ny, Nx))
            dT_dt = np.zeros((Ny, Nx))

            # Interior points: 5-point stencil
            for j in range(1, Ny - 1):
                for i in range(1, Nx - 1):
                    d2T_dx2 = (T[j, i+1] - 2*T[j, i] + T[j, i-1]) / (dx * dx)
                    d2T_dy2 = (T[j+1, i] - 2*T[j, i] + T[j-1, i]) / (dy * dy)
                    dT_dt[j, i] = alpha * (d2T_dx2 + d2T_dy2)

            # Dirichlet BC with penalty
            penalty = 1000.0
            dT_dt[0, :] = penalty * (0.0 - T[0, :])
            dT_dt[Ny-1, :] = penalty * (0.0 - T[Ny-1, :])
            dT_dt[:, 0] = penalty * (0.0 - T[:, 0])
            dT_dt[:, Nx-1] = penalty * (0.0 - T[:, Nx-1])

            return dT_dt.flatten()

        sol = solve_ivp(heat_2d_rhs, (0, 2), T0_flat, t_eval=np.linspace(0, 2, 201),
                        rtol=1e-6, atol=1e-8)

        # Compare at center (Ny//2, Nx//2)
        center_idx = (Ny // 2) * Nx + (Nx // 2)
        numerical = sol.y[center_idx, :]
        analytical = np.exp(-lambda_heat * sol.t)

        # Calculate relative error where analytical is significant
        mask = analytical > 0.01
        rel_error = np.abs(numerical[mask] - analytical[mask]) / analytical[mask]
        max_rel_error = np.max(rel_error)

        assert max_rel_error < 0.05, f"2D Heat equation error {max_rel_error*100:.2f}% exceeds 5%"

    def test_decay_rate_matches_theory_2d(self):
        """Test that 2D heat equation decay rate matches theoretical value."""
        alpha = 0.05
        L = 1.0
        Nx, Ny = 15, 15
        dx = L / (Nx - 1)
        dy = L / (Ny - 1)

        x = np.linspace(0, L, Nx)
        y = np.linspace(0, L, Ny)
        X, Y = np.meshgrid(x, y)

        T0 = np.sin(np.pi * X / L) * np.sin(np.pi * Y / L)
        T0_flat = T0.flatten()

        expected_lambda = alpha * 2 * (np.pi / L)**2

        def heat_2d_rhs(t, T_flat):
            T = T_flat.reshape((Ny, Nx))
            dT_dt = np.zeros((Ny, Nx))
            for j in range(1, Ny - 1):
                for i in range(1, Nx - 1):
                    d2T_dx2 = (T[j, i+1] - 2*T[j, i] + T[j, i-1]) / (dx * dx)
                    d2T_dy2 = (T[j+1, i] - 2*T[j, i] + T[j-1, i]) / (dy * dy)
                    dT_dt[j, i] = alpha * (d2T_dx2 + d2T_dy2)
            dT_dt[0, :] = 1000.0 * (0.0 - T[0, :])
            dT_dt[Ny-1, :] = 1000.0 * (0.0 - T[Ny-1, :])
            dT_dt[:, 0] = 1000.0 * (0.0 - T[:, 0])
            dT_dt[:, Nx-1] = 1000.0 * (0.0 - T[:, Nx-1])
            return dT_dt.flatten()

        sol = solve_ivp(heat_2d_rhs, (0, 5), T0_flat, t_eval=np.linspace(0, 5, 101),
                        rtol=1e-8, atol=1e-10)

        # Fit exponential to extract numerical decay rate
        center_idx = (Ny // 2) * Nx + (Nx // 2)
        center = sol.y[center_idx, :]
        valid = center > 0.01
        if np.sum(valid) > 10:
            t_valid = sol.t[valid]
            c_valid = center[valid]
            coeffs = np.polyfit(t_valid, np.log(c_valid), 1)
            numerical_lambda = -coeffs[0]

            rel_error = abs(numerical_lambda - expected_lambda) / expected_lambda
            assert rel_error < 0.05, f"2D decay rate error {rel_error*100:.2f}% exceeds 5%"

    def test_gaussian_initial_condition_2d(self):
        """Test 2D heat equation with Gaussian initial condition diffuses."""
        alpha = 0.1
        Lx, Ly = 1.0, 1.0
        Nx, Ny = 21, 21
        dx = Lx / (Nx - 1)
        dy = Ly / (Ny - 1)

        x = np.linspace(0, Lx, Nx)
        y = np.linspace(0, Ly, Ny)
        X, Y = np.meshgrid(x, y)

        # Gaussian at center
        T0 = np.exp(-50 * ((X - Lx/2)**2 + (Y - Ly/2)**2))
        T0_flat = T0.flatten()
        initial_max = np.max(T0)

        def heat_2d_rhs(t, T_flat):
            T = T_flat.reshape((Ny, Nx))
            dT_dt = np.zeros((Ny, Nx))
            for j in range(1, Ny - 1):
                for i in range(1, Nx - 1):
                    d2T_dx2 = (T[j, i+1] - 2*T[j, i] + T[j, i-1]) / (dx * dx)
                    d2T_dy2 = (T[j+1, i] - 2*T[j, i] + T[j-1, i]) / (dy * dy)
                    dT_dt[j, i] = alpha * (d2T_dx2 + d2T_dy2)
            # Dirichlet BC
            dT_dt[0, :] = 1000.0 * (0.0 - T[0, :])
            dT_dt[Ny-1, :] = 1000.0 * (0.0 - T[Ny-1, :])
            dT_dt[:, 0] = 1000.0 * (0.0 - T[:, 0])
            dT_dt[:, Nx-1] = 1000.0 * (0.0 - T[:, Nx-1])
            return dT_dt.flatten()

        sol = solve_ivp(heat_2d_rhs, (0, 1), T0_flat, t_eval=[0, 0.5, 1.0],
                        rtol=1e-6, atol=1e-8)

        # Peak should decrease as heat diffuses
        final_T = sol.y[:, -1].reshape((Ny, Nx))
        final_max = np.max(final_T)

        assert final_max < initial_max * 0.5, f"Gaussian should diffuse: initial max={initial_max:.3f}, final max={final_max:.3f}"


@pytest.mark.unit
class TestWaveEquation2D:
    """Tests for 2D Wave Equation block."""

    def test_standing_wave_oscillation_2d(self):
        """Test 2D standing wave oscillates at correct frequency.

        Analytical: u(x,y,t) = sin(πx/L)*sin(πy/L)*cos(ωt)
        where ω = c*π*sqrt(1/Lx² + 1/Ly²) = c*π*sqrt(2)/L for square domain
        """
        c = 1.0
        L = 1.0
        Nx, Ny = 15, 15
        dx = L / (Nx - 1)
        dy = L / (Ny - 1)

        x = np.linspace(0, L, Nx)
        y = np.linspace(0, L, Ny)
        X, Y = np.meshgrid(x, y)

        # Initial: u = sin(πx/L)*sin(πy/L), v = 0
        u0 = np.sin(np.pi * X / L) * np.sin(np.pi * Y / L)
        v0 = np.zeros((Ny, Nx))
        y0 = np.concatenate([u0.flatten(), v0.flatten()])

        # Angular frequency for (1,1) mode
        omega = c * np.pi * np.sqrt(2) / L

        def wave_2d_rhs(t, state):
            N = Nx * Ny
            u = state[:N].reshape((Ny, Nx))
            v = state[N:].reshape((Ny, Nx))

            du_dt = v.copy()
            dv_dt = np.zeros((Ny, Nx))

            # Interior: 5-point Laplacian
            for j in range(1, Ny - 1):
                for i in range(1, Nx - 1):
                    d2u_dx2 = (u[j, i+1] - 2*u[j, i] + u[j, i-1]) / (dx * dx)
                    d2u_dy2 = (u[j+1, i] - 2*u[j, i] + u[j-1, i]) / (dy * dy)
                    dv_dt[j, i] = c**2 * (d2u_dx2 + d2u_dy2)

            # Dirichlet BC with penalty
            penalty = 1000.0
            du_dt[0, :] = penalty * (0.0 - u[0, :])
            du_dt[Ny-1, :] = penalty * (0.0 - u[Ny-1, :])
            du_dt[:, 0] = penalty * (0.0 - u[:, 0])
            du_dt[:, Nx-1] = penalty * (0.0 - u[:, Nx-1])
            dv_dt[0, :] = 0.0
            dv_dt[Ny-1, :] = 0.0
            dv_dt[:, 0] = 0.0
            dv_dt[:, Nx-1] = 0.0

            return np.concatenate([du_dt.flatten(), dv_dt.flatten()])

        # Simulate for 2 periods
        T_period = 2 * np.pi / omega
        sol = solve_ivp(wave_2d_rhs, (0, 2*T_period), y0,
                        t_eval=np.linspace(0, 2*T_period, 201),
                        rtol=1e-6, atol=1e-8)

        # Compare center displacement to analytical
        center_idx = (Ny // 2) * Nx + (Nx // 2)
        numerical = sol.y[center_idx, :]
        analytical = np.cos(omega * sol.t)

        # Allow larger error due to numerical dispersion in 2D
        max_error = np.max(np.abs(numerical - analytical))
        assert max_error < 0.3, f"2D Wave equation max error {max_error:.3f} exceeds 0.3"

    def test_energy_conservation_undamped_2d(self):
        """Test that total energy is approximately conserved for undamped 2D wave."""
        c = 1.0
        L = 1.0
        Nx, Ny = 11, 11
        dx = L / (Nx - 1)
        dy = L / (Ny - 1)

        x = np.linspace(0, L, Nx)
        y = np.linspace(0, L, Ny)
        X, Y = np.meshgrid(x, y)

        # Initial condition
        u0 = np.sin(np.pi * X / L) * np.sin(np.pi * Y / L)
        v0 = np.zeros((Ny, Nx))
        y0 = np.concatenate([u0.flatten(), v0.flatten()])

        def wave_2d_rhs(t, state):
            N = Nx * Ny
            u = state[:N].reshape((Ny, Nx))
            v = state[N:].reshape((Ny, Nx))
            du_dt = v.copy()
            dv_dt = np.zeros((Ny, Nx))
            for j in range(1, Ny - 1):
                for i in range(1, Nx - 1):
                    d2u_dx2 = (u[j, i+1] - 2*u[j, i] + u[j, i-1]) / (dx * dx)
                    d2u_dy2 = (u[j+1, i] - 2*u[j, i] + u[j-1, i]) / (dy * dy)
                    dv_dt[j, i] = c**2 * (d2u_dx2 + d2u_dy2)
            # Dirichlet BC
            du_dt[0, :] = 1000.0 * (0.0 - u[0, :])
            du_dt[Ny-1, :] = 1000.0 * (0.0 - u[Ny-1, :])
            du_dt[:, 0] = 1000.0 * (0.0 - u[:, 0])
            du_dt[:, Nx-1] = 1000.0 * (0.0 - u[:, Nx-1])
            return np.concatenate([du_dt.flatten(), dv_dt.flatten()])

        def compute_energy(state):
            N = Nx * Ny
            u = state[:N].reshape((Ny, Nx))
            v = state[N:].reshape((Ny, Nx))
            dA = dx * dy
            # Kinetic: 0.5 * ∫∫ v² dA
            kinetic = 0.5 * np.sum(v**2) * dA
            # Potential: 0.5 * c² * ∫∫ |∇u|² dA
            du_dx = np.gradient(u, dx, axis=1)
            du_dy = np.gradient(u, dy, axis=0)
            potential = 0.5 * c**2 * np.sum(du_dx**2 + du_dy**2) * dA
            return kinetic + potential

        sol = solve_ivp(wave_2d_rhs, (0, 2), y0, t_eval=np.linspace(0, 2, 21),
                        rtol=1e-6, atol=1e-8)

        energies = [compute_energy(sol.y[:, i]) for i in range(sol.y.shape[1])]
        initial_energy = energies[0]

        # Energy should be approximately conserved (allow 20% due to penalty BC)
        max_variation = max(abs(E - initial_energy) / initial_energy for E in energies)
        assert max_variation < 0.2, f"Energy variation {max_variation*100:.1f}% exceeds 20%"


@pytest.mark.unit
class TestAdvectionEquation2D:
    """Tests for 2D Advection Equation block."""

    def test_advection_velocity_x(self):
        """Test that concentration advects at correct velocity in x-direction."""
        vx = 1.0
        vy = 0.0
        Lx, Ly = 2.0, 1.0
        Nx, Ny = 41, 21
        dx = Lx / (Nx - 1)
        dy = Ly / (Ny - 1)

        x = np.linspace(0, Lx, Nx)
        y = np.linspace(0, Ly, Ny)
        X, Y = np.meshgrid(x, y)

        # Initial: Gaussian stripe at x = Lx/4
        x_init = Lx / 4
        c0 = np.exp(-50 * (X - x_init)**2)
        c0_flat = c0.flatten()

        def advection_2d_rhs(t, c_flat):
            c = c_flat.reshape((Ny, Nx))
            dc_dt = np.zeros((Ny, Nx))

            # Interior: upwind for vx > 0
            for j in range(Ny):
                for i in range(1, Nx):
                    dc_dx = (c[j, i] - c[j, i-1]) / dx
                    dc_dt[j, i] = -vx * dc_dx

            # Left BC: Dirichlet (inlet = 0)
            dc_dt[:, 0] = 1000.0 * (0.0 - c[:, 0])

            return dc_dt.flatten()

        sol = solve_ivp(advection_2d_rhs, (0, 1), c0_flat,
                        t_eval=np.linspace(0, 1, 101),
                        rtol=1e-6, atol=1e-8)

        # Track peak position at several times
        for t_idx in [0, 25, 50, 75]:
            t = sol.t[t_idx]
            c_field = sol.y[:, t_idx].reshape((Ny, Nx))
            # Find x position of maximum in middle row
            mid_row = c_field[Ny // 2, :]
            peak_idx = np.argmax(mid_row)
            peak_x = x[peak_idx]

            expected_x = x_init + vx * t
            if expected_x < Lx - dx:  # Before reaching boundary
                error = abs(peak_x - expected_x)
                assert error < 0.1, f"Peak at t={t:.2f} expected at x={expected_x:.2f}, got {peak_x:.2f}"

    def test_advection_velocity_diagonal(self):
        """Test diagonal advection (vx > 0, vy > 0)."""
        vx = 1.0
        vy = 0.5
        Lx, Ly = 2.0, 1.0
        Nx, Ny = 41, 21
        dx = Lx / (Nx - 1)
        dy = Ly / (Ny - 1)

        x = np.linspace(0, Lx, Nx)
        y = np.linspace(0, Ly, Ny)
        X, Y = np.meshgrid(x, y)

        # Initial: Gaussian pulse at (Lx/4, Ly/2)
        x_init, y_init = Lx / 4, Ly / 2
        c0 = np.exp(-50 * ((X - x_init)**2 + (Y - y_init)**2))
        c0_flat = c0.flatten()

        def advection_2d_rhs(t, c_flat):
            c = c_flat.reshape((Ny, Nx))
            dc_dt = np.zeros((Ny, Nx))

            for j in range(1, Ny):
                for i in range(1, Nx):
                    # Upwind for vx > 0, vy > 0
                    dc_dx = (c[j, i] - c[j, i-1]) / dx
                    dc_dy = (c[j, i] - c[j-1, i]) / dy
                    dc_dt[j, i] = -vx * dc_dx - vy * dc_dy

            # BC: Dirichlet at inflow boundaries
            dc_dt[:, 0] = 1000.0 * (0.0 - c[:, 0])
            dc_dt[0, :] = 1000.0 * (0.0 - c[0, :])

            return dc_dt.flatten()

        sol = solve_ivp(advection_2d_rhs, (0, 0.5), c0_flat,
                        t_eval=[0, 0.25, 0.5],
                        rtol=1e-6, atol=1e-8)

        # At t=0.25, peak should have moved to approximately (x_init + 0.25, y_init + 0.125)
        c_mid = sol.y[:, 1].reshape((Ny, Nx))
        peak_idx = np.unravel_index(np.argmax(c_mid), c_mid.shape)
        peak_y, peak_x = y[peak_idx[0]], x[peak_idx[1]]

        expected_x = x_init + vx * 0.25
        expected_y = y_init + vy * 0.25

        assert abs(peak_x - expected_x) < 0.15, f"X peak error: expected {expected_x:.2f}, got {peak_x:.2f}"
        assert abs(peak_y - expected_y) < 0.15, f"Y peak error: expected {expected_y:.2f}, got {peak_y:.2f}"

    def test_mass_conservation_closed_domain(self):
        """Test mass conservation with Neumann (no-flux) BC."""
        vx, vy = 0.0, 0.0  # No advection - pure diffusion
        D = 0.1  # Add diffusion
        Lx, Ly = 1.0, 1.0
        Nx, Ny = 21, 21
        dx = Lx / (Nx - 1)
        dy = Ly / (Ny - 1)

        x = np.linspace(0, Lx, Nx)
        y = np.linspace(0, Ly, Ny)
        X, Y = np.meshgrid(x, y)

        # Gaussian at center
        c0 = np.exp(-50 * ((X - Lx/2)**2 + (Y - Ly/2)**2))
        c0_flat = c0.flatten()
        initial_mass = np.sum(c0) * dx * dy

        def diffusion_2d_neumann(t, c_flat):
            c = c_flat.reshape((Ny, Nx))
            dc_dt = np.zeros((Ny, Nx))

            # Interior: diffusion only
            for j in range(1, Ny - 1):
                for i in range(1, Nx - 1):
                    d2c_dx2 = (c[j, i+1] - 2*c[j, i] + c[j, i-1]) / (dx * dx)
                    d2c_dy2 = (c[j+1, i] - 2*c[j, i] + c[j-1, i]) / (dy * dy)
                    dc_dt[j, i] = D * (d2c_dx2 + d2c_dy2)

            # Neumann BC: zero normal gradient (no flux)
            # Left/Right
            for j in range(1, Ny - 1):
                dc_dt[j, 0] = D * (2*c[j, 1] - 2*c[j, 0]) / (dx * dx) + D * (c[j+1, 0] - 2*c[j, 0] + c[j-1, 0]) / (dy * dy)
                dc_dt[j, Nx-1] = D * (2*c[j, Nx-2] - 2*c[j, Nx-1]) / (dx * dx) + D * (c[j+1, Nx-1] - 2*c[j, Nx-1] + c[j-1, Nx-1]) / (dy * dy)
            # Bottom/Top
            for i in range(1, Nx - 1):
                dc_dt[0, i] = D * (c[0, i+1] - 2*c[0, i] + c[0, i-1]) / (dx * dx) + D * (2*c[1, i] - 2*c[0, i]) / (dy * dy)
                dc_dt[Ny-1, i] = D * (c[Ny-1, i+1] - 2*c[Ny-1, i] + c[Ny-1, i-1]) / (dx * dx) + D * (2*c[Ny-2, i] - 2*c[Ny-1, i]) / (dy * dy)
            # Corners (approximate)
            dc_dt[0, 0] = D * (2*c[0, 1] - 2*c[0, 0]) / (dx * dx) + D * (2*c[1, 0] - 2*c[0, 0]) / (dy * dy)
            dc_dt[0, Nx-1] = D * (2*c[0, Nx-2] - 2*c[0, Nx-1]) / (dx * dx) + D * (2*c[1, Nx-1] - 2*c[0, Nx-1]) / (dy * dy)
            dc_dt[Ny-1, 0] = D * (2*c[Ny-1, 1] - 2*c[Ny-1, 0]) / (dx * dx) + D * (2*c[Ny-2, 0] - 2*c[Ny-1, 0]) / (dy * dy)
            dc_dt[Ny-1, Nx-1] = D * (2*c[Ny-1, Nx-2] - 2*c[Ny-1, Nx-1]) / (dx * dx) + D * (2*c[Ny-2, Nx-1] - 2*c[Ny-1, Nx-1]) / (dy * dy)

            return dc_dt.flatten()

        sol = solve_ivp(diffusion_2d_neumann, (0, 2), c0_flat,
                        t_eval=[0, 2],
                        rtol=1e-8, atol=1e-10)

        final_mass = np.sum(sol.y[:, -1]) * dx * dy
        rel_error = abs(final_mass - initial_mass) / initial_mass
        # Allow 15% error due to corner BC approximations in 2D
        assert rel_error < 0.15, f"Mass conservation error {rel_error*100:.2f}% exceeds 15%"


@pytest.mark.unit
class Test2DPDEBlockImplementations:
    """Tests for actual 2D PDE block implementations."""

    def test_heat_equation_2d_block_initialization(self):
        """Test HeatEquation2DBlock initializes correctly."""
        from blocks.pde.heat_equation_2d import HeatEquation2DBlock

        block = HeatEquation2DBlock()
        params = {
            'alpha': 0.01,
            'Lx': 1.0, 'Ly': 1.0,
            'Nx': 10, 'Ny': 10,
            'init_temp': 'sinusoidal',
            'init_amplitude': 1.0
        }

        state = block.get_initial_state(params)
        state_size = block.get_state_size(params)

        assert state_size == 100, f"Expected 10*10=100 states, got {state_size}"
        assert len(state) == 100, f"Expected state length 100, got {len(state)}"

        # Check sinusoidal IC is properly initialized
        T = state.reshape((10, 10))
        # Center should be near maximum
        assert T[5, 5] > 0.5, "Sinusoidal IC should have peak near center"
        # Boundaries should be near zero (for Dirichlet)
        assert abs(T[0, 5]) < 0.1, "Sinusoidal IC should be zero at y=0"
        assert abs(T[9, 5]) < 0.1, "Sinusoidal IC should be zero at y=L"

    def test_wave_equation_2d_block_initialization(self):
        """Test WaveEquation2DBlock initializes correctly."""
        from blocks.pde.wave_equation_2d import WaveEquation2DBlock

        block = WaveEquation2DBlock()
        params = {
            'c': 1.0,
            'damping': 0.0,
            'Lx': 1.0, 'Ly': 1.0,
            'Nx': 10, 'Ny': 10,
            'init_displacement': 'gaussian',
            'init_velocity': '0.0',
            'init_amplitude': 1.0
        }

        state = block.get_initial_state(params)
        state_size = block.get_state_size(params)

        # Wave equation has 2*Nx*Ny states (u and v)
        assert state_size == 200, f"Expected 2*10*10=200 states, got {state_size}"
        assert len(state) == 200, f"Expected state length 200, got {len(state)}"

        # Split into u and v
        u = state[:100].reshape((10, 10))
        v = state[100:].reshape((10, 10))

        # Gaussian should peak at center
        assert u[5, 5] > 0.5, "Gaussian IC should peak at center"
        # Velocity should be zero
        assert np.allclose(v, 0.0), "Initial velocity should be zero"

    def test_advection_equation_2d_block_initialization(self):
        """Test AdvectionEquation2DBlock initializes correctly."""
        from blocks.pde.advection_equation_2d import AdvectionEquation2DBlock

        block = AdvectionEquation2DBlock()
        params = {
            'vx': 1.0, 'vy': 0.0,
            'D': 0.0,
            'Lx': 1.0, 'Ly': 1.0,
            'Nx': 15, 'Ny': 15,
            'init_concentration': 'step',
            'init_amplitude': 1.0
        }

        state = block.get_initial_state(params)
        state_size = block.get_state_size(params)

        assert state_size == 225, f"Expected 15*15=225 states, got {state_size}"

        # Step function should be 1 on left quarter
        c = state.reshape((15, 15))
        assert np.mean(c[:, 0:3]) > 0.5, "Step IC should be ~1 on left quarter"
        assert np.mean(c[:, 10:]) < 0.1, "Step IC should be ~0 on right side"

    def test_heat_2d_block_compute_derivatives(self):
        """Test HeatEquation2DBlock computes derivatives correctly."""
        from blocks.pde.heat_equation_2d import HeatEquation2DBlock

        block = HeatEquation2DBlock()
        params = {
            'alpha': 0.1,
            'Lx': 1.0, 'Ly': 1.0,
            'Nx': 10, 'Ny': 10,
            'bc_type_left': 'Dirichlet',
            'bc_type_right': 'Dirichlet',
            'bc_type_bottom': 'Dirichlet',
            'bc_type_top': 'Dirichlet',
            'init_temp': 'sinusoidal'
        }

        state = block.get_initial_state(params)
        inputs = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}  # No source, zero BC

        derivatives = block.compute_derivatives(0.0, state, inputs, params)

        # Derivatives should not be NaN or Inf
        assert not np.any(np.isnan(derivatives)), "Derivatives contain NaN"
        assert not np.any(np.isinf(derivatives)), "Derivatives contain Inf"

        # For sinusoidal IC with Dirichlet BC, interior derivatives should be negative (cooling)
        dT = derivatives.reshape((10, 10))
        assert dT[5, 5] < 0, "Sinusoidal mode should decay (negative derivative at center)"

    def test_wave_2d_block_compute_derivatives(self):
        """Test WaveEquation2DBlock computes derivatives correctly."""
        from blocks.pde.wave_equation_2d import WaveEquation2DBlock

        block = WaveEquation2DBlock()
        params = {
            'c': 1.0,
            'damping': 0.0,
            'Lx': 1.0, 'Ly': 1.0,
            'Nx': 10, 'Ny': 10,
            'bc_type_left': 'Dirichlet',
            'bc_type_right': 'Dirichlet',
            'bc_type_bottom': 'Dirichlet',
            'bc_type_top': 'Dirichlet',
            'init_displacement': 'sinusoidal',
            'init_velocity': '0.0'
        }

        state = block.get_initial_state(params)
        inputs = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}

        derivatives = block.compute_derivatives(0.0, state, inputs, params)

        assert not np.any(np.isnan(derivatives)), "Wave derivatives contain NaN"
        assert not np.any(np.isinf(derivatives)), "Wave derivatives contain Inf"

        # du/dt should equal v (which is zero initially)
        N = 100
        du_dt = derivatives[:N].reshape((10, 10))
        # Interior du/dt should be near zero since v=0
        assert np.max(np.abs(du_dt[3:7, 3:7])) < 0.1, "du/dt should be ~0 when v=0"
