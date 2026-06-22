"""
Linearizer - Compute linearized state-space models from block diagrams

Provides numerical and symbolic linearization capabilities:
- Jacobian computation at operating points
- State-space (A, B, C, D) matrix extraction
- Eigenvalue analysis
- Stability determination
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger(__name__)


class Linearizer:
    """
    Linearizes block diagrams at operating points.

    Computes Jacobians numerically or symbolically to obtain
    linear state-space representations.
    """

    def __init__(self, dsim=None):
        """
        Initialize the linearizer.

        Args:
            dsim: Reference to DSim instance
        """
        self.dsim = dsim
        self.blocks = []
        self.lines = []
        self.state_blocks = []  # Blocks with states (Integrator, TF, etc.)

    def compute_jacobian_numerical(self, f: callable, x0: np.ndarray,
                                  eps: float = 1e-6) -> np.ndarray:
        """
        Compute Jacobian matrix numerically using finite differences.

        Args:
            f: Function that computes dx/dt = f(x)
            x0: Operating point
            eps: Perturbation size

        Returns:
            Jacobian matrix A = df/dx at x0
        """
        # Work in float so an integer-typed x0 cannot truncate the perturbation
        # to zero (which would yield spurious all-zero Jacobian columns).
        x0 = np.asarray(x0, dtype=float)
        n = len(x0)
        A = np.zeros((n, n))

        for j in range(n):
            # Relative step scaled by the magnitude of the component, so the
            # perturbation stays meaningful for both small and large states.
            h = eps * max(1.0, abs(x0[j]))

            x_plus = x0.copy()
            x_plus[j] += h
            x_minus = x0.copy()
            x_minus[j] -= h

            # Central difference: O(h^2) accuracy vs O(h) for one-sided.
            A[:, j] = (f(x_plus) - f(x_minus)) / (2.0 * h)

        return A

    @staticmethod
    def _vec(v) -> np.ndarray:
        """Coerce any signal value to a 1-D float array."""
        return np.atleast_1d(np.asarray(v, dtype=float)).flatten()

    @staticmethod
    def _state_names_from_map(state_map: Dict) -> List[str]:
        """Build per-state names in the compiled state-vector order."""
        names = []
        for b_name, (start, size) in sorted(state_map.items(), key=lambda kv: kv[1][0]):
            if size == 1:
                names.append(b_name)
            else:
                names.extend(f"{b_name}_x{i}" for i in range(size))
        return names

    def _compile_diagram(self) -> Tuple[Any, np.ndarray, Dict]:
        """Compile the current DSim diagram via the engine's fast path.

        Returns (model_func, y0, state_map). model_func(t, y) -> dy/dt, and (when
        the engine exposes it) model_func.evaluate(t, y, overrides) -> (dy, signals)
        plus model_func.source_names. Raises ValueError if the diagram cannot be
        compiled (e.g. it contains interpreter-only blocks or an algebraic loop).
        """
        engine = getattr(self.dsim, 'engine', None)
        if engine is None:
            raise ValueError("DSim has no engine; cannot compile for linearization")

        blocks = self.dsim.blocks_list
        lines = getattr(self.dsim, 'line_list', None)

        # Flatten + resolve (mirrors SimulationEngine.run_compiled_simulation).
        if not engine.active_blocks_list:
            if not engine.initialize_execution(blocks, lines):
                raise ValueError(
                    "Failed to initialize execution (algebraic loop or error); "
                    "cannot linearize."
                )
        current_blocks = engine.active_blocks_list
        current_lines = engine.active_line_list if engine.active_line_list else lines

        from lib.workspace import WorkspaceManager
        wm = WorkspaceManager()
        dt = getattr(self.dsim, 'sim_dt', 0.01) or 0.01
        for b in current_blocks:
            engine._resolve_block_params(b, dt, wm)

        if not engine.compiler.check_compilability(current_blocks):
            raise ValueError(
                "Diagram is not compilable, so numerical linearization is "
                "unavailable. Linearization runs on the fast-solver path; remove "
                "interpreter-only blocks (e.g. Noise, Hysteresis) to enable it."
            )

        sorted_blocks = sorted(current_blocks, key=lambda b: b.hierarchy)
        model_func, y0, state_map, _ = engine.compiler.compile_system(
            current_blocks, sorted_blocks, current_lines
        )
        return model_func, np.asarray(y0, dtype=float), state_map

    def find_operating_point(self, t: float = 0.0,
                             y_guess: np.ndarray = None,
                             input_overrides: Dict[str, Any] = None) -> Dict:
        """
        Find an equilibrium (trim point) of the compiled diagram: a state y* with
        dy/dt = f(t, y*) = 0. Linearization is only physically meaningful at an
        operating point, so this makes the find-equilibrium -> linearize ->
        analyze workflow turnkey.

        Args:
            t: time at which to evaluate the RHS (default 0.0).
            y_guess: initial guess for the root solve (default: the compiled y0).
            input_overrides: optional {source_block: value} held fixed while
                solving for the equilibrium (e.g. a constant control input).

        Returns:
            Dict with success (bool), y (equilibrium state vector), residual,
            message, state_names, and operating_point (a {block_name: value} dict
            ready to pass straight to linearize_at_point(operating_point=...)).
        """
        from scipy.optimize import root

        model_func, y0, state_map = self._compile_diagram()
        y_guess = np.asarray(y_guess if y_guess is not None else y0, dtype=float)

        evaluate = getattr(model_func, 'evaluate', None)
        if input_overrides and evaluate is not None:
            def f(y):
                return evaluate(t, np.asarray(y, dtype=float), input_overrides)[0]
        else:
            def f(y):
                return model_func(t, np.asarray(y, dtype=float))

        sol = root(f, y_guess)
        y_star = np.asarray(sol.x, dtype=float)

        op = {}
        for b_name, (start, size) in state_map.items():
            seg = y_star[start:start + size]
            op[b_name] = float(seg[0]) if size == 1 else seg

        return {
            'success': bool(sol.success),
            'y': y_star,
            'residual': np.asarray(f(y_star), dtype=float),
            'message': str(sol.message),
            'state_names': self._state_names_from_map(state_map),
            'operating_point': op,
        }

    def linearize_at_point(self, operating_point: Dict[str, float] = None,
                          input_blocks: List[str] = None,
                          output_blocks: List[str] = None,
                          t: float = 0.0, eps: float = 1e-6) -> Optional[Dict]:
        """
        Numerically linearize the diagram at an operating point.

        Computes A = ∂f/∂x by finite-differencing the compiled ODE right-hand
        side. When ``input_blocks`` and ``output_blocks`` are given, also computes
        B = ∂f/∂u, C = ∂y/∂x and D = ∂y/∂u by perturbing the named input-source
        signals and reading the named output-block signals — yielding a full
        (A, B, C, D) model that feeds the transfer-function / Bode / Nyquist /
        root-locus / controllability-observability tooling for ANY compilable
        (possibly nonlinear) diagram.

        Args:
            operating_point: optional {block_name: value} overriding the compiled
                initial state of named state blocks (linearize at a non-default
                equilibrium). Scalars or vectors matching the block's state size.
            input_blocks: source block names to treat as inputs u (for B, D).
            output_blocks: block names whose signal is the output y (for C, D).
            t: time at which to evaluate the operating point (default 0.0).
            eps: relative finite-difference step.

        Returns:
            Dict with A (+ B, C, D when I/O given), eigenvalues and stability
            info, transfer function and controllability/observability (when I/O
            given), plus state/input/output names. None if the diagram has no
            continuous states.

        Raises:
            ValueError: no DSim, diagram not compilable, or an unknown I/O block.
        """
        if self.dsim is None:
            raise ValueError("DSim instance required for linearization")

        model_func, y0, state_map = self._compile_diagram()
        y0 = y0.copy()

        # Optionally move the operating point (override compiled initial states).
        if operating_point:
            for b_name, (start, size) in state_map.items():
                if b_name not in operating_point:
                    continue
                val = self._vec(operating_point[b_name])
                if val.size == 1:
                    y0[start:start + size] = val[0]
                elif val.size == size:
                    y0[start:start + size] = val
                else:
                    logger.warning(
                        "operating_point['%s'] has size %d but block has %d "
                        "states; ignoring.", b_name, val.size, size
                    )

        n = len(y0)
        state_names = self._state_names_from_map(state_map)
        if n == 0:
            logger.warning("No continuous states found; nothing to linearize.")
            return None

        # A = ∂f/∂x via central differences on the compiled RHS.
        A = self.compute_jacobian_numerical(lambda x: model_func(t, x), y0, eps=eps)

        result: Dict[str, Any] = {
            'A': A,
            'state_names': state_names,
            'n_states': n,
            'operating_point': y0,
        }
        result.update(self.analyze_stability(A))

        # B, C, D require designated inputs/outputs and the evaluate() hook.
        evaluate = getattr(model_func, 'evaluate', None)
        if input_blocks and output_blocks and evaluate is not None:
            source_names = set(getattr(model_func, 'source_names', []))
            _, sig0 = evaluate(t, y0, None)

            for nm in input_blocks:
                if nm not in source_names:
                    raise ValueError(
                        f"Input block '{nm}' is not a compiled source block. "
                        f"Available sources: {sorted(source_names)}"
                    )
            for nm in output_blocks:
                if nm not in sig0:
                    raise ValueError(
                        f"Output block '{nm}' produces no signal. "
                        f"Available signals: {sorted(sig0)}"
                    )

            u_nom = {nm: self._vec(sig0[nm]) for nm in input_blocks}
            in_channels = [(nm, c) for nm in input_blocks for c in range(u_nom[nm].size)]
            out_channels = [(nm, c) for nm in output_blocks
                            for c in range(self._vec(sig0[nm]).size)]
            m, p = len(in_channels), len(out_channels)

            def read_outputs(signals):
                return np.array([self._vec(signals[nm])[c] for nm, c in out_channels])

            def override(nm, perturbed):
                # Preserve scalar-ness so downstream blocks see the same shape.
                return float(perturbed[0]) if perturbed.size == 1 else perturbed

            B = np.zeros((n, m))
            C = np.zeros((p, n))
            D = np.zeros((p, m))

            # B, D: perturb each input channel.
            for col, (nm, c) in enumerate(in_channels):
                base = u_nom[nm]
                h = eps * max(1.0, abs(base[c]))
                up = base.copy(); up[c] += h
                dn = base.copy(); dn[c] -= h
                dy_p, sig_p = evaluate(t, y0, {nm: override(nm, up)})
                dy_m, sig_m = evaluate(t, y0, {nm: override(nm, dn)})
                B[:, col] = (dy_p - dy_m) / (2.0 * h)
                D[:, col] = (read_outputs(sig_p) - read_outputs(sig_m)) / (2.0 * h)

            # C: perturb each state, read outputs at nominal inputs.
            for j in range(n):
                h = eps * max(1.0, abs(y0[j]))
                yp = y0.copy(); yp[j] += h
                ym = y0.copy(); ym[j] -= h
                _, sig_p = evaluate(t, yp, None)
                _, sig_m = evaluate(t, ym, None)
                C[:, j] = (read_outputs(sig_p) - read_outputs(sig_m)) / (2.0 * h)

            def chan_name(nm, c, total):
                return f"{nm}[{c}]" if total > 1 else nm

            result.update({
                'B': B, 'C': C, 'D': D,
                'input_names': [chan_name(nm, c, u_nom[nm].size) for nm, c in in_channels],
                'output_names': [chan_name(nm, c, self._vec(sig0[nm]).size)
                                 for nm, c in out_channels],
            })
            num, den = self.compute_transfer_function(A, B, C, D)
            result['transfer_function'] = {'num': num, 'den': den}
            result['controllable'] = self.is_controllable(A, B)
            result['observable'] = self.is_observable(A, C)

        return result

    def compute_transfer_function(self, A: np.ndarray, B: np.ndarray,
                                 C: np.ndarray, D: np.ndarray) -> Tuple:
        """
        Compute transfer function from state space matrices.

        G(s) = C(sI - A)^(-1)B + D

        Args:
            A, B, C, D: State space matrices

        Returns:
            Tuple of (numerator coeffs, denominator coeffs)
        """
        from scipy import signal

        try:
            num, den = signal.ss2tf(A, B, C, D)
            # signal.ss2tf returns ``num`` as a 2D array with one row per
            # output. For a single-output system return that row (SISO shape,
            # preserved for backwards compatibility); for multi-output systems
            # return the full numerator array instead of silently dropping the
            # remaining output rows.
            if num.shape[0] == 1:
                return num[0], den
            logger.warning(
                "ss2tf produced %d output rows; returning the full numerator "
                "array for the multi-output system.", num.shape[0]
            )
            return num, den
        except Exception as e:
            logger.error(f"Failed to convert to transfer function: {e}")
            return None, None

    def analyze_stability(self, A: np.ndarray) -> Dict:
        """
        Perform stability analysis on A matrix.

        Args:
            A: System matrix

        Returns:
            Dict with stability information
        """
        eigenvalues = np.linalg.eigvals(A)

        # Real parts
        real_parts = np.real(eigenvalues)

        # Stability
        is_stable = np.all(real_parts < 0)
        is_marginally_stable = np.all(real_parts <= 0) and not is_stable

        # Dominant pole
        dominant_idx = np.argmax(real_parts)
        dominant_pole = eigenvalues[dominant_idx]

        # Time constants (for real negative eigenvalues). Use a tolerance on
        # the imaginary part so genuinely-real eigenvalues that eigvals returns
        # with a tiny numerical imaginary component are still included.
        imag_tol = 1e-9
        time_constants = []
        for ev in eigenvalues:
            if abs(np.imag(ev)) < imag_tol and np.real(ev) < 0:
                time_constants.append(-1.0 / np.real(ev))

        # Natural frequencies and damping (for complex pairs). Only consider
        # eigenvalues with a positive imaginary part so each complex-conjugate
        # pair is counted once rather than twice.
        oscillatory_modes = []
        for i, ev in enumerate(eigenvalues):
            if np.imag(ev) > imag_tol:
                omega_n = np.abs(ev)
                zeta = -np.real(ev) / omega_n if omega_n > 0 else 0
                oscillatory_modes.append({
                    'omega_n': omega_n,
                    'zeta': zeta,
                    'period': 2 * np.pi / np.abs(np.imag(ev)) if np.imag(ev) != 0 else np.inf,
                })

        return {
            'eigenvalues': eigenvalues,
            'is_stable': is_stable,
            'is_marginally_stable': is_marginally_stable,
            'dominant_pole': dominant_pole,
            'time_constants': time_constants,
            'oscillatory_modes': oscillatory_modes,
        }

    def controllability_matrix(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Compute controllability matrix [B, AB, A²B, ..., A^(n-1)B].
        """
        n = A.shape[0]
        C = B.copy()
        AB = B

        for i in range(1, n):
            AB = A @ AB
            C = np.hstack([C, AB])

        return C

    def observability_matrix(self, A: np.ndarray, C: np.ndarray) -> np.ndarray:
        """
        Compute observability matrix [C; CA; CA²; ...; CA^(n-1)].
        """
        n = A.shape[0]
        O = C.copy()
        CA = C

        for i in range(1, n):
            CA = CA @ A
            O = np.vstack([O, CA])

        return O

    def is_controllable(self, A: np.ndarray, B: np.ndarray) -> bool:
        """Check if system is controllable."""
        Cm = self.controllability_matrix(A, B)
        return np.linalg.matrix_rank(Cm) == A.shape[0]

    def is_observable(self, A: np.ndarray, C: np.ndarray) -> bool:
        """Check if system is observable."""
        Om = self.observability_matrix(A, C)
        return np.linalg.matrix_rank(Om) == A.shape[0]
