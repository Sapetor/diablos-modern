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

    def find_state_blocks(self, blocks: List = None) -> List:
        """
        Find all blocks that have states (Integrators, StateSpace, etc.).

        Args:
            blocks: List of blocks (uses dsim if not provided)

        Returns:
            List of state-containing blocks
        """
        if blocks is None and self.dsim is not None:
            blocks = self.dsim.blocks_list

        if blocks is None:
            raise ValueError(
                "No blocks provided and no DSim instance available to source them"
            )

        self.blocks = blocks
        self.state_blocks = []

        # NOTE: keep this set in exact correspondence with the block types
        # handled in get_state_vector / set_state_vector so detected blocks
        # always contribute a consistent number of states. RateLimiter is
        # intentionally excluded: its persisted '_prev' is a discrete
        # sample-hold value, not a continuous state, and it has no branch in
        # the state get/set routines.
        state_block_types = {
            'Integrator', 'StateSpace', 'TransferFcn', 'TranFn',
            'PID', 'HeatEquation1D', 'WaveEquation1D',
            'AdvectionEquation1D', 'DiffusionReaction1D'
        }

        for block in blocks:
            block_type = getattr(block, 'block_fn', '')
            if block_type in state_block_types:
                self.state_blocks.append(block)

        logger.info(f"Found {len(self.state_blocks)} state-containing blocks")
        return self.state_blocks

    def get_state_vector(self) -> Tuple[np.ndarray, List[str]]:
        """
        Get current state vector from all state blocks.

        Returns:
            Tuple of (state_vector, state_names)
        """
        states = []
        names = []

        for block in self.state_blocks:
            block_type = getattr(block, 'block_fn', '')
            params = getattr(block, 'exec_params', block.params)

            if block_type == 'Integrator':
                mem = params.get('mem', np.array([0.0]))
                mem = np.atleast_1d(mem).flatten()
                states.extend(mem)
                for i in range(len(mem)):
                    names.append(f"{block.name}_x{i}")

            elif block_type in ('StateSpace', 'TransferFcn', 'TranFn'):
                # States from state-space representation
                n = params.get('_n_states_', 1)
                x = params.get('_state_', np.zeros(n))
                x = np.atleast_1d(x).flatten()
                states.extend(x)
                for i in range(len(x)):
                    names.append(f"{block.name}_x{i}")

            elif block_type == 'PID':
                # PID has 2 states
                x_i = params.get('_x_i_', 0.0)
                x_d = params.get('_x_d_', 0.0)
                states.extend([x_i, x_d])
                names.extend([f"{block.name}_xi", f"{block.name}_xd"])

            elif block_type in ('HeatEquation1D', 'AdvectionEquation1D',
                               'DiffusionReaction1D'):
                T = params.get('T', params.get('c', np.zeros(20)))
                T = np.atleast_1d(T).flatten()
                states.extend(T)
                for i in range(len(T)):
                    names.append(f"{block.name}_T{i}")

            elif block_type == 'WaveEquation1D':
                u = params.get('u', np.zeros(50))
                v = params.get('v', np.zeros(50))
                u = np.atleast_1d(u).flatten()
                v = np.atleast_1d(v).flatten()
                states.extend(u)
                states.extend(v)
                for i in range(len(u)):
                    names.append(f"{block.name}_u{i}")
                for i in range(len(v)):
                    names.append(f"{block.name}_v{i}")

        return np.array(states), names

    def set_state_vector(self, x: np.ndarray):
        """
        Set state vector in all state blocks.

        Args:
            x: State vector
        """
        idx = 0

        for block in self.state_blocks:
            block_type = getattr(block, 'block_fn', '')
            params = getattr(block, 'exec_params', block.params)

            if block_type == 'Integrator':
                mem = params.get('mem', np.array([0.0]))
                n = np.atleast_1d(mem).size
                params['mem'] = x[idx:idx + n]
                idx += n

            elif block_type in ('StateSpace', 'TransferFcn', 'TranFn'):
                n = params.get('_n_states_', 1)
                params['_state_'] = x[idx:idx + n]
                idx += n

            elif block_type == 'PID':
                params['_x_i_'] = x[idx]
                params['_x_d_'] = x[idx + 1]
                idx += 2

            elif block_type in ('HeatEquation1D', 'AdvectionEquation1D',
                               'DiffusionReaction1D'):
                T = params.get('T', params.get('c', np.zeros(20)))
                n = np.atleast_1d(T).size
                if block_type == 'HeatEquation1D':
                    params['T'] = x[idx:idx + n]
                else:
                    params['c'] = x[idx:idx + n]
                idx += n

            elif block_type == 'WaveEquation1D':
                # Derive n from the actual stored u array so set round-trips
                # exactly what get_state_vector read (which uses the real u/v
                # lengths, not the 'N' param which may disagree).
                u = params.get('u', np.zeros(50))
                n = np.atleast_1d(u).size
                params['u'] = x[idx:idx + n]
                params['v'] = x[idx + n:idx + 2*n]
                idx += 2 * n

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

    def linearize_at_point(self, operating_point: Dict[str, float] = None,
                          input_blocks: List[str] = None,
                          output_blocks: List[str] = None) -> Dict:
        """
        Linearize the system at an operating point.

        Args:
            operating_point: Dict of block_name -> value for inputs
            input_blocks: List of input block names
            output_blocks: List of output block names

        Returns:
            Dict with A, B, C, D matrices and metadata

        Raises:
            NotImplementedError: numerical linearization is not yet implemented.
                The derivative/output evaluation that would produce real A, B, C
                and D matrices has not been built, so this method refuses to run
                rather than return fabricated all-zero matrices that would be
                mistaken for a valid model.
        """
        if self.dsim is None:
            raise ValueError("DSim instance required for linearization")

        # Find state blocks (real work; safe and independently useful).
        self.find_state_blocks()

        if len(self.state_blocks) == 0:
            logger.warning("No state blocks found")
            return None

        # Get current state (real work; the state-vector helpers are correct).
        x0, state_names = self.get_state_vector()
        n = len(x0)

        # The derivative function (A) and the input/output perturbations needed
        # for B, C and D have not been implemented. Computing them requires
        # driving block.execute()/the engine step at the perturbed state, which
        # is out of scope here. Fail loudly instead of returning placeholder
        # zero matrices that downstream TF/controllability/observability code
        # would treat as a real linearization.
        raise NotImplementedError(
            "linearize_at_point is not implemented: numerical computation of "
            f"the A/B/C/D matrices (found {n} states: {state_names}) requires "
            "evaluating state derivatives and outputs at the operating point, "
            "which is not yet wired to the simulation engine. Refusing to "
            "return fabricated all-zero matrices."
        )

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
        imag_parts = np.imag(eigenvalues)

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
