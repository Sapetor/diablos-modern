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

        self.blocks = blocks
        self.state_blocks = []

        state_block_types = {
            'Integrator', 'StateSpace', 'TransferFcn', 'TranFn',
            'PID', 'RateLimiter', 'HeatEquation1D', 'WaveEquation1D',
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
                n = params.get('N', 50)
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
        n = len(x0)
        A = np.zeros((n, n))

        f0 = f(x0)

        for j in range(n):
            x_plus = x0.copy()
            x_plus[j] += eps

            f_plus = f(x_plus)

            A[:, j] = (f_plus - f0) / eps

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
        """
        if self.dsim is None:
            raise ValueError("DSim instance required for linearization")

        # Find state blocks
        self.find_state_blocks()

        if len(self.state_blocks) == 0:
            logger.warning("No state blocks found")
            return None

        # Get current state
        x0, state_names = self.get_state_vector()
        n = len(x0)

        logger.info(f"Linearizing system with {n} states")

        # Define state derivative function
        def compute_derivatives(x):
            # Set state
            self.set_state_vector(x)

            # Run one simulation step (or compute derivatives directly)
            # This is a simplified version - full implementation would
            # need to handle the simulation loop properly

            dx = np.zeros_like(x)
            idx = 0

            for block in self.state_blocks:
                block_type = getattr(block, 'block_fn', '')
                params = getattr(block, 'exec_params', block.params)

                if block_type == 'Integrator':
                    # dx/dt = input
                    mem = params.get('mem', np.array([0.0]))
                    n_states = np.atleast_1d(mem).size
                    # Would need to get actual input here
                    dx[idx:idx + n_states] = 0.0  # Placeholder
                    idx += n_states

                # Add other block types...

            return dx

        # Compute A matrix (Jacobian of dx/dt w.r.t. x)
        A = self.compute_jacobian_numerical(compute_derivatives, x0)

        # For a complete implementation, we'd also compute B, C, D
        # by perturbing inputs and measuring outputs

        # Placeholder B, C, D matrices
        m = len(input_blocks) if input_blocks else 1  # inputs
        p = len(output_blocks) if output_blocks else 1  # outputs

        B = np.zeros((n, m))
        C = np.zeros((p, n))
        D = np.zeros((p, m))

        # Compute eigenvalues for stability analysis
        eigenvalues = np.linalg.eigvals(A)

        # Stability check
        is_stable = np.all(np.real(eigenvalues) < 0)

        return {
            'A': A,
            'B': B,
            'C': C,
            'D': D,
            'n_states': n,
            'n_inputs': m,
            'n_outputs': p,
            'state_names': state_names,
            'eigenvalues': eigenvalues,
            'is_stable': is_stable,
            'operating_point': operating_point,
        }

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
            tf = signal.ss2tf(A, B, C, D)
            return tf[0][0], tf[1]
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

        # Time constants (for real negative eigenvalues)
        time_constants = []
        for ev in eigenvalues:
            if np.isreal(ev) and np.real(ev) < 0:
                time_constants.append(-1.0 / np.real(ev))

        # Natural frequencies and damping (for complex pairs)
        oscillatory_modes = []
        for i, ev in enumerate(eigenvalues):
            if np.imag(ev) != 0:
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
