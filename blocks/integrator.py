from blocks.base_block import BaseBlock
from blocks.param_templates import init_conds_param, method_param, init_flag_param
from blocks.input_helpers import InitStateManager
import numpy as np
from scipy.integrate import solve_ivp
import logging

logger = logging.getLogger(__name__)

# Integration method choices.
#
# NOTE ON THE "RK4" LABEL: this strategy is a *fixed-step* classical 4-stage
# Runge-Kutta scheme, and that is what it is now called. It shipped for years
# mislabelled "RK45" -- scipy's adaptive Runge-Kutta 4(5), which it is not --
# so every diagram saved before the rename stores that spelling. "RK45" is
# therefore kept as a legacy alias: `resolve_method` maps it onto "RK4", which
# is the only name execute() (and the interpreter's four-sub-step gate in
# SimulationEngine.count_rk45_integrators) matches against.
INTEGRATOR_METHODS = ["FWD_EULER", "BWD_EULER", "TUSTIN", "RK4", "SOLVE_IVP"]

# Legacy spellings -> the canonical strategy name used by execute().
METHOD_ALIASES = {"RK45": "RK4"}


def resolve_method(method):
    """Map a stored ``method`` string onto the canonical strategy name.

    Saved diagrams predating the rename hold "RK45" for what is really
    fixed-step RK4; both spellings resolve to "RK4".
    """
    name = str(method)
    return METHOD_ALIASES.get(name, name)


# scipy.integrate.solve_ivp ODE methods exposed for the SOLVE_IVP strategy.
# "RK45" matches scipy's own default, so the default behaviour is unchanged.
SOLVE_IVP_METHODS = ["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"]


def _ivp_constant_rhs(t, y, u):
    """RHS for the SOLVE_IVP integrator strategy.

    The Integrator's input is held constant over a single [t, t+dtime] step,
    so dy/dt is simply the (constant) input vector ``u``. Defining this at
    module scope (and passing ``u`` via solve_ivp's ``args``) avoids rebuilding
    a closure on every time step while producing identical RHS values.
    """
    return u


class IntegratorBlock(BaseBlock):
    def __init__(self):
        super().__init__()

    @property
    def block_name(self):
        return "Integrator"

    @property
    def b_type(self):
        """Memory block - Integrator (accumulates state)."""
        return 1

    @property
    def output_is_post_update(self):
        """execute() returns params['mem'] *after* integrating this step, i.e.
        x[k+1].  The value to hold between samples is the pre-update state
        (params['output'], what the output_only path returns)."""
        return True

    @property
    def category(self):
        return "Control"

    @property
    def color(self):
        return "magenta"

    @property
    def params(self):
        return {
            **init_conds_param(default=0.0, doc="Initial condition value"),
            **method_param(
                INTEGRATOR_METHODS,
                default="SOLVE_IVP",
                doc=(
                    "Integration method, applied by the interpreted solver only. "
                    "FWD_EULER: explicit Euler. BWD_EULER: explicit Euler on the "
                    "previous input sample (see the block doc -- it is NOT implicit "
                    "Euler). TUSTIN: trapezoidal. RK4: fixed-step classical 4-stage "
                    'Runge-Kutta; "RK45" is accepted as a legacy alias for it in '
                    "diagrams saved before the rename. SOLVE_IVP: hands the step to "
                    "scipy. The compiled (fast) solver assembles the whole diagram "
                    "into one ODE system and integrates it with the solver method set "
                    "in Simulation settings, which offers Euler and RK4 as well."
                ),
            ),
            **method_param(
                SOLVE_IVP_METHODS,
                default="RK45",
                param_name="ivp_method",
                doc="scipy ODE solver used when Method is SOLVE_IVP",
            ),
            **init_flag_param(),
            "sampling_time": {
                "default": -1.0,
                "type": "float",
                "doc": "Sample time (-1=continuous, 0=inherited, >0=discrete)",
            },
        }

    @property
    def doc(self):
        return (
            "Continuous-time Integrator (1/s)."
            "\n\nComputes the time integral of the input signal."
            "\ny(t) = y(0) + integral(u(t) dt)"
            "\n\nParameters:"
            "\n- Initial Condition: Value of the output at start time."
            "\n- Limit Output: Enable saturation limits on the integral."
            "\n- Method: Integration method, used by the interpreted solver only;"
            "\n  the compiled solver integrates the whole diagram with the method"
            "\n  from Simulation settings."
            "\n\nMethods (interpreted solver):"
            "\n- FWD_EULER: y[k+1] = y[k] + h*u[k]  (explicit Euler)."
            "\n- BWD_EULER: y[k+1] = y[k] + h*u[k-1]. Despite the name this is NOT"
            "\n  implicit Euler: a plain integrator cannot form y[k+1] = y[k] +"
            "\n  h*u[k+1], because u at the new time is produced by the rest of the"
            "\n  diagram from the new state, so the implicit step would have to be"
            "\n  solved across the whole diagram at once. It is explicit Euler on the"
            "\n  previous input sample, i.e. first order with one extra step of lag."
            "\n  For stiff systems use SOLVE_IVP (Radau/BDF/LSODA) or the compiled"
            "\n  solver instead."
            "\n- TUSTIN: y[k+1] = y[k] + (h/2)*(u[k] + u[k-1])  (trapezoidal)."
            "\n- RK4: fixed-step classical 4-stage Runge-Kutta. It is not scipy's"
            "\n  adaptive RK4(5); the historical label 'RK45' is still accepted for"
            "\n  diagrams saved before the rename."
            "\n- SOLVE_IVP: hands each step to scipy.integrate.solve_ivp using the"
            "\n  method chosen in 'ivp_method'."
            "\n\nUsage:"
            "\nFundamental block for building dynamic system models."
        )

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    def draw_icon(self, block_rect):
        """Integrator uses 1/s text rendering - handled in DBlock switch."""
        return None

    def symbolic_execute(self, inputs, params):
        """
        Symbolic execution for equation extraction.

        In Laplace domain: Y(s) = U(s) / s

        Args:
            inputs: Dict of symbolic input expressions {port_idx: sympy_expr}
            params: Dict of block parameters

        Returns:
            Dict of symbolic output expressions {0: u/s}
        """
        try:
            from sympy import Symbol
        except ImportError:
            return None

        s = Symbol("s")
        u = inputs.get(0, Symbol("u"))

        # Y(s) = U(s) / s (Laplace domain integrator)
        return {0: u / s}

    def execute(self, time, inputs, params, **kwargs):
        """
        Integrator block with multiple integration methods.
        """
        output_only = kwargs.get("output_only", False)
        dtime = kwargs.get("dtime", params.get("dtime", 0.01))
        # Diagrams saved before the rename store "RK45" for what is really
        # fixed-step RK4; accept both here so either name selects the same path.
        method = resolve_method(params.get("method", "SOLVE_IVP"))

        # Initialization
        init_mgr = InitStateManager(params)
        if init_mgr.needs_init():
            params["dtime"] = dtime
            params["mem"] = np.atleast_1d(np.array(params["init_conds"], dtype=float))
            params["output"] = np.atleast_1d(np.array(params["init_conds"], dtype=float))
            params["mem_list"] = [np.zeros_like(params["mem"])]
            params["mem_len"] = 5.0
            init_mgr.mark_initialized()
            params["aux"] = np.zeros_like(params["mem"])

            if method == "RK4":
                params["nb_loop"] = 0
                params["RK45_Klist"] = [0, 0, 0, 0]

        if output_only:
            if method == "RK4" and params.get("nb_loop", 0) != 0:
                # Mid-cycle: publish the RK4 stage state (x_n + K1/2, then
                # x_n + K2/2, then x_n + K3) so the rest of the diagram
                # evaluates the derivative at the stage point.  Publishing the
                # step's start state here instead made all four stages see the
                # same state: K1 = K2 = K3 = K4, and the weighted average
                # (K1 + 2K2 + 2K3 + K4)/6 collapses to K1 -- forward Euler at
                # four times the cost.
                return {0: params["aux"], "E": False}
            result = {0: params.get("output", params["mem"]), "E": False}
            return result

        # Check input dimensions
        if isinstance(inputs.get(0), (float, int)):
            inputs[0] = np.atleast_1d(inputs[0])

        if params["mem"].shape != inputs.get(0, params["mem"]).shape:
            if params["mem"].size == 1:
                logger.warning(
                    f"Expanding initial conditions for {params['_name_']} to match input dimensions."
                )
                params["mem"] = np.full(inputs[0].shape, params["mem"].item())
            else:
                logger.error(f"Dimension Error in initial conditions in {params['_name_']}")
                init_mgr.reset()
                return {"E": True, "error": f"Dimension mismatch in {params['_name_']}"}

        # Integration by method
        if method == "FWD_EULER":
            params["mem"] += params["dtime"] * inputs[0]
        elif method == "BWD_EULER":
            # Explicit Euler on the *previous* input sample, not implicit Euler --
            # see the block doc for why a plain integrator cannot take an implicit
            # step on the interpreted path.
            params["mem"] += params["dtime"] * params["mem_list"][-1]
        elif method == "TUSTIN":
            params["mem"] += 0.5 * params["dtime"] * (inputs[0] + params["mem_list"][-1])
        elif method == "RK4":
            K_list = params["RK45_Klist"]
            K_list[params["nb_loop"]] = params["dtime"] * np.array(inputs[0], dtype=float)
            params["RK45_Klist"] = K_list
            K1, K2, K3, K4 = K_list

            if params["nb_loop"] == 0:
                params["nb_loop"] += 1
                params["aux"] = params["mem"] + 0.5 * K1
                return {"E": False}
            elif params["nb_loop"] == 1:
                params["nb_loop"] += 1
                params["aux"] = params["mem"] + 0.5 * K2
                return {"E": False}
            elif params["nb_loop"] == 2:
                params["nb_loop"] += 1
                params["aux"] = params["mem"] + K3
                return {"E": False}
            elif params["nb_loop"] == 3:
                params["nb_loop"] = 0
                params["mem"] += (1 / 6) * (K1 + 2 * K2 + 2 * K3 + K4)
        elif method == "SOLVE_IVP":
            mem_shape = params["mem"].shape
            y0 = np.atleast_1d(params["mem"]).flatten()

            # Input is constant over the step, so dy/dt = u. Pass u through
            # solve_ivp's args to a module-level RHS instead of rebuilding a
            # closure each call, and honor the configured ODE method (defaults
            # to scipy's own default, RK45, so results are unchanged).
            u = np.atleast_1d(inputs[0]).flatten()
            ivp_method = params.get("ivp_method", "RK45")

            sol = solve_ivp(
                _ivp_constant_rhs, [time, time + dtime], y0, method=ivp_method, args=(u,)
            )
            params["mem"] = sol.y[:, -1].reshape(mem_shape)
            return {0: params["mem"], "E": False}
        else:
            logger.error(f"Unknown integration method {method} in {params.get('_name_', '?')}")
            return {"E": True, "error": f"Unknown method: {method}"}

        aux_list = params["mem_list"]
        aux_list.append(inputs[0])
        if len(aux_list) > params["mem_len"]:
            aux_list = aux_list[-int(params["mem_len"]) :]
        params["mem_list"] = aux_list

        result = {0: params["mem"], "E": False}
        return result
