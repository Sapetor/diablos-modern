"""The validation cases themselves: one diagram, one reference, one tolerance.

Every case builds a diagram, runs it through a DiaBloS execution path, and
reduces the result to a single scalar error against a reference computed
independently of DiaBloS -- a closed-form solution where one exists, otherwise
scipy (``expm``, ``lsim``, ``dlsim``, a tight-tolerance ``solve_ivp``). A case
function returns one :class:`CaseResult` per row it measures.

The registry is the single source for both consumers:

* the test modules in this package, which assert ``result.error <= result.tol``
  and add the structural assertions a scalar cannot carry;
* ``scripts/validation_report.py``, which prints the table in
  ``docs/VALIDATION.md``.

Tolerances are set from measured behaviour with roughly an order of magnitude of
headroom, so a real regression trips them and ordinary floating-point noise does
not. Where a tolerance is loose it is because the *path* is genuinely low order
(the interpreter is a fixed-step loop that samples its inputs once per step),
and the case says so in its ``note``.
"""

import numpy as np
from scipy import signal
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from tests.validation import _harness as H


# --------------------------------------------------------------------------- #
# Result record
# --------------------------------------------------------------------------- #
class CaseResult(object):
    """One measured row: what was run, how far off it was, and how far is allowed."""

    def __init__(self, name, path, method, step, error, tol, note=""):
        self.name = name
        self.path = path
        self.method = method
        self.step = step
        self.error = float(error)
        self.tol = float(tol)
        self.note = note

    @property
    def passed(self):
        return self.error <= self.tol

    def __repr__(self):
        return "CaseResult({!r}, {}, err={:.3e}, tol={:.1e})".format(
            self.name, self.path, self.error, self.tol
        )


# --------------------------------------------------------------------------- #
# Reference solutions
# --------------------------------------------------------------------------- #
def first_order_step(t, gain, tau):
    """Unit step into ``K/(tau s + 1)``."""
    return gain * (1.0 - np.exp(-t / tau))


def second_order_step(t, zeta, wn):
    """Unit step into ``wn^2 / (s^2 + 2 zeta wn s + wn^2)``, all damping regimes."""
    if zeta < 1.0:
        wd = wn * np.sqrt(1.0 - zeta**2)
        return 1.0 - np.exp(-zeta * wn * t) * (
            np.cos(wd * t) + zeta / np.sqrt(1.0 - zeta**2) * np.sin(wd * t)
        )
    if zeta == 1.0:
        return 1.0 - np.exp(-wn * t) * (1.0 + wn * t)
    root = wn * np.sqrt(zeta**2 - 1.0)
    r1, r2 = -wn * zeta + root, -wn * zeta - root
    return 1.0 + (r2 * np.exp(r1 * t) - r1 * np.exp(r2 * t)) / (r1 - r2)


def pid_closed_loop_tf(Kp, Ki, Kd, N, plant_num, plant_den):
    """``(num, den)`` of ``CP/(1+CP)`` for the filtered-derivative PID.

    ``C(s) = Kp + Ki/s + Kd N s/(s+N)``, which is the transfer function the PID
    block documents and the compiled kernel realises (its second state obeys
    ``x_d' = N(e - x_d)`` and contributes ``Kd N (e - x_d)``).
    """
    c_num = np.polyadd(np.polymul([Kp, Ki], [1.0, N]), [Kd * N, 0.0, 0.0])
    c_den = np.polymul([1.0, 0.0], [1.0, N])
    open_num = np.polymul(c_num, plant_num)
    open_den = np.polymul(c_den, plant_den)
    return open_num, np.polyadd(open_den, open_num)


def zoh_discretization(A, B, dt):
    """Exact zero-order-hold ``(Ad, Bd)`` via one matrix exponential.

    ``expm([[A, B], [0, 0]] dt)`` carries both blocks, so this is exact for a
    piecewise-constant input -- no quadrature, and no inverse of ``A``.
    """
    n = A.shape[0]
    m = B.shape[1]
    block = np.zeros((n + m, n + m))
    block[:n, :n] = A * dt
    block[:n, n:] = B * dt
    expanded = expm(block)
    return expanded[:n, :n], expanded[:n, n:]


# --------------------------------------------------------------------------- #
# Diagram builders
# --------------------------------------------------------------------------- #
def build_transfer_function(source, source_params, num, den, sim_time, sim_dt):
    """``source -> TranFn(num/den) -> Scope('y')``."""
    builder = H.build(sim_time, sim_dt)
    H.add(builder, source, "u", source_params)
    H.add(builder, "TranFn", "G", {"numerator": list(num), "denominator": list(den)})
    H.add(builder, "Scope", "sc", {"labels": "y"})
    builder.connect("u", 0, "G", 0)
    builder.connect("G", 0, "sc", 0)
    return builder


def build_integrator(source, source_params, sim_time, sim_dt, method="SOLVE_IVP"):
    """``source -> Integrator -> Scope('y')``."""
    builder = H.build(sim_time, sim_dt)
    H.add(builder, source, "u", source_params)
    H.add(builder, "Integrator", "I", {"init_conds": 0.0, "method": method})
    H.add(builder, "Scope", "sc", {"labels": "y"})
    builder.connect("u", 0, "I", 0)
    builder.connect("I", 0, "sc", 0)
    return builder


def build_state_space(A, B, C, D, x0, sim_time, sim_dt):
    """Unit step into a StateSpace block with a non-zero initial state."""
    builder = H.build(sim_time, sim_dt)
    H.add(builder, "Step", "u", {"value": 1.0, "delay": 0.0, "type": "up"})
    H.add(
        builder,
        "StateSpace",
        "S",
        {
            "A": np.asarray(A).tolist(),
            "B": np.asarray(B).tolist(),
            "C": np.asarray(C).tolist(),
            "D": np.asarray(D).tolist(),
            "init_conds": list(x0),
        },
    )
    H.add(builder, "Scope", "sc", {"labels": "y"})
    builder.connect("u", 0, "S", 0)
    builder.connect("S", 0, "sc", 0)
    return builder


def build_pid_loop(gains, plant_num, plant_den, sim_time, sim_dt):
    """Unit-step reference -> PID -> plant -> back to the PID's measurement port."""
    Kp, Ki, Kd, N = gains
    builder = H.build(sim_time, sim_dt)
    H.add(builder, "Step", "r", {"value": 1.0, "delay": 0.0, "type": "up"})
    H.add(builder, "PID", "C", {"Kp": Kp, "Ki": Ki, "Kd": Kd, "N": N}, in_ports=2)
    H.add(
        builder,
        "TranFn",
        "P",
        {"numerator": list(plant_num), "denominator": list(plant_den)},
    )
    H.add(builder, "Scope", "sc", {"labels": "y"})
    builder.connect("r", 0, "C", 0)
    builder.connect("P", 0, "C", 1)
    builder.connect("C", 0, "P", 0)
    builder.connect("P", 0, "sc", 0)
    return builder


def build_saturating_ramp(limit, sim_time, sim_dt):
    """``1 -> Integrator -> Saturation(max=limit) -> Integrator -> Scope('z')``.

    The first integrator makes ``y(t) = t``, so the saturation corner is a
    *state*-dependent switching surface crossed exactly once, at ``t = limit``.
    Integrating the clipped signal turns the kink into a visible trajectory:
    ``z = t^2/2`` until the corner and a straight line of slope ``limit`` after.
    """
    builder = H.build(sim_time, sim_dt)
    H.add(builder, "Constant", "c", {"value": 1.0})
    H.add(builder, "Integrator", "ramp", {"init_conds": 0.0})
    H.add(builder, "Saturation", "sat", {"min": -1e9, "max": limit})
    H.add(builder, "Integrator", "acc", {"init_conds": 0.0})
    H.add(builder, "Scope", "sc", {"labels": "z"})
    builder.connect("c", 0, "ramp", 0)
    builder.connect("ramp", 0, "sat", 0)
    builder.connect("sat", 0, "acc", 0)
    builder.connect("acc", 0, "sc", 0)
    return builder


def saturating_ramp_reference(t, limit):
    return np.where(t < limit, t**2 / 2.0, limit**2 / 2.0 + limit * (t - limit))


def build_switched_ramp(threshold, sim_time, sim_dt):
    """A Switch whose control is a ramp: it flips exactly once, at ``threshold``.

    ``Switch`` selects data input 0 while ``ctrl >= threshold`` and input 1
    below it, so with ``+1`` on the low branch and ``-1`` on the high branch the
    integrated output rises to ``threshold`` and then falls at unit rate.
    """
    builder = H.build(sim_time, sim_dt)
    H.add(builder, "Constant", "c", {"value": 1.0})
    H.add(builder, "Integrator", "ramp", {"init_conds": 0.0})
    H.add(builder, "Constant", "hi", {"value": -1.0})
    H.add(builder, "Constant", "lo", {"value": 1.0})
    H.add(
        builder,
        "Switch",
        "sw",
        {"threshold": threshold, "n_inputs": 2, "mode": "threshold"},
        in_ports=3,
    )
    H.add(builder, "Integrator", "acc", {"init_conds": 0.0})
    H.add(builder, "Scope", "sc", {"labels": "z"})
    builder.connect("c", 0, "ramp", 0)
    builder.connect("ramp", 0, "sw", 0)
    builder.connect("hi", 0, "sw", 1)
    builder.connect("lo", 0, "sw", 2)
    builder.connect("sw", 0, "acc", 0)
    builder.connect("acc", 0, "sc", 0)
    return builder


def switched_ramp_reference(t, threshold):
    return np.where(t < threshold, t, threshold - (t - threshold))


def build_van_der_pol(mu, x0, sim_time, sim_dt):
    """Van der Pol from primitives: ``x1' = x2``, ``x2' = mu(1-x1^2)x2 - x1``.

    Wired as two Integrators plus the algebraic blocks that form the right-hand
    side, so the compiled system is assembled from the diagram rather than from
    a single block that already knows the equation.
    """
    builder = H.build(sim_time, sim_dt)
    H.add(builder, "Integrator", "x2", {"init_conds": float(x0[1])})
    H.add(builder, "Integrator", "x1", {"init_conds": float(x0[0])})
    H.add(builder, "SgProd", "square", {})
    H.add(builder, "Constant", "one", {"value": 1.0})
    H.add(builder, "Sum", "one_minus_sq", {"sign": "+-"})
    H.add(builder, "SgProd", "damping", {})
    H.add(builder, "Gain", "mu", {"gain": float(mu)})
    H.add(builder, "Sum", "dx2", {"sign": "+-"})
    H.add(builder, "Scope", "sc", {"labels": "x1"})
    builder.connect("x2", 0, "x1", 0)
    builder.connect("x1", 0, "square", 0)
    builder.connect("x1", 0, "square", 1)
    builder.connect("one", 0, "one_minus_sq", 0)
    builder.connect("square", 0, "one_minus_sq", 1)
    builder.connect("one_minus_sq", 0, "damping", 0)
    builder.connect("x2", 0, "damping", 1)
    builder.connect("damping", 0, "mu", 0)
    builder.connect("mu", 0, "dx2", 0)
    builder.connect("x1", 0, "dx2", 1)
    builder.connect("dx2", 0, "x2", 0)
    builder.connect("x1", 0, "sc", 0)
    return builder


def build_heat_1d(alpha, L, N, sim_time, sim_dt):
    """Heat block with homogeneous Dirichlet ends and a ``sin(pi x/L)`` start."""
    builder = H.build(sim_time, sim_dt)
    H.add(builder, "Constant", "zero", {"value": 0.0})
    H.add(
        builder,
        "HeatEquation1D",
        "heat",
        {
            "alpha": alpha,
            "L": L,
            "N": N,
            "init_conds": "sine",
            "bc_type_left": "Dirichlet",
            "bc_type_right": "Dirichlet",
        },
        in_ports=3,
        out_ports=2,
    )
    H.add(builder, "Scope", "field", {"labels": "T"})
    H.add(builder, "Scope", "avg", {"labels": "T_avg"})
    for port in (0, 1, 2):
        builder.connect("zero", 0, "heat", port)
    builder.connect("heat", 0, "field", 0)
    builder.connect("heat", 1, "avg", 0)
    return builder


def build_advection_1d(velocity, L, N, sim_time, sim_dt):
    """Advection block carrying its built-in Gaussian pulse, zero inlet."""
    builder = H.build(sim_time, sim_dt)
    H.add(builder, "Constant", "zero", {"value": 0.0})
    H.add(
        builder,
        "AdvectionEquation1D",
        "adv",
        {
            "velocity": velocity,
            "L": L,
            "N": N,
            "bc_type": "Dirichlet",
            "init_conds": "gaussian",
        },
        in_ports=1,
        out_ports=2,
    )
    H.add(builder, "Scope", "field", {"labels": "c"})
    H.add(builder, "Scope", "mass", {"labels": "c_total"})
    builder.connect("zero", 0, "adv", 0)
    builder.connect("adv", 0, "field", 0)
    builder.connect("adv", 1, "mass", 0)
    return builder


# --------------------------------------------------------------------------- #
# 1. Closed-form linear responses
# --------------------------------------------------------------------------- #
FIRST_ORDER = dict(gain=2.0, tau=0.5, sim_time=3.0, sim_dt=0.005)


def case_first_order_lag():
    """Unit step into ``K/(tau s + 1)``: ``y = K(1 - e^{-t/tau})``."""
    cfg = FIRST_ORDER
    rows = []
    for compiled, tol in ((True, 1e-7), (False, 1e-9)):
        builder = build_transfer_function(
            "Step",
            {"value": 1.0, "delay": 0.0, "type": "up"},
            [cfg["gain"]],
            [cfg["tau"], 1.0],
            cfg["sim_time"],
            cfg["sim_dt"],
        )
        result = H.run(builder, compiled=compiled)
        t, y = result.signal("y")
        rows.append(
            CaseResult(
                "First-order lag K/(tau s+1), unit step",
                "compiled" if compiled else "interpreter",
                "RK45" if compiled else "exact ZOH",
                cfg["sim_dt"],
                H.max_abs_error(y, first_order_step(t, cfg["gain"], cfg["tau"])),
                tol,
                note="closed form",
            )
        )
        H.release(result)
    return rows


SECOND_ORDER_ZETAS = (0.3, 1.0, 2.0)
SECOND_ORDER = dict(wn=2.0, sim_time=8.0, sim_dt=0.01)


def case_second_order_step():
    """Unit step into a second-order plant, under/critically/over-damped."""
    cfg = SECOND_ORDER
    wn = cfg["wn"]
    rows = []
    for zeta in SECOND_ORDER_ZETAS:
        for compiled, tol in ((True, 1e-7), (False, 1e-9)):
            builder = build_transfer_function(
                "Step",
                {"value": 1.0, "delay": 0.0, "type": "up"},
                [wn * wn],
                [1.0, 2 * zeta * wn, wn * wn],
                cfg["sim_time"],
                cfg["sim_dt"],
            )
            result = H.run(builder, compiled=compiled)
            t, y = result.signal("y")
            rows.append(
                CaseResult(
                    "Second-order step, zeta={0}".format(zeta),
                    "compiled" if compiled else "interpreter",
                    "RK45" if compiled else "exact ZOH",
                    cfg["sim_dt"],
                    H.max_abs_error(y, second_order_step(t, zeta, wn)),
                    tol,
                    note="closed form",
                )
            )
            H.release(result)
    return rows


RAMP_SLOPE = 2.0
SINE_AMPLITUDE, SINE_OMEGA = 1.5, 3.0
TRACKING = dict(sim_time=4.0, sim_dt=0.01)


def case_integrator_tracking():
    """Integrator against two closed forms: a ramp and a sine.

    ``int_0^t a s ds = a t^2 / 2`` and ``int_0^t A sin(w s) ds = A(1-cos wt)/w``.
    The interpreter runs its fixed-step RK4, which evaluates the source at the
    stage times; its default ``SOLVE_IVP`` method instead freezes the input over
    the step and is only first-order accurate on a time-varying input (see
    ``test_convergence.py``).
    """
    cfg = TRACKING
    rows = []
    ramp_ref = lambda t: 0.5 * RAMP_SLOPE * t**2  # noqa: E731
    sine_ref = lambda t: SINE_AMPLITUDE / SINE_OMEGA * (1 - np.cos(SINE_OMEGA * t))  # noqa: E731
    specs = (
        ("Integrator of a ramp -> a t^2/2", "Ramp", {"slope": RAMP_SLOPE, "delay": 0.0}, ramp_ref),
        (
            "Integrator of a sine -> A(1-cos wt)/w",
            "Sine",
            {"amplitude": SINE_AMPLITUDE, "omega": SINE_OMEGA, "init_angle": 0.0},
            sine_ref,
        ),
    )
    for name, source, params, reference in specs:
        # The compiled path integrates the whole diagram with the Simulation
        # settings method and warns when a block asks for its own, so leave the
        # Integrator on its default there and select RK4 for the interpreter.
        for compiled, method, tol in ((True, "RK45", 1e-7), (False, "RK4", 1e-6)):
            builder = build_integrator(
                source,
                params,
                cfg["sim_time"],
                cfg["sim_dt"],
                method="SOLVE_IVP" if compiled else "RK4",
            )
            result = H.run(builder, compiled=compiled)
            t, y = result.signal("y")
            rows.append(
                CaseResult(
                    name,
                    "compiled" if compiled else "interpreter",
                    method,
                    cfg["sim_dt"],
                    H.max_abs_error(y, reference(t)),
                    tol,
                    note="closed form",
                )
            )
            H.release(result)
    return rows


# --------------------------------------------------------------------------- #
# 2. Independent scipy references for the LTI blocks
# --------------------------------------------------------------------------- #
SS_A = np.array([[0.0, 1.0], [-4.0, -0.6]])
SS_B = np.array([[0.0], [1.0]])
SS_C = np.array([[1.0, 0.0]])
SS_D = np.array([[0.0]])
SS_X0 = (0.5, -0.2)
SS = dict(sim_time=6.0, sim_dt=0.01)


def state_space_reference(n_samples, dt):
    """Exact ZOH recursion for a unit step, from ``scipy.linalg.expm``."""
    Ad, Bd = zoh_discretization(SS_A, SS_B, dt)
    x = np.zeros((n_samples, SS_A.shape[0]))
    x[0] = SS_X0
    for k in range(n_samples - 1):
        x[k + 1] = Ad.dot(x[k]) + Bd[:, 0]
    return (SS_C.dot(x.T)).ravel()


def case_state_space_vs_expm():
    """StateSpace against the exact matrix-exponential discretization.

    A step input is piecewise constant, so the ZOH recursion built from
    ``expm`` is not an approximation of the continuous solution -- it *is* the
    continuous solution sampled on the output grid.
    """
    cfg = SS
    n = int(round(cfg["sim_time"] / cfg["sim_dt"])) + 1
    reference = state_space_reference(n, cfg["sim_dt"])
    rows = []
    for compiled, tol in ((True, 1e-7), (False, 1e-11)):
        builder = build_state_space(SS_A, SS_B, SS_C, SS_D, SS_X0, cfg["sim_time"], cfg["sim_dt"])
        result = H.run(builder, compiled=compiled)
        _t, y = result.signal("y")
        rows.append(
            CaseResult(
                "StateSpace vs scipy.linalg.expm (exact ZOH)",
                "compiled" if compiled else "interpreter",
                "RK45" if compiled else "exact ZOH",
                cfg["sim_dt"],
                H.max_abs_error(y, reference[: len(y)]),
                tol,
                note="scipy.linalg.expm",
            )
        )
        H.release(result)
    return rows


LSIM_NUM, LSIM_DEN = [2.0, 1.0], [1.0, 0.8, 3.0]
LSIM = dict(omega=2.5, sim_time=6.0, sim_dt=0.002)


def case_transfer_function_vs_lsim():
    """TranFn driven by a sine, against ``scipy.signal.lsim``.

    ``lsim`` interpolates the input between samples, so on this grid it is
    itself accurate to about 1e-6 -- which is what bounds the compiled
    tolerance. The interpreter samples the sine once per step and holds it, so
    its error is first order in ``dt``; the tolerance says so rather than
    pretending otherwise.
    """
    cfg = LSIM
    t_grid = np.arange(0.0, cfg["sim_time"] + 1e-12, cfg["sim_dt"])
    _, reference, _ = signal.lsim((LSIM_NUM, LSIM_DEN), U=np.sin(cfg["omega"] * t_grid), T=t_grid)
    rows = []
    for compiled, tol in ((True, 1e-4), (False, 2e-2)):
        builder = build_transfer_function(
            "Sine",
            {"amplitude": 1.0, "omega": cfg["omega"], "init_angle": 0.0},
            LSIM_NUM,
            LSIM_DEN,
            cfg["sim_time"],
            cfg["sim_dt"],
        )
        result = H.run(builder, compiled=compiled)
        _t, y = result.signal("y")
        rows.append(
            CaseResult(
                "TranFn vs scipy.signal.lsim (sine input)",
                "compiled" if compiled else "interpreter",
                "RK45" if compiled else "ZOH input",
                cfg["sim_dt"],
                H.max_abs_error(y, reference[: len(y)]),
                tol,
                note="scipy.signal.lsim",
            )
        )
        H.release(result)
    return rows


# --------------------------------------------------------------------------- #
# 3. PID closed loop
# --------------------------------------------------------------------------- #
PID_GAINS = (4.0, 2.0, 0.5, 20.0)
PID_PLANT_WN, PID_PLANT_ZETA = 2.0, 0.3
PID_PLANT_NUM = [PID_PLANT_WN**2]
PID_PLANT_DEN = [1.0, 2 * PID_PLANT_ZETA * PID_PLANT_WN, PID_PLANT_WN**2]
PID_LOOP = dict(sim_time=8.0, sim_dt=0.002)


def pid_loop_reference(sim_time, sim_dt):
    """Step response of ``CP/(1+CP)``, simulated with ``lsim``."""
    num, den = pid_closed_loop_tf(
        PID_GAINS[0], PID_GAINS[1], PID_GAINS[2], PID_GAINS[3], PID_PLANT_NUM, PID_PLANT_DEN
    )
    t_grid = np.arange(0.0, sim_time + 1e-12, sim_dt)
    _, y, _ = signal.lsim((num, den), U=np.ones_like(t_grid), T=t_grid)
    return y


def case_pid_closed_loop():
    """A PID around a second-order plant, against the analytic closed loop.

    The whole loop -- controller, plant and feedback -- is assembled from
    blocks; the reference is the single transfer function ``CP/(1+CP)`` formed
    with polynomial algebra and simulated by ``lsim``. Agreement means the
    compiler realises exactly the controller the block documents, in exactly
    the loop the diagram draws.

    The interpreter runs the same loop as a fixed-step difference equation --
    a one-sample delay around the feedback path and a backward-Euler filtered
    derivative -- so it is first order in ``dt`` rather than exact, and its
    tolerance says so. ``test_closed_loop.py`` measures the order.
    """
    cfg = PID_LOOP
    reference = pid_loop_reference(cfg["sim_time"], cfg["sim_dt"])
    rows = []
    for compiled, method, tol, note in (
        (True, "RK45", 1e-7, "scipy.signal.lsim"),
        (False, "fixed step", 2e-2, "first order in dt by construction"),
    ):
        builder = build_pid_loop(
            PID_GAINS, PID_PLANT_NUM, PID_PLANT_DEN, cfg["sim_time"], cfg["sim_dt"]
        )
        result = H.run(builder, compiled=compiled)
        _t, y = result.signal("y")
        rows.append(
            CaseResult(
                "PID closed loop vs analytic CP/(1+CP)",
                "compiled" if compiled else "interpreter",
                method,
                cfg["sim_dt"],
                H.max_abs_error(y, reference[: len(y)]),
                tol,
                note=note,
            )
        )
        H.release(result)
    return rows


def pid_loop_errors(steps):
    """Interpreted closed-loop error against ``CP/(1+CP)`` at each step size."""
    errors = []
    for dt in steps:
        builder = build_pid_loop(PID_GAINS, PID_PLANT_NUM, PID_PLANT_DEN, PID_LOOP["sim_time"], dt)
        result = H.run(builder, compiled=False)
        _t, y = result.signal("y")
        reference = pid_loop_reference(PID_LOOP["sim_time"], dt)
        errors.append(H.max_abs_error(y, reference[: len(y)]))
        H.release(result)
    return errors


# --------------------------------------------------------------------------- #
# 4. Discrete time
# --------------------------------------------------------------------------- #
DISCRETE = dict(tau=0.5, Ts=0.1, sim_time=3.0, sim_dt=0.01)


def discrete_plant():
    """A first-order ZOH-equivalent in ``z``, as descending-power coefficients."""
    pole = np.exp(-DISCRETE["Ts"] / DISCRETE["tau"])
    return [1.0 - pole, 0.0], [1.0, -pole]


def case_discrete_transfer_function():
    """DiscreteTranFn against ``scipy.signal.dlsim``, plus hold semantics.

    Two separate claims are measured. At the sample instants the block must
    reproduce the z-domain recursion exactly (same ``tf2ss`` realisation, same
    arithmetic, so the bar is machine precision). *Between* them the output must
    not move at all -- a discrete block that drifts between samples is running
    at the solver's rate instead of its own.
    """
    cfg = DISCRETE
    num, den = discrete_plant()
    builder = H.build(cfg["sim_time"], cfg["sim_dt"])
    H.add(builder, "Step", "u", {"value": 1.0, "delay": 0.0, "type": "up"})
    H.add(
        builder,
        "DiscreteTranFn",
        "G",
        {"numerator": num, "denominator": den, "sampling_time": cfg["Ts"]},
    )
    H.add(builder, "Scope", "sc", {"labels": "y"})
    builder.connect("u", 0, "G", 0)
    builder.connect("G", 0, "sc", 0)

    # A discrete rate forces the interpreter: the compiled right-hand side has
    # no notion of sample instants.
    result = H.run(builder, compiled=False)
    t, y = result.signal("y")

    sample_times = np.arange(0.0, cfg["sim_time"] + 1e-12, cfg["Ts"])
    _, y_ref = signal.dlsim((num, den, cfg["Ts"]), np.ones_like(sample_times))
    y_ref = np.asarray(y_ref).ravel()
    sample_idx = [int(round(ts / cfg["sim_dt"])) for ts in sample_times]
    sample_idx = [i for i in sample_idx if i < len(y)]
    at_samples = H.max_abs_error(y[sample_idx], y_ref[: len(sample_idx)])

    hold_error = 0.0
    for k in range(len(sample_idx) - 1):
        segment = y[sample_idx[k] : sample_idx[k + 1]]
        hold_error = max(hold_error, float(np.max(np.abs(segment - segment[0]))))
    H.release(result)

    return [
        CaseResult(
            "DiscreteTranFn vs scipy.signal.dlsim (at samples)",
            "interpreter",
            "z-domain, Ts=0.1",
            cfg["sim_dt"],
            at_samples,
            1e-12,
            note="scipy.signal.dlsim",
        ),
        CaseResult(
            "Discrete output is constant between sample instants",
            "interpreter",
            "z-domain, Ts=0.1",
            cfg["sim_dt"],
            hold_error,
            1e-12,
            note="sample-time semantics",
        ),
    ]


ZOH = dict(Ts=0.1, omega=3.0, sim_time=2.0, sim_dt=0.005)


def case_zero_order_hold():
    """ZeroOrderHold of a sine against the analytic staircase.

    ``y(t) = sin(w Ts floor(t/Ts))``: the value of the last sample instant,
    held. This pins both the sampled value and the instants at which it moves.
    """
    cfg = ZOH
    builder = H.build(cfg["sim_time"], cfg["sim_dt"])
    H.add(builder, "Sine", "u", {"amplitude": 1.0, "omega": cfg["omega"], "init_angle": 0.0})
    H.add(builder, "ZeroOrderHold", "Z", {"sampling_time": cfg["Ts"]})
    H.add(builder, "Scope", "sc", {"labels": "y"})
    builder.connect("u", 0, "Z", 0)
    builder.connect("Z", 0, "sc", 0)

    result = H.run(builder, compiled=False)
    t, y = result.signal("y")
    staircase = np.sin(cfg["omega"] * (np.floor(t / cfg["Ts"] + 1e-9) * cfg["Ts"]))
    row = CaseResult(
        "ZeroOrderHold of a sine vs analytic staircase",
        "interpreter",
        "ZOH, Ts=0.1",
        cfg["sim_dt"],
        H.max_abs_error(y, staircase),
        1e-9,
        note="closed form",
    )
    H.release(result)
    return [row]


# --------------------------------------------------------------------------- #
# 5. Transport delay
# --------------------------------------------------------------------------- #
DELAY = dict(omega=2.0, sim_time=3.0, sim_dt=0.005)


def _run_transport_delay(tau, sim_dt):
    cfg = DELAY
    builder = H.build(cfg["sim_time"], sim_dt)
    H.add(builder, "Sine", "u", {"amplitude": 1.0, "omega": cfg["omega"], "init_angle": 0.0})
    H.add(builder, "TransportDelay", "D", {"delay_time": tau, "initial_value": 0.0})
    H.add(builder, "Scope", "sc", {"labels": "y"})
    builder.connect("u", 0, "D", 0)
    builder.connect("D", 0, "sc", 0)
    result = H.run(builder, compiled=False)
    t, y = result.signal("y")
    shifted = np.where(t < tau, 0.0, np.sin(cfg["omega"] * (t - tau)))
    settled = t >= tau + sim_dt
    after = H.max_abs_error(y[settled], shifted[settled])
    before = float(np.max(np.abs(y[t < tau]))) if np.any(t < tau) else 0.0
    H.release(result)
    return after, before


def case_transport_delay():
    """A delayed sine against ``sin(w(t - tau))``, on and off the output grid.

    With ``tau`` an exact multiple of ``dt`` the buffer replays stored samples
    and the result is exact. Off the grid the block interpolates, and the error
    is the interpolation error -- second order in ``dt``, not first, which is
    what distinguishes interpolation from nearest-sample lookup.
    """
    cfg = DELAY
    on_grid, on_grid_pre = _run_transport_delay(0.35, cfg["sim_dt"])
    off_grid, _ = _run_transport_delay(0.333, cfg["sim_dt"])
    return [
        CaseResult(
            "Transport delay of a sine, tau on the output grid",
            "interpreter",
            "buffer replay",
            cfg["sim_dt"],
            on_grid,
            1e-12,
            note="closed form",
        ),
        CaseResult(
            "Transport delay of a sine, tau off the grid",
            "interpreter",
            "buffer + interpolation",
            cfg["sim_dt"],
            off_grid,
            1e-4,
            note="closed form",
        ),
        CaseResult(
            "Transport delay holds its initial value for t < tau",
            "interpreter",
            "buffer replay",
            cfg["sim_dt"],
            on_grid_pre,
            1e-12,
            note="closed form",
        ),
    ]


# --------------------------------------------------------------------------- #
# 6. Events and discontinuities
# --------------------------------------------------------------------------- #
SATURATION = dict(limit=0.7, sim_time=2.0, sim_dt=0.01)
SWITCH = dict(threshold=0.5, sim_time=2.0, sim_dt=0.01)


def first_event_time(result):
    """The first zero-crossing instant the compiled run located, or ``nan``."""
    info = result.diagnostics.get("zero_crossing") or {}
    events = info.get("first_events") or []
    return float(events[0][0]) if events else float("nan")


def case_saturation_event():
    """A ramp crossing a saturation limit at a time known exactly.

    The corner is at ``t = limit`` because the integrator makes ``y = t``. With
    zero-crossing detection on, the instant is *located*, not straddled, and the
    integrated trajectory follows the analytic parabola-then-line to a tolerance
    far below what step-accuracy smearing would allow -- which the interpreter
    row quantifies: same diagram, no event machinery, error five orders larger.
    """
    cfg = SATURATION
    rows = []

    builder = build_saturating_ramp(cfg["limit"], cfg["sim_time"], cfg["sim_dt"])
    result = H.run(builder, compiled=True, zero_crossing=True)
    t, z = result.signal("z")
    rows.append(
        CaseResult(
            "Saturation corner: located switching instant |t - 0.7|",
            "compiled",
            "RK45 + events",
            cfg["sim_dt"],
            abs(first_event_time(result) - cfg["limit"]),
            1e-9,
            note="closed form",
        )
    )
    rows.append(
        CaseResult(
            "Saturation corner: trajectory vs analytic",
            "compiled",
            "RK45 + events",
            cfg["sim_dt"],
            H.max_abs_error(z, saturating_ramp_reference(t, cfg["limit"])),
            1e-6,
            note="closed form",
        )
    )
    H.release(result)

    builder = build_saturating_ramp(cfg["limit"], cfg["sim_time"], cfg["sim_dt"])
    result = H.run(builder, compiled=False)
    t, z = result.signal("z")
    rows.append(
        CaseResult(
            "Saturation corner: trajectory vs analytic",
            "interpreter",
            "fixed step, no events",
            cfg["sim_dt"],
            H.max_abs_error(z, saturating_ramp_reference(t, cfg["limit"])),
            1e-2,
            note="first order in dt by construction",
        )
    )
    H.release(result)
    return rows


def case_switch_event():
    """A Switch flipping at a control threshold reached at a known instant."""
    cfg = SWITCH
    builder = build_switched_ramp(cfg["threshold"], cfg["sim_time"], cfg["sim_dt"])
    result = H.run(builder, compiled=True, zero_crossing=True)
    t, z = result.signal("z")
    rows = [
        CaseResult(
            "Switch threshold: located switching instant |t - 0.5|",
            "compiled",
            "RK45 + events",
            cfg["sim_dt"],
            abs(first_event_time(result) - cfg["threshold"]),
            1e-9,
            note="closed form",
        ),
        CaseResult(
            "Switch threshold: trajectory vs analytic",
            "compiled",
            "RK45 + events",
            cfg["sim_dt"],
            H.max_abs_error(z, switched_ramp_reference(t, cfg["threshold"])),
            1e-6,
            note="closed form",
        ),
    ]
    H.release(result)
    return rows


# --------------------------------------------------------------------------- #
# 7. Stiff systems
# --------------------------------------------------------------------------- #
VDP = dict(mu=1000.0, x0=(2.0, 0.0), sim_time=1.0, sim_dt=0.005)


def van_der_pol_reference(mu, x0, sim_time, sim_dt):
    """Van der Pol on the output grid from a tight-tolerance Radau solve."""
    grid = np.arange(0.0, sim_time + 1e-12, sim_dt)
    sol = solve_ivp(
        lambda t, y: [y[1], mu * (1.0 - y[0] ** 2) * y[1] - y[0]],
        (0.0, sim_time),
        list(x0),
        method="Radau",
        rtol=1e-10,
        atol=1e-12,
        t_eval=grid,
    )
    if not sol.success:
        raise AssertionError("reference solve failed: {}".format(sol.message))
    return sol.y[0]


def case_van_der_pol_stiff():
    """Van der Pol at ``mu = 1000``, against a reference two orders tighter.

    At this stiffness an explicit method is unusable, so the case is a check on
    the *stiff* solver settings reaching the compiled system at all: the same
    equations, integrated by ``solve_ivp`` directly at ``rtol=1e-10``, must come
    back within the run's own tolerance.
    """
    cfg = VDP
    reference = van_der_pol_reference(cfg["mu"], cfg["x0"], cfg["sim_time"], cfg["sim_dt"])
    rows = []
    for method in ("Radau", "LSODA"):
        builder = build_van_der_pol(cfg["mu"], cfg["x0"], cfg["sim_time"], cfg["sim_dt"])
        result = H.run(builder, compiled=True, solver_method=method, rtol=1e-9, atol=1e-12)
        _t, x1 = result.signal("x1")
        rows.append(
            CaseResult(
                "Van der Pol, mu=1000",
                "compiled",
                method,
                cfg["sim_dt"],
                H.max_abs_error(x1, reference[: len(x1)]),
                1e-6,
                note="solve_ivp Radau, rtol=1e-10",
            )
        )
        H.release(result)
    return rows


# --------------------------------------------------------------------------- #
# 8. PDE
# --------------------------------------------------------------------------- #
HEAT = dict(alpha=0.2, L=1.0, sim_time=0.5, sim_dt=0.005)
HEAT_GRIDS = (21, 41, 81)


def case_heat_eigenmode():
    """The fundamental heat eigenmode, and the order at which it converges.

    ``sin(pi x/L)`` under homogeneous Dirichlet ends is an eigenfunction, so the
    field decays in place at ``alpha (pi/L)^2`` and its spatial mean decays at
    the same rate. A second-order central difference resolves that eigenvalue to
    ``O(dx^2)``, so refining the grid must shrink the departure from the
    *continuous* rate by four each time -- a much sharper statement than any
    single-grid tolerance, and one no coincidence passes.
    """
    cfg = HEAT
    rate = cfg["alpha"] * (np.pi / cfg["L"]) ** 2
    errors = []
    for N in HEAT_GRIDS:
        builder = build_heat_1d(cfg["alpha"], cfg["L"], N, cfg["sim_time"], cfg["sim_dt"])
        result = H.run(builder, compiled=True)
        t, avg = result.signal("T_avg")
        analytic = avg[0] * np.exp(-rate * t)
        errors.append(H.max_abs_error(avg, analytic) / abs(avg[0]))
        H.release(result)

    orders = H.observed_order(errors)
    rows = [
        CaseResult(
            "Heat 1-D eigenmode decay, N={0}".format(HEAT_GRIDS[-1]),
            "compiled",
            "RK45, method of lines",
            cfg["sim_dt"],
            errors[-1],
            2e-4,
            note="closed form (relative)",
        ),
        CaseResult(
            "Heat 1-D spatial convergence order |p - 2|",
            "compiled",
            "N = {0}".format(", ".join(str(n) for n in HEAT_GRIDS)),
            cfg["sim_dt"],
            max(abs(p - 2.0) for p in orders),
            0.1,
            note="observed order {0}".format(", ".join("%.2f" % p for p in orders)),
        ),
    ]
    return rows


ADVECTION = dict(velocity=1.0, L=2.0, sim_time=0.5, sim_dt=0.005)
ADVECTION_GRIDS = (101, 201, 401)


def advection_reference(x, L, velocity, t_end):
    """The block's built-in Gaussian pulse, translated by ``velocity * t``."""
    return np.exp(-25.0 * (x - L / 4.0 - velocity * t_end) ** 2)


def case_advection_pulse():
    """A Gaussian pulse advected at constant velocity: shape, order and mass.

    ``c_t + v c_x = 0`` translates the initial profile rigidly, so the exact
    solution at ``t_end`` is the initial Gaussian shifted by ``v t_end``. The
    block's second-order upwind stencil must reproduce it to ``O(dx^2)``, and
    -- since the pulse never reaches a boundary -- must conserve its integral.
    """
    cfg = ADVECTION
    errors = []
    mass_drift = 0.0
    for N in ADVECTION_GRIDS:
        builder = build_advection_1d(cfg["velocity"], cfg["L"], N, cfg["sim_time"], cfg["sim_dt"])
        result = H.run(builder, compiled=True)
        t, field = result.field("field")
        x = np.linspace(0.0, cfg["L"], N)
        exact = advection_reference(x, cfg["L"], cfg["velocity"], t[-1])
        errors.append(float(np.max(np.abs(field[-1] - exact))))
        _tm, mass = result.signal("c_total")
        mass_drift = max(mass_drift, abs(mass[-1] - mass[0]) / abs(mass[0]))
        H.release(result)

    orders = H.observed_order(errors)
    return [
        CaseResult(
            "Advection 1-D Gaussian pulse, N={0}".format(ADVECTION_GRIDS[-1]),
            "compiled",
            "RK45, 2nd-order upwind",
            cfg["sim_dt"],
            errors[-1],
            5e-3,
            note="closed form",
        ),
        CaseResult(
            "Advection 1-D spatial convergence order |p - 2|",
            "compiled",
            "N = {0}".format(", ".join(str(n) for n in ADVECTION_GRIDS)),
            cfg["sim_dt"],
            max(abs(p - 2.0) for p in orders),
            0.2,
            note="observed order {0}".format(", ".join("%.2f" % p for p in orders)),
        ),
        CaseResult(
            "Advection 1-D mass conservation (relative drift)",
            "compiled",
            "RK45, 2nd-order upwind",
            cfg["sim_dt"],
            mass_drift,
            1e-4,
            note="conservation law",
        ),
    ]


# --------------------------------------------------------------------------- #
# 9. Convergence order of the interpreter's integration methods
# --------------------------------------------------------------------------- #
ORDER_STEPS = (0.02, 0.01, 0.005)
ORDER_RUN = dict(omega=2.0, sim_time=4.0)
EXPECTED_ORDERS = (("FWD_EULER", 1.0, 0.1), ("RK4", 4.0, 0.15))


def integrator_order(method):
    """Errors and observed orders for ``method`` on ``int sin(wt) dt``."""
    cfg = ORDER_RUN
    errors = []
    for dt in ORDER_STEPS:
        builder = build_integrator(
            "Sine",
            {"amplitude": 1.0, "omega": cfg["omega"], "init_angle": 0.0},
            cfg["sim_time"],
            dt,
            method=method,
        )
        result = H.run(builder, compiled=False)
        t, y = result.signal("y")
        analytic = (1.0 - np.cos(cfg["omega"] * t)) / cfg["omega"]
        errors.append(H.max_abs_error(y, analytic))
        H.release(result)
    return errors, H.observed_order(errors)


def case_integration_order():
    """Halving ``dt`` must shrink the error by the method's own order.

    Run on the interpreter, which is where the per-block integration method
    actually applies: explicit Euler is first order, classical RK4 fourth. The
    measured quantity is the observed order itself, so the case fails both when
    a method loses accuracy and when it silently stops being the method asked
    for.
    """
    rows = []
    for method, expected, tol in EXPECTED_ORDERS:
        _errors, orders = integrator_order(method)
        rows.append(
            CaseResult(
                "Integrator {0}: observed order |p - {1:.0f}|".format(method, expected),
                "interpreter",
                method,
                "{0} -> {1}".format(ORDER_STEPS[0], ORDER_STEPS[-1]),
                max(abs(p - expected) for p in orders),
                tol,
                note="observed order {0}".format(", ".join("%.2f" % p for p in orders)),
            )
        )
    return rows


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def format_failures(rows):
    """One line per row that exceeded its tolerance; ``""`` when all passed."""
    return "\n".join(
        "{0} [{1}, {2}]: error {3:.3e} > tolerance {4:.1e}".format(
            row.name, row.path, row.method, row.error, row.tol
        )
        for row in rows
        if not row.passed
    )


ALL_CASES = (
    ("first_order_lag", case_first_order_lag),
    ("second_order_step", case_second_order_step),
    ("integrator_tracking", case_integrator_tracking),
    ("state_space_vs_expm", case_state_space_vs_expm),
    ("transfer_function_vs_lsim", case_transfer_function_vs_lsim),
    ("pid_closed_loop", case_pid_closed_loop),
    ("discrete_transfer_function", case_discrete_transfer_function),
    ("zero_order_hold", case_zero_order_hold),
    ("transport_delay", case_transport_delay),
    ("saturation_event", case_saturation_event),
    ("switch_event", case_switch_event),
    ("van_der_pol_stiff", case_van_der_pol_stiff),
    ("heat_eigenmode", case_heat_eigenmode),
    ("advection_pulse", case_advection_pulse),
    ("integration_order", case_integration_order),
)
