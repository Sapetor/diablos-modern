"""S1 - continuous plant under a digital controller (sampled-data loop).

S1a  examples/discrete_pi_zoh.diablos (ZOH 0.1 s -> discrete PI -> 2/((s+1)(s+2)))
     with the reference step moved from t=1.0 to t=0.93 so it does not land on
     a sampling instant (where "sample before or after the step" is a convention,
     not an accuracy question). Reference: exact lifted solution via expm.
S1b  nonlinear, stiff plant built from Integrators: damped pendulum driven
     through a fast first-order actuator (tau = 1 ms) under a discrete PI.
     Reference: piecewise Radau (rtol 1e-12) between sampling instants.

Any discrete sample time sends the whole DiaBloS diagram to the fixed-step
interpreter, so the DiaBloS axis is sim_dt; the PathSim axis is the adaptive
local-error tolerance. Each point is (wall time, max |error|) on the plant
output, with the reference evaluated at each tool's own output times.
"""

import json
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from common import REPO, diablos_run, diablos_signal, time_call, write_result

T_STEP = 0.93  # reference step time, off both the 0.1 s and 0.05 s sampling grids

# ------------------------------------------------------------------ references


def _pi_update(state, e, num, den):
    """Direct-form discrete TF y[k] = (num/den) e[k] for first-order num/den."""
    u_prev, e_prev = state
    u = (num[0] * e + num[1] * e_prev - den[1] * u_prev) / den[0]
    return (u, e), u


def reference_linear(query_t, t_end, T, num_c, den_c, num_d, den_d, delay):
    """Exact sampled-data response of a strictly proper LTI plant.

    ``delay`` = 0: u[k] computed from e[k] is applied on [kT, (k+1)T).
    ``delay`` = 1: the controller output is applied one period later.
    """
    A, B, C, _ = signal.tf2ss(num_c, den_c)
    n = A.shape[0]

    def phi_gamma(h):
        M = np.zeros((n + 1, n + 1))
        M[:n, :n], M[:n, n:] = A * h, B * h
        E = expm(M)
        return E[:n, :n], E[:n, n:]

    Phi, Gam = phi_gamma(T)
    n_k = int(np.ceil(t_end / T)) + 1
    xs, us = [], []
    x, st, u_applied_prev = np.zeros((n, 1)), (0.0, 0.0), 0.0
    for k in range(n_k):
        tk = k * T
        y = (C @ x).item()
        r = 1.0 if tk >= T_STEP else 0.0
        st, u_new = _pi_update(st, r - y, num_d, den_d)
        u = u_new if delay == 0 else u_applied_prev
        u_applied_prev = u_new
        xs.append(x.copy())
        us.append(u)
        x = Phi @ x + Gam * u
    out = np.empty_like(query_t)
    for i, t in enumerate(query_t):
        k = min(int(np.floor(t / T + 1e-12)), n_k - 1)
        P, G = phi_gamma(t - k * T)
        out[i] = (C @ (P @ xs[k] + G * us[k])).item()
    return out


def pendulum_rhs(t, x, u, tau):
    x1, x2, a = x
    return [x2, -np.sin(x1) - 0.5 * x2 + a, (u - a) / tau]


def reference_nonlinear(query_t, t_end, T, num_d, den_d, tau, delay):
    n_k = int(np.ceil(t_end / T - 1e-9))
    x = np.zeros(3)
    st, u_prev_new = (0.0, 0.0), 0.0
    out = np.full_like(query_t, np.nan)
    for k in range(n_k):
        tk, tk1 = k * T, (k + 1) * T
        r = 1.0 if tk >= T_STEP else 0.0
        st, u_new = _pi_update(st, r - x[0], num_d, den_d)
        u = u_new if delay == 0 else u_prev_new
        u_prev_new = u_new
        sol = solve_ivp(
            pendulum_rhs,
            (tk, tk1),
            x,
            method="Radau",
            rtol=1e-12,
            atol=1e-14,
            dense_output=True,
            args=(u, tau),
        )
        # The last interval absorbs float round-off at t_end.
        hi = np.inf if k == n_k - 1 else tk1 + 1e-12
        mask = (query_t >= tk - 1e-12) & (query_t <= hi)
        if mask.any():
            out[mask] = sol.sol(np.clip(query_t[mask], tk, tk1))[0]
        x = sol.y[:, -1]
    return out


def max_err(t, y, ref_fn):
    """Max error against both hold conventions; report the matching one."""
    errs = {d: float(np.max(np.abs(y - ref_fn(t, d)))) for d in (0, 1)}
    d = min(errs, key=errs.get)
    return errs[d], {"matched": d, "err_delay0": errs[0], "err_delay1": errs[1]}


# ------------------------------------------------------------------ S1a


S1A = {
    "T": 0.1,
    "num_c": [2.0],
    "den_c": [1.0, 3.0, 2.0],
    "num_d": [1.2, -1.0],
    "den_d": [1.0, -1.0],
    "t_end": 15.0,
}


def s1a_diablos_file(tmp):
    data = json.loads((REPO / "examples" / "discrete_pi_zoh.diablos").read_text())
    for b in data["blocks_data"]:
        if b.get("block_fn") == "Step":
            b["params"]["delay"] = T_STEP
    path = tmp / "s1a.diablos"
    path.write_text(json.dumps(data))
    return path


def s1a_pathsim(rtol, Solver):
    from pathsim import Connection, Simulation
    from pathsim.blocks import (
        Adder,
        DiscreteTransferFunction,
        SampleHold,
        Scope,
        Source,
        TransferFunctionNumDen,
    )

    p = S1A

    def once():
        r = Source(lambda t: 1.0 if t >= T_STEP else 0.0)
        e = Adder("+-")
        zoh = SampleHold(T=p["T"])
        pi = DiscreteTransferFunction(Num=p["num_d"], Den=p["den_d"], T=p["T"])
        G = TransferFunctionNumDen(Num=p["num_c"], Den=p["den_c"])
        sc = Scope()
        conns = [
            Connection(r, e[0]),
            Connection(G, e[1], sc[0]),
            Connection(e, zoh),
            Connection(zoh, pi),
            Connection(pi, G),
        ]
        sim = Simulation(
            [r, e, zoh, pi, G, sc],
            conns,
            dt=0.01,
            Solver=Solver,
            tolerance_lte_rel=rtol,
            tolerance_lte_abs=rtol * 1e-3,
            log=False,
        )
        t0 = time.perf_counter()
        sim.run(p["t_end"])
        el = time.perf_counter() - t0
        t, d = sc.read()
        return el, (np.asarray(t), np.asarray(d)[0])

    return time_call(once)


# ------------------------------------------------------------------ S1b


S1B = {
    "T": 0.05,
    "num_d": [1.025, -1.0],  # PI Kp=1, Ki=0.5
    "den_d": [1.0, -1.0],
    "tau": 1e-3,
    "t_end": 10.0,
}


def s1b_diablos_file(tmp, method):
    from lib.diagram_builder import DiagramBuilder

    p = S1B
    b = DiagramBuilder(sim_time=p["t_end"], sim_dt=1e-3)
    r = b.add_block("Step", 50, 100, name="r", params={"value": 1.0, "delay": T_STEP, "type": "up"})
    e = b.add_block("Sum", 120, 100, name="e", params={"sign": "+-"}, in_ports=2)
    zoh = b.add_block(
        "ZeroOrderHold",
        190,
        100,
        name="zoh",
        params={"sampling_time": p["T"]},
        in_ports=1,
        out_ports=1,
    )
    pi = b.add_block(
        "DiscreteTranFn",
        260,
        100,
        name="pi",
        params={"numerator": p["num_d"], "denominator": p["den_d"], "sampling_time": p["T"]},
        in_ports=1,
        out_ports=1,
    )
    ae = b.add_block("Sum", 330, 100, name="ae", params={"sign": "+-"}, in_ports=2)
    g = b.add_block("Gain", 400, 100, name="g", params={"gain": 1.0 / p["tau"]})
    integ = {"init_conds": 0.0, "method": method}
    a = b.add_block("Integrator", 470, 100, name="a", params=integ)
    f = b.add_block(
        "Function",
        540,
        100,
        name="f",
        params={"expression": "-sin(u[0]) - 0.5*u[1] + u[2]"},
        in_ports=3,
        out_ports=1,
    )
    x2 = b.add_block("Integrator", 610, 100, name="x2", params=integ)
    x1 = b.add_block("Integrator", 680, 100, name="x1", params=integ)
    sc = b.add_block(
        "Scope", 750, 100, name="scope", params={"labels": "x1"}, in_ports=1, out_ports=0
    )
    for src, dst, port in [
        (r, e, 0),
        (x1, e, 1),
        (e, zoh, 0),
        (zoh, pi, 0),
        (pi, ae, 0),
        (a, ae, 1),
        (ae, g, 0),
        (g, a, 0),
        (x1, f, 0),
        (x2, f, 1),
        (a, f, 2),
        (f, x2, 0),
        (x2, x1, 0),
        (x1, sc, 0),
    ]:
        b.connect(src, 0, dst, port)
    path = tmp / f"s1b_{method}.diablos"
    b.save(str(path))
    return path


def s1b_pathsim(rtol, Solver):
    from pathsim import Connection, Simulation
    from pathsim.blocks import (
        Adder,
        Amplifier,
        DiscreteTransferFunction,
        Function,
        Integrator,
        SampleHold,
        Scope,
        Source,
    )

    p = S1B

    def once():
        r = Source(lambda t: 1.0 if t >= T_STEP else 0.0)
        e = Adder("+-")
        zoh = SampleHold(T=p["T"])
        pi = DiscreteTransferFunction(Num=p["num_d"], Den=p["den_d"], T=p["T"])
        ae = Adder("+-")
        g = Amplifier(1.0 / p["tau"])
        a, x2, x1 = Integrator(0.0), Integrator(0.0), Integrator(0.0)
        f = Function(lambda q1, q2, qa: -np.sin(q1) - 0.5 * q2 + qa)
        sc = Scope()
        conns = [
            Connection(r, e[0]),
            Connection(x1, e[1], f[0], sc[0]),
            Connection(e, zoh),
            Connection(zoh, pi),
            Connection(pi, ae[0]),
            Connection(a, ae[1], f[2]),
            Connection(ae, g),
            Connection(g, a),
            Connection(x2, f[1], x1),
            Connection(f, x2),
        ]
        sim = Simulation(
            [r, e, zoh, pi, ae, g, a, f, x2, x1, sc],
            conns,
            dt=1e-3,
            Solver=Solver,
            tolerance_lte_rel=rtol,
            tolerance_lte_abs=rtol * 1e-3,
            log=False,
        )
        t0 = time.perf_counter()
        sim.run(p["t_end"])
        el = time.perf_counter() - t0
        t, d = sc.read()
        return el, (np.asarray(t), np.asarray(d)[0])

    return time_call(once)


# ------------------------------------------------------------------ driver


def main():
    from pathsim.solvers import ESDIRK43, RKDP54

    tmp = Path(tempfile.mkdtemp(prefix="diablos_s1_"))
    out = {}

    # S1a
    p = S1A

    def ref_a(t, d):
        return reference_linear(
            t, p["t_end"], p["T"], p["num_c"], p["den_c"], p["num_d"], p["den_d"], d
        )

    rows = []
    path = s1a_diablos_file(tmp)
    for dt in (0.01, 0.005, 0.001):
        timing, res = time_call(lambda dt=dt: diablos_run(path, p["t_end"], dt))
        t, y = diablos_signal(res, "plantoutput")  # harvester strips spaces
        err, conv = max_err(t, y, ref_a)
        rows.append(
            {
                "tool": "diablos",
                "knob": f"sim_dt={dt}",
                "path": res["solver_path"],
                "fallback_reason": res["fallback_reason"],
                "timing": timing,
                "max_err": err,
                "hold_delay": conv,
                "n_out": len(t),
            }
        )
        print("S1a", rows[-1])
    for Solver in (RKDP54, ESDIRK43):
        for rtol in (1e-4, 1e-6, 1e-8, 1e-10):
            timing, (t, y) = s1a_pathsim(rtol, Solver)
            err, conv = max_err(t, y, ref_a)
            rows.append(
                {
                    "tool": "pathsim",
                    "knob": f"{Solver.__name__} rtol={rtol:g}",
                    "timing": timing,
                    "max_err": err,
                    "hold_delay": conv,
                    "n_out": len(t),
                }
            )
            print("S1a", rows[-1])
    out["s1a_linear"] = {"params": p, "points": rows}

    # S1b
    p = S1B

    def ref_b(t, d):
        return reference_nonlinear(t, p["t_end"], p["T"], p["num_d"], p["den_d"], p["tau"], d)

    rows = []
    for method in ("SOLVE_IVP", "RK4"):
        path = s1b_diablos_file(tmp, method)
        for dt in (1e-3, 5e-4, 2.5e-4, 1e-4):
            timing, res = time_call(lambda dt=dt, path=path: diablos_run(path, p["t_end"], dt))
            if not res["ok"]:
                rows.append(
                    {"tool": "diablos", "knob": f"{method} sim_dt={dt}", "error": res["error"]}
                )
                print("S1b", rows[-1])
                continue
            t, y = diablos_signal(res, "x1")
            err, conv = max_err(t, y, ref_b)
            rows.append(
                {
                    "tool": "diablos",
                    "knob": f"{method} sim_dt={dt}",
                    "path": res["solver_path"],
                    "fallback_reason": res["fallback_reason"],
                    "timing": timing,
                    "max_err": err,
                    "hold_delay": conv,
                    "n_out": len(t),
                }
            )
            print("S1b", rows[-1])
    for Solver in (RKDP54, ESDIRK43):
        for rtol in (1e-4, 1e-6, 1e-8):
            timing, (t, y) = s1b_pathsim(rtol, Solver)
            err, conv = max_err(t, y, ref_b)
            rows.append(
                {
                    "tool": "pathsim",
                    "knob": f"{Solver.__name__} rtol={rtol:g}",
                    "timing": timing,
                    "max_err": err,
                    "hold_delay": conv,
                    "n_out": len(t),
                }
            )
            print("S1b", rows[-1])
    out["s1b_nonlinear_stiff"] = {"params": p, "points": rows}

    write_result("s1_sampled_data", out)


if __name__ == "__main__":
    main()
