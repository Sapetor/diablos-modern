"""S3 - delayed platoon, scaling in the number of vehicles N.

Leader: p0 = t, v0 = 1 (t >= 0). Follower i (double integrator):
    a_i = KP * (p_{i-1}(t - TAU) - p_i) + KD * (v_{i-1}(t - TAU) - v_i)
i.e. predecessor-following with a communication delay TAU. Zero history.

Reference: method of steps -- on each [k TAU, (k+1) TAU] the delayed terms
come from the previous interval's dense solution, integrated with DOP853 at
rtol 1e-12. Error is reported on the last vehicle's position.
"""

import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from common import diablos_run, diablos_signal, time_call, write_result

KP, KD, TAU, T_END, DT = 1.0, 2.0, 0.1, 20.0, 0.01
SIZES = (5, 10, 20, 50)


def reference(n, query_t):
    """States [p1, v1, ..., pn, vn]; returns p_n at ``query_t``."""
    pieces = []  # (t0, t1, dense) per interval

    def hist(t):
        if t <= 0.0:
            return np.zeros(2 * n)
        for t0, t1, s in reversed(pieces):
            if t0 - 1e-14 <= t <= t1 + 1e-14:
                return s(t)
        raise ValueError(t)

    def rhs(t, x):
        xd = hist(t - TAU)
        dx = np.empty_like(x)
        for i in range(n):
            p, v = x[2 * i], x[2 * i + 1]
            if i == 0:
                pd, vd = max(t - TAU, 0.0), (1.0 if t - TAU >= 0 else 0.0)
            else:
                pd, vd = xd[2 * (i - 1)], xd[2 * (i - 1) + 1]
            dx[2 * i] = v
            dx[2 * i + 1] = KP * (pd - p) + KD * (vd - v)
        return dx

    x = np.zeros(2 * n)
    k = 0
    while k * TAU < T_END - 1e-12:
        t0, t1 = k * TAU, min((k + 1) * TAU, T_END)
        sol = solve_ivp(
            rhs, (t0, t1), x, method="DOP853", rtol=1e-12, atol=1e-14, dense_output=True
        )
        pieces.append((t0, t1, sol.sol))
        x = sol.y[:, -1]
        k += 1
    # Clamp to the integrated span: output grids can overshoot T_END by round-off.
    return np.array([hist(min(t, T_END))[2 * (n - 1)] for t in query_t])


def build_diablos(n, path):
    from lib.diagram_builder import DiagramBuilder

    b = DiagramBuilder(sim_time=T_END, sim_dt=DT)
    prev_p = b.add_block("Ramp", 50, 50, name="p0", params={"slope": 1.0, "delay": 0.0})
    prev_v = b.add_block("Constant", 50, 120, name="v0", params={"value": 1.0})
    for i in range(1, n + 1):
        y = 100 * i
        dp = b.add_block(
            "TransportDelay",
            150,
            y,
            name=f"dp{i}",
            params={"delay_time": TAU, "initial_value": 0.0},
        )
        dv = b.add_block(
            "TransportDelay",
            150,
            y + 40,
            name=f"dv{i}",
            params={"delay_time": TAU, "initial_value": 0.0},
        )
        ep = b.add_block("Sum", 250, y, name=f"ep{i}", params={"sign": "+-"}, in_ports=2)
        ev = b.add_block("Sum", 250, y + 40, name=f"ev{i}", params={"sign": "+-"}, in_ports=2)
        kp = b.add_block("Gain", 330, y, name=f"kp{i}", params={"gain": KP})
        kd = b.add_block("Gain", 330, y + 40, name=f"kd{i}", params={"gain": KD})
        acc = b.add_block("Sum", 410, y, name=f"a{i}", params={"sign": "++"}, in_ports=2)
        v = b.add_block("Integrator", 490, y, name=f"v{i}", params={"init_conds": 0.0})
        p = b.add_block("Integrator", 570, y, name=f"p{i}", params={"init_conds": 0.0})
        for src, dst, port in [
            (prev_p, dp, 0),
            (prev_v, dv, 0),
            (dp, ep, 0),
            (p, ep, 1),
            (dv, ev, 0),
            (v, ev, 1),
            (ep, kp, 0),
            (ev, kd, 0),
            (kp, acc, 0),
            (kd, acc, 1),
            (acc, v, 0),
            (v, p, 0),
        ]:
            b.connect(src, 0, dst, port)
        prev_p, prev_v = p, v
    sc = b.add_block(
        "Scope", 700, 100, name="scope", params={"labels": "plast"}, in_ports=1, out_ports=0
    )
    b.connect(prev_p, 0, sc, 0)
    b.save(str(path))


def run_pathsim(n, rtol):
    from pathsim import Connection, Simulation
    from pathsim.blocks import Adder, Amplifier, Constant, Delay, Integrator, Scope, Source
    from pathsim.solvers import RKDP54

    def once():
        blocks, conns = [], []
        prev_p, prev_v = Source(lambda t: max(t, 0.0)), Constant(1.0)
        blocks += [prev_p, prev_v]
        for _ in range(n):
            dp, dv = Delay(TAU), Delay(TAU)
            ep, ev, acc = Adder("+-"), Adder("+-"), Adder("++")
            kp, kd = Amplifier(KP), Amplifier(KD)
            v, p = Integrator(0.0), Integrator(0.0)
            blocks += [dp, dv, ep, ev, acc, kp, kd, v, p]
            conns += [
                Connection(prev_p, dp),
                Connection(prev_v, dv),
                Connection(dp, ep[0]),
                Connection(dv, ev[0]),
                Connection(p, ep[1]),
                Connection(v, ev[1], p),
                Connection(ep, kp),
                Connection(ev, kd),
                Connection(kp, acc[0]),
                Connection(kd, acc[1]),
                Connection(acc, v),
            ]
            prev_p, prev_v = p, v
        sc = Scope()
        blocks.append(sc)
        conns.append(Connection(prev_p, sc))
        sim = Simulation(
            blocks,
            conns,
            dt=DT,
            Solver=RKDP54,
            tolerance_lte_rel=rtol,
            tolerance_lte_abs=rtol * 1e-3,
            log=False,
        )
        t0 = time.perf_counter()
        sim.run(T_END)
        el = time.perf_counter() - t0
        t, d = sc.read()
        return el, (np.asarray(t), np.asarray(d)[0])

    return time_call(once)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="diablos_s3_"))
    rows = []
    for n in SIZES:
        path = tmp / f"platoon_{n}.diablos"
        build_diablos(n, path)
        # A finer step only where it stays affordable (cost is linear in 1/dt).
        for dt in (DT, 0.001) if n in (5, 20) else (DT,):
            timing, res = time_call(lambda path=path, dt=dt: diablos_run(path, T_END, dt))
            if res["ok"]:
                t, y = diablos_signal(res, "plast")
                err = float(np.max(np.abs(y - reference(n, t))))
            else:
                err = None
            rows.append(
                {
                    "tool": "diablos",
                    "n": n,
                    "knob": f"sim_dt={dt}",
                    "n_blocks": 2 + 9 * n + 1,
                    "path": res["solver_path"],
                    "fallback_reason": res["fallback_reason"],
                    "error": res["error"],
                    "timing": timing,
                    "max_err_last_pos": err,
                }
            )
            print(rows[-1])
        for rtol in (1e-6, 1e-9):
            timing, (t, y) = run_pathsim(n, rtol)
            rows.append(
                {
                    "tool": "pathsim",
                    "n": n,
                    "knob": f"RKDP54 rtol={rtol:g}",
                    "timing": timing,
                    "max_err_last_pos": float(np.max(np.abs(y - reference(n, t)))),
                }
            )
            print(rows[-1])
    write_result(
        "s3_platoon",
        {"params": {"kp": KP, "kd": KD, "tau": TAU, "t_end": T_END, "dt": DT}, "points": rows},
    )


if __name__ == "__main__":
    main()
