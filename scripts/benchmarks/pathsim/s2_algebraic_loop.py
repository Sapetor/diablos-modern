"""S2 - algebraic loops.

Linear loop      y = u - k*y          (closed form y = u / (1 + k))
Nonlinear loop   y = u - tanh(k*y)    (reference: scalar brentq root)

DiaBloS detects and rejects a genuine algebraic loop, so the DiaBloS side
records the rejection verbatim and then runs the workaround a user would
reach for: a one-step ``Delay`` in the feedback path, which turns the loop
into the fixed-point iteration y[n] = u - f(y[n-1]). PathSim solves the loop
directly with its accelerated fixed-point "boosters".
"""

import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from common import diablos_run, diablos_signal, time_call, write_result

U, K, T_END, DT = 1.0, 3.0, 2.0, 0.01

# k = 3 makes |f'(y*)| > 1, where the delayed (fixed-point) workaround cannot
# converge; k = 0.5 is the contraction case where it should.
CASES = {
    "linear_k0.5": {
        "expr": "0.5*u[0]",
        "f": lambda y: 0.5 * y,
    },
    "linear": {
        "expr": f"{K}*u[0]",
        "f": lambda y: K * y,
    },
    "tanh": {
        "expr": f"tanh({K}*u[0])",
        "f": lambda y: np.tanh(K * y),
    },
}


def reference(f):
    return brentq(lambda y: y - (U - f(y)), -10.0, 10.0, xtol=1e-15)


def build_diablos(expr, with_delay, path):
    from lib.diagram_builder import DiagramBuilder

    b = DiagramBuilder(sim_time=T_END, sim_dt=DT)
    u = b.add_block("Constant", 50, 100, name="u", params={"value": U})
    s = b.add_block("Sum", 150, 100, name="sum", params={"sign": "+-"}, in_ports=2)
    f = b.add_block("Function", 250, 200, name="f", params={"expression": expr})
    sc = b.add_block(
        "Scope", 350, 100, name="scope", params={"labels": "y"}, in_ports=1, out_ports=0
    )
    b.connect(u, 0, s, 0)
    b.connect(s, 0, f, 0)
    if with_delay:
        d = b.add_block(
            "Delay", 250, 300, name="z1", params={"delay_steps": 1, "initial_value": 0.0}
        )
        b.connect(f, 0, d, 0)
        b.connect(d, 0, s, 1)
    else:
        b.connect(f, 0, s, 1)
    b.connect(s, 0, sc, 0)
    b.save(str(path))


def run_pathsim(f_expr_fn):
    from pathsim import Connection, Simulation
    from pathsim.blocks import Adder, Constant, Function, Scope

    def once():
        u = Constant(U)
        s = Adder("+-")
        f = Function(f_expr_fn)
        sc = Scope()
        sim = Simulation(
            [u, s, f, sc],
            [Connection(u, s[0]), Connection(s, f, sc), Connection(f, s[1])],
            dt=DT,
            log=False,
        )
        t0 = time.perf_counter()
        sim.run(T_END)
        el = time.perf_counter() - t0
        t, d = sc.read()
        return el, (np.asarray(t), np.asarray(d)[0])

    return time_call(once)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="diablos_s2_"))
    out = {}
    for name, case in CASES.items():
        y_ref = reference(case["f"])
        row = {"y_ref": y_ref}

        # DiaBloS, direct loop: expect rejection.
        direct = tmp / f"{name}_direct.diablos"
        build_diablos(case["expr"], with_delay=False, path=direct)
        _, res = diablos_run(direct, T_END, DT)
        row["diablos_direct"] = {
            "ok": res["ok"],
            "error": res["error"],
            "solver_path": res["solver_path"],
        }

        # DiaBloS workaround: unit delay in the feedback path.
        delayed = tmp / f"{name}_delay.diablos"
        build_diablos(case["expr"], with_delay=True, path=delayed)
        timing, res = time_call(lambda p=delayed: diablos_run(p, T_END, DT))
        if res["ok"]:
            t, y = diablos_signal(res, "y")
            row["diablos_delay_workaround"] = {
                "timing": timing,
                "solver_path": res["solver_path"],
                "y_final": float(y[-1]),
                "err_final": float(abs(y[-1] - y_ref)),
                # For |f'(y*)| >= 1 the delayed iteration diverges or oscillates.
                "err_max_second_half": float(np.max(np.abs(y[len(y) // 2 :] - y_ref))),
            }
        else:
            row["diablos_delay_workaround"] = {"ok": False, "error": res["error"]}

        # PathSim, direct loop.
        timing, (t, y) = run_pathsim(case["f"])
        row["pathsim"] = {
            "timing": timing,
            "y_final": float(y[-1]),
            "err_max": float(np.max(np.abs(y - y_ref))),
        }
        out[name] = row
        print(name, row)
    write_result(
        "s2_algebraic_loop", {"params": {"u": U, "k": K, "t_end": T_END, "dt": DT}, "cases": out}
    )


if __name__ == "__main__":
    main()
