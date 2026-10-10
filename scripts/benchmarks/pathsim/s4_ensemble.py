"""S4 - seeded Monte Carlo ensemble over a lossy, noisy feedback loop.

Model: examples/monte_carlo_robustness.diablos - continuous PI (Kp 2, Ki 1.5)
on 1/(s+1); the measurement y + N(0, 0.02^2) crosses a Bernoulli packet-loss
link (p = 0.2, sampled every 0.05 s, hold on drop) back to the controller.

The two tools draw from different RNG streams, so trajectories are not
comparable run by run. The legs are therefore:
  * wall time for an N-run ensemble (per-run setup included on both sides,
    because that is what an experiment costs);
  * reproducibility: same master seed twice -> bitwise identical ensemble;
    different master seed -> different ensemble;
  * statistical agreement: ensemble-mean trajectories should agree within
    a few standard errors.
"""

import time

import numpy as np

from common import REPO, write_result

N_RUNS = 50
T_END, DT = 20.0, 0.01
LOSS_P, T_LINK, SIGMA = 0.2, 0.05, 0.02
EXAMPLE = REPO / "examples" / "monte_carlo_robustness.diablos"


def diablos_ensemble(master_seed):
    from lib.analysis.monte_carlo import MonteCarloRunner
    from lib.cli import load_diagram

    dsim, _ = load_diagram(str(EXAMPLE))
    dsim.use_fast_solver = True
    t0 = time.perf_counter()
    res = MonteCarloRunner(dsim).run(N_RUNS, master_seed=master_seed, sim_time=T_END, sim_dt=DT)
    el = time.perf_counter() - t0
    sig = res["signals"]["measuredoutput"]
    return el, res["timeline"], sig["runs"]


def pathsim_ensemble(master_seed, solver="RK4", rtol=None, n_runs=None):
    from pathsim import Connection, Simulation
    from pathsim.blocks import PID, Adder, Block, Scope, Source, TransferFunctionNumDen
    from pathsim.events import Schedule
    from pathsim import solvers

    n_runs = N_RUNS if n_runs is None else n_runs
    tol = {} if rtol is None else {"tolerance_lte_rel": rtol, "tolerance_lte_abs": rtol * 1e-3}

    class LossyLink(Block):
        """Sample y + noise every T; deliver with prob 1-p, else hold."""

        def __init__(self, rng):
            super().__init__()

            def _sample(t):
                if rng.random() >= LOSS_P:
                    y = self.inputs.to_array() + SIGMA * rng.standard_normal()
                    self.outputs.update_from_array(y)

            self.events = [Schedule(t_start=0.0, t_period=T_LINK, func_act=_sample)]

        def __len__(self):
            return 0

    t_grid = np.arange(0.0, T_END + DT / 2, DT)
    runs = []
    t0 = time.perf_counter()
    for i in range(n_runs):
        rng = np.random.default_rng([master_seed, i])
        r = Source(lambda t: 1.0 if t >= 1.0 else 0.0)
        e = Adder("+-")
        pid = PID(Kp=2.0, Ki=1.5, Kd=0.0)
        G = TransferFunctionNumDen(Num=[1.0], Den=[1.0, 1.0])
        link = LossyLink(rng)
        sc = Scope()
        sim = Simulation(
            [r, e, pid, G, link, sc],
            [
                Connection(r, e[0]),
                Connection(e, pid),
                Connection(pid, G),
                Connection(G, link),
                Connection(link, e[1], sc[0]),
            ],
            dt=DT,
            Solver=getattr(solvers, solver),
            log=False,
            **tol,
        )
        sim.run(T_END)
        t, d = sc.read()
        # Held signal: previous-value interpolation onto the common grid.
        idx = np.searchsorted(np.asarray(t), t_grid, side="right") - 1
        runs.append(np.asarray(d)[0][np.clip(idx, 0, None)])
    el = time.perf_counter() - t0
    return el, t_grid, np.vstack(runs)


def legs(fn):
    el_a, t, A = fn(12345)
    el_b, _, B = fn(12345)
    _, _, C = fn(999)
    return (
        {
            "wall_s": [el_a, el_b],
            "per_run_ms": 1e3 * min(el_a, el_b) / N_RUNS,
            "same_seed_bitwise_identical": bool(np.array_equal(A, B)),
            "different_seed_differs": not np.array_equal(A, C),
            "final_mean": float(A[:, -1].mean()),
            "final_std": float(A[:, -1].std()),
        },
        t,
        A,
    )


# Per-run cost of one PathSim run vs solver setting (DiaBloS runs fixed-step).
SENSITIVITY = [("RK4", None), ("SSPRK22", None), ("RKDP54", 1e-6), ("RKDP54", 1e-9)]


def main():
    d_legs, td, D = legs(diablos_ensemble)
    # Headline leg: fixed-step RK4 at the same dt as DiaBloS's interpreter.
    p_legs, tp, P = legs(pathsim_ensemble)
    sensitivity = {}
    for solver, rtol in SENSITIVITY:
        el, _, _ = pathsim_ensemble(1, solver=solver, rtol=rtol, n_runs=3)
        sensitivity[f"{solver}" + (f" rtol={rtol:g}" if rtol else f" fixed dt={DT}")] = 1e3 * el / 3
    # Compare ensemble means on the DiaBloS grid after the transient.
    L = min(len(td), len(tp))
    mask = td[:L] >= 5.0
    diff = np.abs(D[:, :L].mean(0) - P[:, :L].mean(0))[mask]
    se = np.sqrt(D[:, :L].var(0) / N_RUNS + P[:, :L].var(0) / N_RUNS)[mask]
    agreement = {
        "max_abs_mean_diff_t_ge_5": float(diff.max()),
        "max_diff_in_standard_errors": float((diff / np.maximum(se, 1e-12)).max()),
        "frac_within_3se": float(np.mean(diff <= 3 * se)),
    }
    out = {
        "params": {
            "n_runs": N_RUNS,
            "t_end": T_END,
            "dt": DT,
            "loss_p": LOSS_P,
            "t_link": T_LINK,
            "sigma": SIGMA,
        },
        "diablos": d_legs,
        "pathsim_rk4_fixed": p_legs,
        "agreement": agreement,
        "pathsim_per_run_ms_by_solver": sensitivity,
    }
    print(out)
    write_result("s4_ensemble", out)


if __name__ == "__main__":
    main()
