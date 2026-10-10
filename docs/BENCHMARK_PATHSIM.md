# DiaBloS vs PathSim: head-to-head benchmark

Measured 2026-10-10 on DiaBloS `b21f236`, PathSim 0.27.0 (MIT), numpy 2.5.3,
scipy 1.18.1, Python 3.12.3, AMD Ryzen 7 9800X3D under WSL2. Harness:
`scripts/benchmarks/pathsim/` (one script per scenario, raw results in
`out/*.json`). Timings are the median of 5 runs (3 for S3) after one warm-up
and cover only the simulate call -- not diagram loading, Qt start-up or model
construction.

Accuracy is always measured against an independent reference (closed form,
exact `expm` discretization, or tight-tolerance `solve_ivp`), never against
either tool, and the reference is evaluated at each tool's own output times.

## Summary

| Scenario | DiaBloS | PathSim | Takeaway |
|---|---|---|---|
| S1a linear plant + digital PI | exact (2e-14), 0.09 s, interpreter | 6e-12 at 0.1 s (RKDP54) | Fallback costs nothing for linear plants, **but DiaBloS applies the control one sample late** |
| S1b nonlinear stiff plant + digital PI | **1st order**: 1.6e-3 (3.2 s) ... 3e-5 (40 s) | 1.7e-10 at 0.9 s; 1.9e-13 at 1.8 s | The interpreter fallback is the real accuracy gap |
| S2 algebraic loop | rejected; delay workaround diverges or is wrong unless the loop contracts | solved to 3e-13, 2 ms | Genuine capability gap |
| S3 delayed platoon, N=50 | 1.8e-2 at 16 s | 2.0e-3 at 11 s; 2.4e-5 at 109 s | Comparable; delays blunt adaptive stepping |
| S4 50-run seeded ensemble | 163 ms/run, bitwise reproducible | 164 ms/run (RK4, same dt), bitwise reproducible | Parity at matched settings |

## S1 - digital controller on a continuous plant

Any block with a discrete sample time makes the whole diagram fall back to
the fixed-step interpreter (recorded reason: `ZeroOrderHold: it has a
discrete sample time`). The step reference is placed at t=0.93 s, off every
sampling grid, so "sample before or after the step" is not a factor.

**S1a** (`examples/discrete_pi_zoh.diablos`: ZOH 0.1 s -> PI -> 2/((s+1)(s+2))).

| Tool | Setting | Wall | Max error |
|---|---|---|---|
| DiaBloS | sim_dt = 0.01 / 0.005 / 0.001 | 0.09 / 0.17 / 0.77 s | 2e-14 / 3e-14 / 2e-13 (*) |
| PathSim RKDP54 | rtol 1e-4 / 1e-6 / 1e-8 / 1e-10 | 0.02 / 0.03 / 0.05 / 0.11 s | 8e-8 / 4e-8 / 6e-10 / 6e-12 |
| PathSim ESDIRK43 | rtol 1e-4 ... 1e-10 | 0.19 ... 1.8 s | 8e-7 ... 4e-9 |

(*) Against the reference **with one controller period of extra delay**.
Against the standard semantics (u[k] computed from e[k] is applied on
[kT, (k+1)T), what Simulink and PathSim do) the DiaBloS error is 0.106.
`TranFn` is discretized exactly (`cont2discrete`) in the interpreter, so for
a linear plant the fallback costs no accuracy.

**Finding: a closed-loop sampled controller acts one period late in DiaBloS.**
In the run above `u` changes at t~1.10 instead of 1.00. Open-loop
ZOH -> DiscreteTranFn shows no delay, so this comes from the closed-loop
execution order. It is not the documented one-`sim_dt` interpreter delay
(`docs/SOLVER_SEMANTICS.md` 4.2): it is a full sample period T and adds
phase lag the user did not model. In S1b it raises the peak from 1.26 to
1.34. The mechanism has not been traced yet.

**S1b** (pendulum `x1'' = -sin x1 - 0.5 x1' + a`, actuator `a' = (u-a)/1e-3`,
built from `Integrator` blocks; discrete PI Kp=1, Ki=0.5, T=0.05 s; 10 s).

| Tool | Setting | Wall | Max error |
|---|---|---|---|
| DiaBloS, Integrator SOLVE_IVP (default) | sim_dt 1e-3 / 5e-4 / 2.5e-4 / 1e-4 | 3.2 / 5.6 / 11.1 / 25.8 s | 1.6e-3 / 7.9e-4 / 4.0e-4 / 1.6e-4 |
| DiaBloS, Integrator RK4 | sim_dt 1e-3 / 5e-4 / 2.5e-4 / 1e-4 | 4.0 / 7.9 / 15.5 / 39.9 s | 3.1e-4 / 1.6e-4 / 7.8e-5 / 3.1e-5 |
| PathSim RKDP54 | rtol 1e-4 / 1e-6 / 1e-8 | 0.9 / 1.2 / 1.8 s | 1.7e-10 / 1.8e-12 / 1.9e-13 |
| PathSim ESDIRK43 | rtol 1e-4 / 1e-6 / 1e-8 | 4.6 / 5.0 / 8.2 s | 4.1e-9 / 3.1e-10 / 1.7e-10 |

(DiaBloS errors are again against the one-period-delay reference, so they
isolate integration error from the semantics finding above.)

Both interpreter integrator modes converge at **first order**. The
interpreter holds each block's input constant over a step: the SOLVE_IVP mode
integrates `x' = u_held` exactly, and the RK4 label does not give fourth
order in a closed loop. Reaching 3e-5 takes 40 s, while PathSim reaches 2e-12
in 1.2 s. For an explicit solver the 1 ms actuator is only mildly stiff over
this horizon; the implicit ESDIRK43 does not pay off here.

*Roadmap implication:* this is the strongest result for "adaptive integration
between scheduled sample instants". A compiled hybrid path -- `solve_ivp`
from sample hit to sample hit, discrete updates as events -- would bring
S1b to PathSim-class accuracy. The one-period delay should be fixed first,
because the hybrid path must reproduce the standard semantics and the current
interpreter would then disagree with it.

## S2 - algebraic loops

`y = u - f(y)`, u = 1, solved directly in PathSim. DiaBloS rejects every case
(`Algebraic loop detected involving blocks: [...]`), so its column is the
workaround a user would try: a one-step `Delay` in the feedback path.

| Loop | Exact y | DiaBloS + Delay | PathSim |
|---|---|---|---|
| f(y) = 0.5 y | 0.6667 | exact (contraction) | exact |
| f(y) = 3 y | 0.25 | diverges (1e95) | exact |
| f(y) = tanh(3 y) | 0.2934 | wrong: period-2 oscillation between 0.006 and 0.984 | 3e-13 |

The delay turns the loop into the fixed-point iteration y <- u - f(y), which
only converges when |f'(y*)| < 1. Run time is negligible on both sides
(~10 ms vs ~2 ms).

*Roadmap implication:* a Newton / accelerated fixed-point solve of strongly
connected algebraic components is a self-contained, high-value addition.
PathSim's `NewtonAnderson` booster is a working MIT-licensed example to
compare against.

## S3 - delayed platoon, scaling in N

Predecessor-following double integrators,
`a_i = 1 (p_{i-1}(t-0.1) - p_i) + 2 (v_{i-1}(t-0.1) - v_i)`, leader `p0 = t`,
20 s. Reference: method of steps with DOP853 at rtol 1e-12. `TransportDelay`
also forces the interpreter (`it has no compiled kernel`).

| N (blocks) | DiaBloS dt=0.01 | DiaBloS dt=0.001 | PathSim rtol 1e-6 | PathSim rtol 1e-9 |
|---|---|---|---|---|
| 5 (48) | 1.5 s, 1.8e-2 | 14.9 s, 9.9e-4 | 0.8 s, 7.5e-2 | 3.2 s, 2.6e-5 |
| 10 (93) | 2.8 s, 3.8e-2 | - | 1.7 s, 9.0e-2 | 10.4 s, 1.2e-4 |
| 20 (183) | 5.6 s, 2.4e-1 | 56.5 s, 2.1e-2 | 3.7 s, 1.4e-2 | 35.0 s, 4.3e-4 |
| 50 (453) | 16.3 s, 1.8e-2 | - | 10.7 s, 2.0e-3 | 108.6 s, 2.4e-5 |

(wall time, max error on the last vehicle's position)

Both tools scale roughly linearly in N. The leader's velocity step reaches
each follower as a delayed discontinuity that neither solver is told about,
so adaptive stepping buys PathSim much less here than in S1. At loose
tolerance PathSim can even be less accurate than DiaBloS at dt=0.01 (N=5,
10). Neither tool is clearly better here. Nothing in this scenario argues
for an engine swap.

## S4 - seeded Monte Carlo ensemble

`examples/monte_carlo_robustness.diablos` (PI on 1/(s+1), noisy measurement
over a 20 % Bernoulli packet-loss link sampled at 0.05 s, hold on drop), 50
runs x 20 s. DiaBloS uses `MonteCarloRunner`. PathSim is a plain Python loop
with a custom scheduled-event loss block and `default_rng([seed, i])`. Per-run
setup is included on both sides.

| | DiaBloS | PathSim (RK4, dt 0.01) |
|---|---|---|
| Per run | 163 ms | 164 ms |
| Same master seed twice | bitwise identical | bitwise identical |
| Different master seed | differs | differs |
| Final value mean / std | 1.002 / 0.020 | 0.999 / 0.021 |

Ensemble means agree within 3 standard errors at every t >= 5 s (the RNG
streams differ, so only statistics are comparable). PathSim's per-run cost
depends strongly on the solver: SSPRK22 fixed 92 ms, RK4 fixed 162 ms,
RKDP54 rtol 1e-6 1.1 s, rtol 1e-9 3.7 s. Tight adaptive tolerances are
expensive when sample events restart the step every 50 ms. Building the same
experiment in PathSim took about 40 lines, including a custom block. DiaBloS
gives it as a menu action with ensemble plots.

## What this means for the build-vs-adopt question

- The external analysis's two biggest claims hold up under measurement:
  sampled-data accuracy (S1b) and algebraic loops (S2) are real gaps, not
  cosmetic ones.
- Two of its implicit worries do not: for linear plants the interpreter
  fallback is exact (S1a), and seeded ensembles are already on par in cost
  and reproducibility (S4). Delayed multi-agent models (S3) are a wash.
- New and more urgent than either: the **one-period closed-loop sampling
  delay** (S1). It silently changes every digital-control result DiaBloS
  produces and should be fixed before any engine rework.
- Suggested order: (1) fix the sampling delay and add a regression test
  against the S1a exact reference; (2) hybrid compiled path; (3) algebraic
  loop solver. A PathSim backend adapter would get (2) and (3) at once. The
  S1/S2 scripts here are the acceptance tests for any of these routes.

## Reproducing

```bash
python3 -m venv ~/.venvs/diablos-bench
~/.venvs/diablos-bench/bin/pip install -r requirements.txt -r scripts/benchmarks/pathsim/requirements.txt
cd scripts/benchmarks/pathsim
~/.venvs/diablos-bench/bin/python s1_sampled_data.py   # ~6 min
~/.venvs/diablos-bench/bin/python s2_algebraic_loop.py # seconds
DIABLOS_BENCH_REPEATS=3 ~/.venvs/diablos-bench/bin/python s3_platoon.py  # ~25 min
~/.venvs/diablos-bench/bin/python s4_ensemble.py       # ~2 min
```

`DIABLOS_BENCH_REPEATS` sets the timed repeats (default 5), and
`DIABLOS_BENCH_OUT` the output directory.
