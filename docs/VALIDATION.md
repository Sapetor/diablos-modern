# Numerical Validation

DiaBloS results are checked against solutions computed **outside DiaBloS**: a
closed-form expression where one exists, and otherwise an independent numerical
reference (`scipy.linalg.expm`, `scipy.signal.lsim` / `dlsim`, or
`scipy.integrate.solve_ivp` run at a tolerance two orders tighter than the case
under test). Every case states its own tolerance, and the table below is
regenerated from a real run rather than transcribed.

```bash
pytest tests/validation -q          # assert every case is inside tolerance
python scripts/validation_report.py # the same cases, printed as this table
```

The suite takes about 8 seconds.

## What is being validated

Two independent execution engines produce DiaBloS results, and a case is run
through both wherever both apply:

* **Compiled** (`docs/FAST_SOLVER.md`) — `SystemCompiler` folds the whole
  diagram into one ODE right-hand side which `scipy.integrate.solve_ivp`
  integrates adaptively, optionally with located zero crossings. This is the
  accurate path, and its tolerances are correspondingly tight.
* **Interpreter** — a fixed-step loop that calls each block's `execute()` once
  per `sim_dt`, with per-block integration methods. Blocks that sample their
  input once per step (`Integrator` under `SOLVE_IVP`, `TranFn` under a
  time-varying input) are first-order accurate by construction, so their
  tolerances are looser *and the case says why*. A block whose discretization
  is exact — the interpreter's `TranFn` and `StateSpace` use an exact
  zero-order-hold — is held to machine precision instead.

Some cases are single-path by nature: a discrete rate (`DiscreteTranFn`,
`ZeroOrderHold`) or a `TransportDelay` forces the interpreter, since the
compiled right-hand side has no notion of sample instants or of a signal's
history; the stiff case is compiled-only, because a fixed-step explicit method
at `mu = 1000` is step-limited by stability rather than by accuracy and would be
a different experiment.

## Methodology

**References.** Nothing is compared against a previous DiaBloS run. The
references are:

| Case family | Reference |
| --- | --- |
| First- and second-order step responses | closed form, all three damping regimes |
| Integrator of a ramp / sine | closed form |
| StateSpace under a step | exact ZOH via `scipy.linalg.expm` of the augmented `[[A,B],[0,0]]` |
| TranFn under a sine | `scipy.signal.lsim` |
| PID closed loop | `CP/(1+CP)` formed by polynomial algebra, simulated with `lsim` |
| Discrete transfer function | `scipy.signal.dlsim` |
| Zero-order hold, transport delay | closed form (staircase, shifted sine) |
| Saturation / Switch corners | closed form, with the switching instant known exactly |
| Van der Pol, `mu = 1000` | `solve_ivp(method="Radau", rtol=1e-10, atol=1e-12)` |
| 1-D heat eigenmode | closed form `e^{-alpha (pi/L)^2 t}` |
| 1-D advection of a Gaussian | closed form (rigid translation) |

**Error measure.** Each case reduces to one scalar: the maximum absolute
difference between the DiaBloS trace and the reference on the common output
grid, except where the table says otherwise (the heat row is relative to the
initial amplitude; the mass-conservation row is a relative drift; the "observed
order" rows are `|p - p_expected|`).

**Convergence orders.** A tolerance at one step size says a run was accurate;
an order says *why*, and catches a method that quietly stops being the method
it is named after. Three cases measure one:

* the interpreter's `FWD_EULER` and `RK4` integrating `sin(wt)` at `dt`, `dt/2`
  and `dt/4` — observed 1.00 and 4.00;
* the 1-D heat block's spatial discretization on `N = 21, 41, 81` — observed
  2.00, which is the second-order central difference resolving the Laplacian
  eigenvalue to `O(dx^2)`;
* the 1-D advection block on `N = 101, 201, 401` — observed ≈ 1.97, the
  second-order upwind stencil.

**Tolerances.** Set from measured behaviour with roughly an order of magnitude
of headroom: tight enough that a real regression trips them, loose enough that
floating-point noise and scipy's own version-to-version drift do not. They live
next to the case in `tests/validation/_cases.py`, not scattered through the
tests.

**Reference self-checks.** A validation suite whose reference is wrong
validates nothing, so several references are themselves pinned: the
second-order closed forms are substituted back into `y'' + 2 zeta wn y' + wn^2 y
= wn^2`; the `expm` discretization is checked against the textbook
`A^{-1}(Ad - I)B`; the PID closed loop is checked for stability and unit DC
gain; the Van der Pol case asserts its own Jacobian is stiff.

## Results

| Case | Path | Method | dt / grid | Max error | Tolerance | Pass |
| --- | --- | --- | --- | ---: | ---: | :---: |
| First-order lag K/(tau s+1), unit step | compiled | RK45 | 0.005 | 3.86e-10 | 1.0e-07 | yes |
| First-order lag K/(tau s+1), unit step | interpreter | exact ZOH | 0.005 | 2.00e-15 | 1.0e-09 | yes |
| Second-order step, zeta=0.3 | compiled | RK45 | 0.01 | 4.21e-10 | 1.0e-07 | yes |
| Second-order step, zeta=0.3 | interpreter | exact ZOH | 0.01 | 8.22e-15 | 1.0e-09 | yes |
| Second-order step, zeta=1.0 | compiled | RK45 | 0.01 | 1.16e-10 | 1.0e-07 | yes |
| Second-order step, zeta=1.0 | interpreter | exact ZOH | 0.01 | 9.99e-16 | 1.0e-09 | yes |
| Second-order step, zeta=2.0 | compiled | RK45 | 0.01 | 1.91e-10 | 1.0e-07 | yes |
| Second-order step, zeta=2.0 | interpreter | exact ZOH | 0.01 | 2.11e-15 | 1.0e-09 | yes |
| Integrator of a ramp -> a t^2/2 | compiled | RK45 | 0.01 | 1.24e-14 | 1.0e-07 | yes |
| Integrator of a ramp -> a t^2/2 | interpreter | RK4 | 0.01 | 3.16e-13 | 1.0e-06 | yes |
| Integrator of a sine -> A(1-cos wt)/w | compiled | RK45 | 0.01 | 5.65e-09 | 1.0e-07 | yes |
| Integrator of a sine -> A(1-cos wt)/w | interpreter | RK4 | 0.01 | 2.81e-10 | 1.0e-06 | yes |
| StateSpace vs scipy.linalg.expm (exact ZOH) | compiled | RK45 | 0.01 | 1.97e-10 | 1.0e-07 | yes |
| StateSpace vs scipy.linalg.expm (exact ZOH) | interpreter | exact ZOH | 0.01 | 0.00e+00 | 1.0e-11 | yes |
| TranFn vs scipy.signal.lsim (sine input) | compiled | RK45 | 0.002 | 3.42e-06 | 1.0e-04 | yes |
| TranFn vs scipy.signal.lsim (sine input) | interpreter | ZOH input | 0.002 | 3.81e-03 | 2.0e-02 | yes |
| PID closed loop vs analytic CP/(1+CP) | compiled | RK45 | 0.002 | 2.64e-10 | 1.0e-07 | yes |
| DiscreteTranFn vs scipy.signal.dlsim (at samples) | interpreter | z-domain, Ts=0.1 | 0.01 | 0.00e+00 | 1.0e-12 | yes |
| Discrete output is constant between sample instants | interpreter | z-domain, Ts=0.1 | 0.01 | 0.00e+00 | 1.0e-12 | yes |
| ZeroOrderHold of a sine vs analytic staircase | interpreter | ZOH, Ts=0.1 | 0.005 | 5.97e-14 | 1.0e-09 | yes |
| Transport delay of a sine, tau on the output grid | interpreter | buffer replay | 0.005 | 2.22e-16 | 1.0e-12 | yes |
| Transport delay of a sine, tau off the grid | interpreter | buffer + interpolation | 0.005 | 1.20e-05 | 1.0e-04 | yes |
| Transport delay holds its initial value for t < tau | interpreter | buffer replay | 0.005 | 0.00e+00 | 1.0e-12 | yes |
| Saturation corner: located switching instant \|t - 0.7\| | compiled | RK45 + events | 0.01 | 0.00e+00 | 1.0e-09 | yes |
| Saturation corner: trajectory vs analytic | compiled | RK45 + events | 0.01 | 1.07e-08 | 1.0e-06 | yes |
| Saturation corner: trajectory vs analytic | interpreter | fixed step, no events | 0.01 | 3.50e-03 | 1.0e-02 | yes |
| Switch threshold: located switching instant \|t - 0.5\| | compiled | RK45 + events | 0.01 | 0.00e+00 | 1.0e-09 | yes |
| Switch threshold: trajectory vs analytic | compiled | RK45 + events | 0.01 | 2.77e-08 | 1.0e-06 | yes |
| Van der Pol, mu=1000 | compiled | Radau | 0.005 | 2.00e-15 | 1.0e-06 | yes |
| Van der Pol, mu=1000 | compiled | LSODA | 0.005 | 1.24e-09 | 1.0e-06 | yes |
| Heat 1-D eigenmode decay, N=81 | compiled | RK45, method of lines | 0.005 | 4.73e-05 | 2.0e-04 | yes |
| Heat 1-D spatial convergence order \|p - 2\| | compiled | N = 21, 41, 81 | 0.005 | 2.07e-04 | 1.0e-01 | yes |
| Advection 1-D Gaussian pulse, N=401 | compiled | RK45, 2nd-order upwind | 0.005 | 2.04e-03 | 5.0e-03 | yes |
| Advection 1-D spatial convergence order \|p - 2\| | compiled | N = 101, 201, 401 | 0.005 | 2.58e-02 | 2.0e-01 | yes |
| Advection 1-D mass conservation (relative drift) | compiled | RK45, 2nd-order upwind | 0.005 | 1.46e-05 | 1.0e-04 | yes |
| Integrator FWD_EULER: observed order \|p - 1\| | interpreter | FWD_EULER | 0.02 -> 0.005 | 4.60e-03 | 1.0e-01 | yes |
| Integrator RK4: observed order \|p - 4\| | interpreter | RK4 | 0.02 -> 0.005 | 6.83e-03 | 1.5e-01 | yes |

## Known defects

Three cases currently fail and are pinned as `xfail(strict=True)` in
`tests/validation/test_known_defects.py` — each with a full reproducer in its
docstring, so the marker comes off the moment the defect is fixed. They are
wrong answers or crashes, not tolerance quibbles:

1. **The interpreted PID's derivative branch never sees a step in its input.**
   `blocks/pid.py` seeds `_prev_e` with the first error sample on the
   initializing call, so `de = 0` at `t0` and the filtered derivative's response
   to the reference step is lost. The compiled kernel starts its filter state at
   zero and does produce it, matching the documented
   `C(s) = Kp + Ki/s + Kd N s/(s+N)`. The resulting closed-loop error does not
   vanish with `dt`: 0.2866 / 0.2843 / 0.2834 / 0.2831 at
   `dt = 8e-3 / 4e-3 / 2e-3 / 1e-3`, an observed order of 0.00. With
   `Kp = Ki = 0, Kd = 0.5` the interpreted loop output is identically zero while
   the true response peaks at 0.36.
2. **`Integrator` crashes under `TUSTIN` and `BWD_EULER` on a 0-d input.**
   `blocks/sine.py` returns `np.array(scalar)` (shape `()`); the integrator's
   promotion guard only tests `isinstance(x, (float, int))`, so its shape check
   rewrites `params['mem']` as a 0-d array while `params['mem_list'][0]` was
   allocated with shape `(1,)`. The in-place `mem += ...` of exactly those two
   branches then raises `ValueError: non-broadcastable output operand`.
   `FWD_EULER`, `RK4` and `SOLVE_IVP` are unaffected, as are `Step` / `Ramp` /
   `Constant` sources.
3. **A single monotone state crossing trips the chattering guard.**
   `lib/engine/zero_crossing.py` restarts each segment at the located root plus
   a nudge in *time*, carrying the state across unchanged — so a guard written
   on the state (`Saturation`'s `u - max`, where `u` is an integrator output) is
   still exactly zero at the new segment start and is re-detected. The run logs
   `chattering: 20 consecutive events closer than 2e-09s` and finishes on the
   fixed-step fallback. The first instant is still located exactly (the
   validated row above), but later switches in the same run fall back to step
   accuracy. The documented nudge protects only guards that are functions of `t`
   (`Step`, `Ramp`), not of `y`.

## Adding a case

Add a `case_*` function to `tests/validation/_cases.py` returning one
`CaseResult` per measured row, register it in `ALL_CASES`, and assert it from a
test module in `tests/validation/`. Both the pytest suite and
`scripts/validation_report.py` pick it up from the registry; nothing needs to be
duplicated. Diagrams are built headlessly through
`tests/validation/_harness.py`, which runs either path and returns the Scope
traces under their declared labels.
