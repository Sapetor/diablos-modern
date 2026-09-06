# Fix Compiled Solver D≠0 Feedthrough Bug

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix incorrect simulation output in compiled solver when closed-loop diagrams contain non-strictly-proper transfer functions (D≠0) or PID blocks.

**Architecture:** Split the three-way block sort into a four-way sort that distinguishes D=0 state blocks (safe to pre-populate with C*x) from D≠0 state blocks (must execute in topological order alongside algebraic blocks, since their output depends on input via D*u). Also fix PID pre-population which stores raw state instead of computed output.

**Tech Stack:** Python, NumPy, SciPy (signal.tf2ss, solve_ivp), pytest

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `lib/engine/system_compiler.py` | Modify | Four-way sort + fix pre-population |
| `tests/regression/test_feedthrough_bug.py` | Create | Regression tests for D≠0 and PID bugs |

---

### Task 1: Write Regression Tests That Expose the Bug

**Files:**
- Create: `tests/regression/test_feedthrough_bug.py`

These tests MUST fail before the fix and pass after.

- [ ] **Step 1: Write test for D≠0 TF in closed loop (PI controller)**

This is the exact bug from `examples/test_c02`: a PI controller `(2s+1)/s` (D=2) in a closed loop with plant TFs.

```python
"""
Regression tests for compiled solver feedthrough (D≠0) bug.

Bug: Pre-population of state-block outputs computes C*x only, omitting D*u.
For non-strictly-proper TFs (D≠0), this drops the feedthrough term entirely.
When another state block reads the pre-populated value before exec_ss corrects
it, the ODE derivatives are wrong.

Fix: Four-way sort separates D=0 (pre-populatable) from D≠0 (must execute
in topological order alongside algebraic blocks).
"""

import pytest
import numpy as np
from PyQt5.QtCore import QRect, QPoint
from PyQt5.QtGui import QColor


def _make_block(block_fn, sid, username, in_ports, out_ports, params, b_type=2):
    from lib.simulation.block import DBlock
    return DBlock(
        block_fn=block_fn,
        sid=sid,
        coords=QRect(0, 0, 50, 40),
        color=QColor(150, 150, 150),
        in_ports=in_ports,
        out_ports=out_ports,
        params=params,
        username=username,
        b_type=b_type,
    )


def _make_line(sid, src, srcport, dst, dstport):
    from lib.simulation.connection import DLine
    return DLine(
        sid=sid,
        srcblock=src,
        srcport=srcport,
        dstblock=dst,
        dstport=dstport,
        points=[QPoint(0, 0), QPoint(100, 0)],
    )


def _run_compiled(blocks, lines, t_end=20.0):
    """Compile and solve, return (sol, state_map, block_matrices)."""
    from lib.engine.system_compiler import SystemCompiler
    from scipy.integrate import solve_ivp

    compiler = SystemCompiler()
    model_func, y0, state_map, block_matrices = compiler.compile_system(
        blocks, blocks, lines
    )
    sol = solve_ivp(
        model_func, (0, t_end), y0, method='RK45',
        t_eval=np.linspace(0, t_end, 2000), rtol=1e-8, atol=1e-10,
    )
    assert sol.success, f"solve_ivp failed: {sol.message}"
    return sol, state_map, block_matrices


@pytest.mark.regression
class TestFeedthroughBug:
    """Test that D≠0 state blocks produce correct output in closed loops."""

    def test_pi_controller_closed_loop(self, qapp):
        """
        Closed loop with PI controller as TF:

            Step(1) -> Sum(+-) -> TranFn[(2s+1)/s] -> TranFn[1/(s+1)] -> Scope
                         ^                                      |
                         +------------- feedback ---------------+

        Forward path: C(s)*G(s) = (2s+1)/s * 1/(s+1) = (2s+1)/(s^2+s)
        Closed-loop TF: (2s+1)/(s^2+3s+1)
        Poles at s = (-3 ± sqrt(5))/2 ≈ -0.382, -2.618 (both stable)
        Steady-state to unit step: lim s->0 of s * CL(s)/s = (2*0+1)/(0+0+1) = 1.0

        The PI controller TF (2s+1)/s has D=2 (non-strictly-proper).
        Bug: pre-population computes C*x only, missing the 2*error term.
        """
        blocks = [
            _make_block('Step', 0, '', 0, 1,
                        {'value': 1.0, 'delay': 0.0, 'type': 'up'}, b_type=0),
            _make_block('Sum', 0, '', 2, 1, {'sign': '+-'}),
            _make_block('TranFn', 0, '', 1, 1,
                        {'numerator': [2.0, 1.0], 'denominator': [1.0, 0.0]}),
            _make_block('TranFn', 1, '', 1, 1,
                        {'numerator': [1.0], 'denominator': [1.0, 1.0]}),
        ]

        lines = [
            _make_line(0, 'step0', 0, 'sum0', 0),
            _make_line(1, 'sum0', 0, 'tranfn0', 0),
            _make_line(2, 'tranfn0', 0, 'tranfn1', 0),
            _make_line(3, 'tranfn1', 0, 'sum0', 1),  # feedback
        ]

        sol, state_map, _ = _run_compiled(blocks, lines, t_end=20.0)

        # Extract plant output (tranfn1 is strictly proper: output = C*x = state)
        start1, size1 = state_map['tranfn1']
        final_output = sol.y[start1, -1]

        # Steady-state should be 1.0 (PI controller drives error to zero)
        assert abs(final_output - 1.0) < 0.05, \
            f"PI closed-loop SS should be ~1.0, got {final_output:.4f}"

    def test_lead_compensator_closed_loop(self, qapp):
        """
        Lead compensator (2s+1)/(s+10) has D=2 (degree num = degree den).

            Step(1) -> Sum(+-) -> TranFn[(2s+1)/(s+10)] -> TranFn[1/(s+1)] -> Scope
                         ^                                          |
                         +----------------- feedback ---------------+

        Open-loop: L(s) = (2s+1)/((s+10)(s+1))
        CL TF = L/(1+L) = (2s+1)/(s^2+11s+10+2s+1) = (2s+1)/(s^2+13s+11)
        SS = 1/11 ≈ 0.0909

        D matrix for (2s+1)/(s+10): tf2ss gives D=2.
        """
        blocks = [
            _make_block('Step', 0, '', 0, 1,
                        {'value': 1.0, 'delay': 0.0, 'type': 'up'}, b_type=0),
            _make_block('Sum', 0, '', 2, 1, {'sign': '+-'}),
            _make_block('TranFn', 0, '', 1, 1,
                        {'numerator': [2.0, 1.0], 'denominator': [1.0, 10.0]}),
            _make_block('TranFn', 1, '', 1, 1,
                        {'numerator': [1.0], 'denominator': [1.0, 1.0]}),
        ]

        lines = [
            _make_line(0, 'step0', 0, 'sum0', 0),
            _make_line(1, 'sum0', 0, 'tranfn0', 0),
            _make_line(2, 'tranfn0', 0, 'tranfn1', 0),
            _make_line(3, 'tranfn1', 0, 'sum0', 1),
        ]

        sol, state_map, _ = _run_compiled(blocks, lines, t_end=10.0)

        start1, size1 = state_map['tranfn1']
        final_output = sol.y[start1, -1]

        expected_ss = 1.0 / 11.0
        assert abs(final_output - expected_ss) < 0.01, \
            f"Lead compensator CL SS should be ~{expected_ss:.4f}, got {final_output:.4f}"

    def test_pid_block_closed_loop(self, qapp):
        """
        PID block in closed loop:

            Step(1) -> PID(Kp=2, Ki=1, Kd=0) -> TranFn[1/(s+1)] -> Scope
              (sp)          ^                          |
                            +------ feedback (meas) ---+

        Equivalent to PI: C(s) = 2 + 1/s = (2s+1)/s, same as test above.
        PID pre-population stores raw state [x_i, x_d] instead of Kp*e + Ki*x_i.

        Steady-state should be 1.0.
        """
        blocks = [
            _make_block('Step', 0, '', 0, 1,
                        {'value': 1.0, 'delay': 0.0, 'type': 'up'}, b_type=0),
            _make_block('PID', 0, '', 2, 1,
                        {'Kp': 2.0, 'Ki': 1.0, 'Kd': 0.0, 'N': 20.0,
                         'u_min': -1e6, 'u_max': 1e6}),
            _make_block('TranFn', 0, '', 1, 1,
                        {'numerator': [1.0], 'denominator': [1.0, 1.0]}),
        ]

        # PID has 2 inputs: port 0 = setpoint, port 1 = measurement
        lines = [
            _make_line(0, 'step0', 0, 'pid0', 0),       # setpoint
            _make_line(1, 'tranfn0', 0, 'pid0', 1),     # measurement (feedback)
            _make_line(2, 'pid0', 0, 'tranfn0', 0),     # control signal
        ]

        sol, state_map, _ = _run_compiled(blocks, lines, t_end=20.0)

        start_tf, size_tf = state_map['tranfn0']
        final_output = sol.y[start_tf, -1]

        assert abs(final_output - 1.0) < 0.05, \
            f"PID closed-loop SS should be ~1.0, got {final_output:.4f}"

    def test_strictly_proper_tf_still_works(self, qapp):
        """
        Sanity check: strictly proper TFs (D=0) should still work after fix.

            Step(8) -> Sum(+-) -> TranFn[1/(s+0.2)] -> feedback
            CL: 1/(s+1.2), SS = 8/1.2 ≈ 6.667
        """
        blocks = [
            _make_block('Step', 0, '', 0, 1,
                        {'value': 8.0, 'delay': 0.0, 'type': 'up'}, b_type=0),
            _make_block('Sum', 0, '', 2, 1, {'sign': '+-'}),
            _make_block('TranFn', 0, '', 1, 1,
                        {'numerator': [1.0], 'denominator': [1.0, 0.2]}),
        ]

        lines = [
            _make_line(0, 'step0', 0, 'sum0', 0),
            _make_line(1, 'sum0', 0, 'tranfn0', 0),
            _make_line(2, 'tranfn0', 0, 'sum0', 1),
        ]

        sol, state_map, _ = _run_compiled(blocks, lines, t_end=50.0)

        start, _ = state_map['tranfn0']
        final_output = sol.y[start, -1]

        expected_ss = 8.0 / 1.2
        assert abs(final_output - expected_ss) < 0.05, \
            f"D=0 closed-loop SS should be ~{expected_ss:.3f}, got {final_output:.3f}"

    def test_compiled_vs_analytical_three_tf_chain(self, qapp):
        """
        Three TFs in closed loop (from test_c02 diagram):

            Step -> Sum(+-) -> (2s+1)/s -> 1/(s+1) -> 1/s -> feedback + Scope

        Open-loop: L(s) = (2s+1)/(s^2*(s+1))
        CL: (2s+1)/(s^3+s^2+2s+1)

        With type-2 system + PI, steady-state to unit step = 1.0.
        """
        blocks = [
            _make_block('Step', 0, '', 0, 1,
                        {'value': 1.0, 'delay': 0.0, 'type': 'up'}, b_type=0),
            _make_block('Sum', 0, '', 2, 1, {'sign': '+-'}),
            _make_block('TranFn', 0, '', 1, 1,
                        {'numerator': [2.0, 1.0], 'denominator': [1.0, 0.0]}),
            _make_block('TranFn', 1, '', 1, 1,
                        {'numerator': [1.0], 'denominator': [1.0, 1.0]}),
            _make_block('TranFn', 2, '', 1, 1,
                        {'numerator': [1.0], 'denominator': [1.0, 0.0]}),
        ]

        lines = [
            _make_line(0, 'step0', 0, 'sum0', 0),
            _make_line(1, 'sum0', 0, 'tranfn0', 0),
            _make_line(2, 'tranfn0', 0, 'tranfn1', 0),
            _make_line(3, 'tranfn1', 0, 'tranfn2', 0),
            _make_line(4, 'tranfn2', 0, 'sum0', 1),  # feedback
        ]

        sol, state_map, _ = _run_compiled(blocks, lines, t_end=30.0)

        # Get output of last TF (tranfn2, integrator: D=0, output = state)
        start2, _ = state_map['tranfn2']
        final_output = sol.y[start2, -1]

        assert abs(final_output - 1.0) < 0.05, \
            f"3-TF closed-loop SS should be ~1.0, got {final_output:.4f}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/apeters/Documents/APR/02-Projects/diablos-modern && python -m pytest tests/regression/test_feedthrough_bug.py -v`

Expected: `test_pi_controller_closed_loop`, `test_lead_compensator_closed_loop`, `test_pid_block_closed_loop`, and `test_compiled_vs_analytical_three_tf_chain` FAIL. `test_strictly_proper_tf_still_works` PASSES (D=0 case works today).

- [ ] **Step 3: Commit failing tests**

```bash
git add tests/regression/test_feedthrough_bug.py
git commit -m "test: add regression tests for D≠0 feedthrough bug in compiled solver

Tests currently FAIL, exposing the pre-population bug where C*x
is computed without D*u for non-strictly-proper TFs and PID blocks."
```

---

### Task 2: Fix the Four-Way Sort and Pre-Population

**Files:**
- Modify: `lib/engine/system_compiler.py:1614-1641` (sort logic)
- Modify: `lib/engine/system_compiler.py:1889-1917` (pre-population)

- [ ] **Step 1: Detect D≠0 blocks at compile time and split state blocks**

At `system_compiler.py:1614-1641`, replace the three-way sort with a four-way sort. D≠0 state blocks go into the algebraic group (they have direct feedthrough and need their input computed first). They still get state allocation and derivatives computed — they just execute earlier.

Replace this block (lines 1618-1641):
```python
        # 0. Re-order blocks: Sources first, then algebraic blocks, then state blocks last.
        # State blocks (TranFn, StateSpace, Integrator, PID, etc.) must execute AFTER
        # their algebraic input chain so derivatives read correct signal values.
        # Their outputs are pre-populated (y=Cx or y=state) before the sequence runs,
        # so feedback through algebraic blocks resolves correctly.
        source_fns = ('Step', 'Sine', 'Constant', 'From', 'Ramp', 'Exponential', 'Noise', 'Wavegenerator', 'Prbs')
        state_fns = ('Tranfn', 'Transferfcn', 'TransferFcn', 'Statespace', 'StateSpace',
                     'Integrator', 'Pid', 'PID', 'Ratelimiter', 'RateLimiter',
                     'Heatequation1D', 'Waveequation1D', 'Advectionequation1D',
                     'Diffusionreaction1D', 'Heatequation2D', 'Waveequation2D',
                     'Advectionequation2D')
        sources = []
        algebraic = []
        state_blocks = []
        for b in sorted_order:
            fn = b.block_fn.title() if b.block_fn else ''
            if fn in source_fns:
                sources.append(b)
            elif fn in state_fns:
                state_blocks.append(b)
            else:
                algebraic.append(b)

        sorted_order = sources + algebraic + state_blocks
```

With:
```python
        # 0. Re-order blocks into four groups:
        #   1. Sources (no inputs)
        #   2. Algebraic + D≠0 state blocks (feedthrough blocks need input first)
        #   3. D=0 state blocks (pre-populated with C*x, execute last)
        #
        # D≠0 state blocks (non-strictly-proper TFs, PIDs) have direct feedthrough:
        # their output = C*x + D*u depends on the current input u. They CANNOT be
        # pre-populated with just C*x because downstream state blocks would read
        # the wrong value. Instead, they execute alongside algebraic blocks in
        # topological order so their input is available when they run.
        #
        # D=0 state blocks (strictly proper TFs, integrators, PDE blocks) can be
        # safely pre-populated with C*x (since D*u = 0), breaking feedback loops.
        source_fns = ('Step', 'Sine', 'Constant', 'From', 'Ramp', 'Exponential', 'Noise', 'Wavegenerator', 'Prbs')
        state_fns = ('Tranfn', 'Transferfcn', 'TransferFcn', 'Statespace', 'StateSpace',
                     'Integrator', 'Pid', 'PID', 'Ratelimiter', 'RateLimiter',
                     'Heatequation1D', 'Waveequation1D', 'Advectionequation1D',
                     'Diffusionreaction1D', 'Heatequation2D', 'Waveequation2D',
                     'Advectionequation2D')
        sources = []
        algebraic = []
        state_blocks_d0 = []     # D=0: safe to pre-populate
        state_blocks_dnz = []    # D≠0: must execute with algebraic blocks
        for b in sorted_order:
            fn = b.block_fn.title() if b.block_fn else ''
            if fn in source_fns:
                sources.append(b)
            elif fn in state_fns:
                # Check if this state block has direct feedthrough (D≠0)
                has_feedthrough = False
                if b.name in block_matrices:
                    _, _, _, D = block_matrices[b.name]
                    if np.any(D != 0):
                        has_feedthrough = True
                elif fn in ('Pid', 'PID'):
                    # PID always has feedthrough (output depends on current error)
                    has_feedthrough = True
                if has_feedthrough:
                    state_blocks_dnz.append(b)
                else:
                    state_blocks_d0.append(b)
            else:
                algebraic.append(b)

        sorted_order = sources + algebraic + state_blocks_dnz + state_blocks_d0
```

Note: `block_matrices` is populated in section 2 (state identification) which runs before this sort at section 0. But wait — the sort is at the TOP of compile_system (section 0) and `block_matrices` is populated in section 2. We need to move the sort AFTER section 2.

- [ ] **Step 2: Move the sort after state identification**

The sort at section 0 (lines 1618-1641) must move to AFTER section 2 (after line 1881 `y0 = np.array(...)`) so that `block_matrices` is populated when we check D.

Move the entire sort block to just before section 3 (line 1883 `# 3. Compile Execution Sequence`). The section comment numbers will shift but the logic remains the same.

Concretely, the new order in `compile_system` becomes:
1. Build dependency graph (input_map)
2. Identify states (state_map, block_matrices, y0)
3. **Sort blocks** (using block_matrices to detect D≠0)
4. Compile execution sequence
5. Build pre-population list
6. Create model_func closure

- [ ] **Step 3: Fix pre-population to skip D≠0 blocks**

Replace the pre-population section (lines 1889-1901):
```python
        # 4. Build pre-population list for state-based block outputs.
        # In feedback loops, algebraic blocks (Sum, Gain) may depend on
        # state-based block outputs (TranFn, StateSpace, Integrator) that
        # execute later in the topological order. Pre-populating y = C*x
        # (or y = x for Integrators) breaks these implicit algebraic loops.
        state_output_preloads = []
        for b_name, (start, size) in state_map.items():
            if b_name in block_matrices:
                A, B, C, D = block_matrices[b_name]
                state_output_preloads.append((b_name, start, size, C))
            else:
                # Integrator: output = state
                state_output_preloads.append((b_name, start, size, None))
```

With:
```python
        # 4. Build pre-population list for D=0 state-block outputs ONLY.
        # D≠0 blocks (non-strictly-proper TFs, PIDs) are NOT pre-populated
        # because their output = C*x + D*u depends on the input u, which
        # isn't available yet. They execute in the algebraic group instead.
        state_output_preloads = []
        d0_block_names = {b.name for b in state_blocks_d0}
        for b_name, (start, size) in state_map.items():
            if b_name not in d0_block_names:
                continue  # Skip D≠0 blocks — they execute with algebraic group
            if b_name in block_matrices:
                A, B, C, D = block_matrices[b_name]
                state_output_preloads.append((b_name, start, size, C))
            else:
                # Integrator: output = state
                state_output_preloads.append((b_name, start, size, None))
```

- [ ] **Step 4: Run all regression tests**

Run: `cd /Users/apeters/Documents/APR/02-Projects/diablos-modern && python -m pytest tests/regression/ -v`

Expected: ALL tests pass, including the new feedthrough tests and the existing `test_feedback_loop.py` tests.

- [ ] **Step 5: Run full test suite**

Run: `cd /Users/apeters/Documents/APR/02-Projects/diablos-modern && python -m pytest tests/ -v --timeout=60`

Expected: No regressions. All 675+ tests pass.

- [ ] **Step 6: Commit the fix**

```bash
git add lib/engine/system_compiler.py
git commit -m "fix: compiled solver D≠0 feedthrough bug in closed-loop simulation

Non-strictly-proper TFs (D≠0) and PID blocks were pre-populated with
C*x only, omitting the D*u feedthrough term. When downstream state
blocks read this value before exec_ss corrected it, ODE derivatives
were wrong — the entire proportional term of PI controllers was lost.

Fix: four-way block sort separates D=0 state blocks (safe to
pre-populate) from D≠0 state blocks (execute alongside algebraic
blocks in topological order). Pre-population now only applies to
D=0 blocks."
```

---

### Task 3: Manual Verification with test_c02 Diagram

- [ ] **Step 1: Run the app and simulate test_c02**

Run: `cd /Users/apeters/Documents/APR/02-Projects/diablos-modern && python diablos_modern.py`

Open `examples/test_c02` (the closed-loop PI + plant diagram). Run simulation. The scope output should show the system settling to 1.0 with proper transient dynamics (not the sluggish/wrong response seen before the fix).

- [ ] **Step 2: Compare with a known-good reference**

In Python, compute the analytical closed-loop step response and verify the scope output matches:

```python
from scipy import signal
import numpy as np

# Forward path: (2s+1)/s * 1/(s+1) * 1/s
num_ol = [2, 1]
den_ol = [1, 1, 0, 0]  # s^2*(s+1)

# Closed-loop: num_ol / (den_ol + num_ol)
num_cl = num_ol
den_cl = [1, 1, 2, 1]  # s^3 + s^2 + 2s + 1

sys_cl = signal.TransferFunction(num_cl, den_cl)
t = np.linspace(0, 30, 3000)
_, y = signal.step(sys_cl, T=t)
print(f"Steady-state: {y[-1]:.4f}")  # Should be 1.0
```

---

### Task 4: Update Lessons File

- [ ] **Step 1: Add lesson to tasks/lessons.md**

Append:
```markdown
## Compiled solver pre-population only works for D=0 blocks (2026-03-23)

**Bug:** Pre-populating state-block outputs with `C*x` in `model_func` is only valid when `D=0` (strictly proper). For D≠0 blocks (PI controllers as TFs, lead compensators, PID blocks), the output is `C*x + D*u` and depends on the current input. If another state block reads the pre-populated value before exec_ss runs, it gets the wrong signal.

**Fix:** Four-way sort: sources → algebraic + D≠0 state blocks → D=0 state blocks. Only D=0 blocks get pre-populated.

**Rule:** When adding any new block type with state to the compiled solver, always check whether its output depends on the current input (feedthrough). If yes, it must NOT be pre-populated and must execute with the algebraic group.
```
