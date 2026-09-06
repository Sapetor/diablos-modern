# Analysis Blocks

List of available blocks in the **Analysis** category.

These four are *markers*, not signal-processing blocks: they carry no state,
produce no output, and do nothing during a simulation. You drop one on the
canvas, wire its input to the dynamic block you want to study, then **right-click
it** and pick the generate action. A plot window opens immediately -- no run
required.

| Block | Description |
|-------|-------------|
| [BodeMag](#bodemag) | Bode magnitude plot of a connected dynamic block. |
| [BodePhase](#bodephase) | Bode phase plot of a connected dynamic block. |
| [Nyquist](#nyquist) | Nyquist (polar) plot of a connected dynamic block. |
| [RootLocus](#rootlocus) | Root-locus plot of a connected dynamic block. |

!!! tip "Marker blocks vs. whole-diagram analysis"
    These blocks analyse **one** `TranFn` or `StateSpace` block. To get the
    frequency response, poles and margins of the *whole closed-loop diagram* --
    nonlinearities included, linearized about an operating point -- use
    **Analysis > Linearize & Analyze...** instead. See
    [Analysis & Experiments](../user-guide/analysis.md).

---

### BodeMag

Right-click > **Generate Bode magnitude plot**.

Displays the frequency-response magnitude (gain in dB) against frequency
(rad/s) on a log scale. The window is titled `Bode Magnitude Plot: <block name>`.

Connect the input to a `TranFn` or `StateSpace` block.

**Ports**: 1 In, 0 Out. No parameters.

---

### BodePhase

Right-click > **Generate Bode phase plot**.

Displays the frequency-response phase (degrees) against frequency (rad/s) on a
log scale. The window is titled `Bode Phase Plot: <block name>`.

**Ports**: 1 In, 0 Out. No parameters.

---

### Nyquist

Right-click > **Generate Nyquist plot**.

Displays the frequency response as a polar plot (real vs. imaginary part). Use
it for stability analysis: count the encirclements of the `-1` point. The window
is titled `Nyquist Plot: <block name>`.

**Ports**: 1 In, 0 Out. No parameters.

---

### RootLocus

Right-click > **Generate root-locus plot**.

Traces the closed-loop poles as a parameter (typically the loop gain `K`)
varies, showing the pole trajectories and the stability boundary. The window is
titled `Root Locus: <block name>`.

**Ports**: 1 In, 0 Out. No parameters.

---
