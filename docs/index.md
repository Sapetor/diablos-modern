---
hide:
  - navigation
---

# DiaBloS Modern

**A Python-native visual laboratory for dynamics and control.**

Draw a block diagram, press <kbd>F5</kbd>, and get a plot. Then linearize it, sweep
a parameter, run a thousand seeded Monte Carlo trials, or export the whole model
as a standalone numpy + scipy script you can put in a paper or a CI job.

DiaBloS Modern is a desktop application built with PyQt6. It ships as a prebuilt
app for macOS, Windows and Linux, runs from a checkout with
`python diablos_modern.py`, and speaks English and Spanish.

![The DiaBloS Modern canvas](images/screenshot.png)

[Install it](getting-started/installation.md){ .md-button .md-button--primary }
[60-second quick start](getting-started/quickstart.md){ .md-button }
[Block reference](wiki/Home.md){ .md-button }

---

## What it does

<div class="grid cards" markdown>

-   __Draw and simulate__

    ---

    Drag blocks from the palette, drag from an output port to an input port to
    wire them, press <kbd>F5</kbd>. Subsystems, Goto/From tag routing,
    auto-routed orthogonal wires, alignment guides, undo/redo, a command palette
    and a minimap.

    [:octicons-arrow-right-24: Creating diagrams](user-guide/creating-diagrams.md)

-   __Two solvers, one contract__

    ---

    A compiled fast path flattens the diagram into a single `rhs(t, x)` for
    `scipy.integrate.solve_ivp` (RK45, RK23, DOP853, Radau, BDF, LSODA) with
    zero-crossing detection at switching instants. An interpreted fixed-step
    path (RK4, Euler) covers everything else.

    [:octicons-arrow-right-24: Running simulations](user-guide/running-simulations.md)

-   __Control analysis built in__

    ---

    Numerically linearize the whole closed loop about a trim point, then read
    poles and zeros, a Bode plot, gain and phase margins, controllability and
    observability, and step / impulse responses. Root locus, Nyquist and LQR
    design too.

    [:octicons-arrow-right-24: Analysis & experiments](user-guide/analysis.md)

-   __Experiments, not just runs__

    ---

    Seeded Monte Carlo ensembles with percentile bands, and 1-D / 2-D parameter
    sweeps with response families and outcome heatmaps. Every stochastic block
    derives its sub-seed from one master seed, so an ensemble is reproducible
    from a single number.

    [:octicons-arrow-right-24: Monte Carlo and sweeps](user-guide/analysis.md#monte-carlo-ensembles)

-   __96 blocks, PDEs included__

    ---

    Sources, sinks, math, logic, routing, discrete and multi-rate blocks, a full
    control family, 1D/2D heat, wave, advection and diffusion-reaction PDEs with
    field visualization, and optimization primitives you can wire into an
    algorithm.

    [:octicons-arrow-right-24: Block reference](wiki/Home.md)

-   __Get your work out__

    ---

    Export the diagram as TikZ for a paper, as PNG/SVG, or as a self-contained
    Python script that reproduces the compiled solver. Or skip the GUI entirely:
    `diablos_modern.py run model.diablos -o out.csv`.

    [:octicons-arrow-right-24: Export and CLI](user-guide/running-simulations.md#export-as-python-script)

-   __Your own blocks__

    ---

    Mask a subsystem to give it a parameter dialog and an icon, save it to your
    library, and it shows up in the palette under **USER LIBRARY**. Or write a
    Python class in `blocks/` -- four properties and an `execute()` method.

    [:octicons-arrow-right-24: Developer guide](DEVELOPER_GUIDE.md)

-   __Bilingual__

    ---

    English and Spanish, switchable live from **View ▸ Language**. Catalogs are
    plain JSON keyed by the English source string, so adding a language is one
    file and no code.

    [:octicons-arrow-right-24: Language & translations](user-guide/localization.md)

</div>

---

## Where DiaBloS fits

Python already has good block-diagram simulators. They mostly optimize for
something different, and it is worth being clear about what.

| | DiaBloS Modern | [PathSim](https://github.com/milanofthe/pathsim) | [bdsim](https://github.com/petercorke/bdsim) | [pysimCoder](https://github.com/robertobucher/pysimCoder) | [Xcos](https://www.scilab.org/software/xcos) |
|---|---|---|---|---|---|
| **Primary interface** | Desktop GUI | Python API (+ PathView editor) | Python API (+ bdedit editor) | GUI editor | GUI editor |
| **Runs on** | Python 3.9+ | Python | Python | Python | Scilab |
| **Main target** | Teaching and exploration | Programmatic simulation | Robotics/control teaching in code | Real-time C code generation | General modelling |
| **Ships binaries** | Yes: DMG / zip / tarball | pip | pip | source | Scilab installer |
| **Non-English UI** | English + Spanish | — | — | — | several |

Read that table as *different*, not *better*. If you want simulation as a
library call inside a larger Python program, PathSim and bdsim are the more
natural fit and their APIs are far nicer than driving a GUI. If your endpoint is
C running on real hardware, pysimCoder does the thing DiaBloS does not do at
all. Xcos is older, broader and more battle-tested than any of the Python ones.

What DiaBloS optimizes for is the **desktop teaching and exploration loop**: a
student or researcher opens an app, draws a loop, runs it, and then — without
leaving the window or writing any code — linearizes it, reads the margins,
sweeps a gain, and runs a stochastic ensemble. The control-analysis toolbox is
first-class rather than an add-on, the app installs by dragging an icon, and
the interface is available in Spanish. That is the niche.

Honest limitations, in the same spirit:

- It is a **desktop application**, not a library. There is a headless CLI and a
  Python-script exporter, but no importable simulation API.
- The **compiled fast path does not cover every block**. Blocks outside it fall
  back to a fixed-step interpreter, which is slower and less accurate. See
  [Fast Solver](FAST_SOLVER.md) for what is covered.
- **No code generation** for embedded targets.
- The prebuilt apps are **unsigned**, so macOS and Windows both need a one-time
  override to open them.

---

## Get started

- [Installation](getting-started/installation.md) — binaries, source, headless
- [Quick start](getting-started/quickstart.md) — first diagram in a minute
- [User Manual](USER_MANUAL.md) — the long-form guide
- [Block reference](wiki/Home.md) — every block, by category
- [Architecture](ARCHITECTURE.md) and [Developer Guide](DEVELOPER_GUIDE.md) — how it is built
- [Releasing](RELEASING.md) — how a version gets cut

DiaBloS Modern's source is MIT licensed. The prebuilt bundles additionally
embed Qt 6 (LGPL v3) and the PyQt6 bindings (GPL-3.0-only), so a bundle may
only be redistributed under GPL v3 — see
[THIRD_PARTY_LICENSES.md](https://github.com/Sapetor/diablos-modern/blob/main/THIRD_PARTY_LICENSES.md)
for the notices and the corresponding source. Source, issues and releases are
on [GitHub](https://github.com/Sapetor/diablos-modern).
