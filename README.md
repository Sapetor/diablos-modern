# DiaBloS Modern

**A Python-native visual laboratory for dynamics and control.**

[![CI](https://github.com/Sapetor/diablos-modern/actions/workflows/ci.yml/badge.svg)](https://github.com/Sapetor/diablos-modern/actions/workflows/ci.yml)
[![Docs](https://github.com/Sapetor/diablos-modern/actions/workflows/docs.yml/badge.svg)](https://sapetor.github.io/diablos-modern/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Bundled binaries: GPL-3.0](https://img.shields.io/badge/bundled%20binaries-GPL--3.0-orange.svg)](THIRD_PARTY_LICENSES.md)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.12-blue.svg)](https://www.python.org/)

Draw a block diagram, press `F5`, get a plot. Then linearize it, sweep a
parameter, run a seeded Monte Carlo ensemble, or export the whole model as a
standalone numpy + scipy script.

![DiaBloS Modern](screenshot.png)

## Features

- **Visual editor** — drag-and-drop palette, drag-to-connect wiring, bezier or
  auto-routed orthogonal wires, subsystems, Goto/From tag routing, undo/redo,
  command palette, minimap.
- **Two solvers** — a compiled fast path that flattens the diagram into one
  `rhs(t, x)` for `scipy.integrate.solve_ivp` (RK45, RK23, DOP853, Radau, BDF,
  LSODA), and a fixed-step interpreter (RK4, Euler) that covers everything else.
- **Zero-crossing detection** — the fast solver stops exactly at each switching
  instant (Switch, Saturation, Deadband, Hysteresis, Step, PRBS…) instead of
  smearing it across an adaptive step.
- **Control analysis** — numeric linearization of the closed loop with
  pole-zero, Bode, gain/phase margins, controllability and observability; trim
  solving; root locus, Nyquist and LQR design.
- **Experiments** — seeded Monte Carlo ensembles with percentile bands and
  1-D/2-D parameter sweeps, both reproducible from a single master seed.
- **96 blocks** — sources, sinks, math, logic, routing, discrete and multi-rate
  blocks, a full control family, 1D/2D PDEs (heat, wave, advection,
  diffusion-reaction) with field visualization, and optimization primitives.
- **Masks and user libraries** — mask a subsystem, save it to your library, and
  it appears in the palette under **USER LIBRARY**.
- **Export** — TikZ for papers, PNG/SVG, or a self-contained Python script that
  reproduces the compiled solver.
- **Bilingual** — English and Spanish, switchable live from View ▸ Language;
  adding a language is one JSON file.
- **Tested** — 2,700+ tests on a Python 3.10 + 3.12 CI matrix, with `ruff` lint
  and format gates.

## Install

**Prebuilt apps** are on the
[Releases page](https://github.com/Sapetor/diablos-modern/releases): a macOS
arm64 `.dmg`, a Windows x64 `.zip` and a Linux x86_64 `.tar.gz`. No Python
needed. The builds are unsigned — macOS needs
`xattr -rd com.apple.quarantine /Applications/DiaBloS-arm64.app` once, and
Windows SmartScreen needs *More info ▸ Run anyway*.

**From source** (Python 3.10+):

```bash
git clone https://github.com/Sapetor/diablos-modern.git
cd diablos-modern
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python diablos_modern.py
```

Full instructions, including conda and Linux system libraries:
[Installation](https://sapetor.github.io/diablos-modern/getting-started/installation/).

## 60-second quick start

1. Drag **Step** (Sources), **TranFn** (Control) and **Scope** (Sinks) onto the
   canvas.
2. Press on an output port, drag to an input port, release. Repeat.
3. Click the TranFn and set `numerator` `[1]`, `denominator` `[1, 1]`.
4. Press `F5`, set **Simulation Duration** to 10, press **Simulate**.
5. The scope opens with the step response.

`Ctrl+S` saves a `.diablos` file. `F1` lists every keyboard shortcut.
**File ▸ Examples** opens the bundled diagrams.

## Command line

Two subcommands never build a GUI, so they work over SSH and in CI:

```bash
# Simulate headlessly, write every Scope trace to CSV (or .npz)
python diablos_modern.py run model.diablos -o out.csv --time 30 --dt 0.005

# Export a standalone numpy + scipy script that reproduces the compiled solver
python diablos_modern.py export-python model.diablos -o model.py
python model.py --out run.csv --no-plot
```

`run` also takes `--solver {compiled,interpreter}`, `--no-zero-crossing` and `--verify` (print the verification report, exit 3 on a failed check).

## Documentation

<https://sapetor.github.io/diablos-modern/>

| | |
|---|---|
| [Installation](https://sapetor.github.io/diablos-modern/getting-started/installation/) | Binaries, source, headless, language |
| [Quick start](https://sapetor.github.io/diablos-modern/getting-started/quickstart/) | First diagram, shortcuts |
| [Running simulations](https://sapetor.github.io/diablos-modern/user-guide/running-simulations/) | Solvers, tolerances, zero crossings, diagnostics |
| [Analysis & experiments](https://sapetor.github.io/diablos-modern/user-guide/analysis/) | Linearization, Bode, LQR, Monte Carlo, sweeps |
| [Block reference](https://sapetor.github.io/diablos-modern/wiki/Home/) | Every block, by category |
| [Architecture](docs/ARCHITECTURE.md) / [Developer Guide](docs/DEVELOPER_GUIDE.md) | How it is built, how to add blocks |
| [Releasing](docs/RELEASING.md) | Version bump, tag, artifacts |

## Comparison with similar tools

| | DiaBloS Modern | PathSim | bdsim | pysimCoder | Xcos |
|---|---|---|---|---|---|
| Primary interface | Desktop GUI | Python API (+ PathView) | Python API (+ bdedit) | GUI editor | GUI editor |
| Host language | Python 3.10+ | Python | Python | Python | Scilab |
| Main target | Teaching, exploration | Programmatic simulation | Control/robotics in code | Real-time C codegen | General modelling |
| Prebuilt binaries | DMG / zip / tarball | pip | pip | source | Scilab installer |
| Non-English UI | English + Spanish | — | — | — | several |

Different, not better. If you want simulation as a library call, PathSim and
bdsim have far nicer APIs. If your endpoint is C on real hardware, pysimCoder
does what DiaBloS does not do at all. Xcos is older and broader than any of the
Python ones. DiaBloS optimizes for the desktop teaching loop: draw a loop, run
it, linearize it, sweep it — without leaving the window or writing code.

Known limits: it is an application rather than an importable library; the
compiled fast path does not cover every block (the rest fall back to a slower
fixed-step interpreter); there is no embedded code generation; and the release
binaries are unsigned.

## Development

```bash
pip install -r requirements-dev.txt
QT_QPA_PLATFORM=offscreen pytest      # headless test suite
ruff check .                          # lint gate
ruff format --check .                 # format gate
python tools/sync_block_registry.py --check
```

Contributions are welcome — see
[docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md) for the block contract,
the localization rules and the project conventions, and
[docs/RELEASING.md](docs/RELEASING.md) for how a version is cut. Please run the
lint, format and test gates before opening a pull request.

## Citation

DiaBloS Modern grew out of DiaBloS. Until it has a publication of its own,
please cite the original paper: M. Torres-Torriti and M. Rojas-Sepúlveda,
"An algorithm for processing block diagram models of dynamical systems and an
open-source visual-programming simulation tool", *Simulation Modelling Practice
and Theory* 138 (2025) 103030,
[doi:10.1016/j.simpat.2024.103030](https://doi.org/10.1016/j.simpat.2024.103030).

## License

DiaBloS Modern's own source code is **MIT** — see [LICENSE](LICENSE). Copy it,
vendor a block, build a course around it, ship a closed-source tool on top of
it. That is not changing.

The prebuilt bundles on the [releases
page](https://github.com/Sapetor/diablos-modern/releases) are a different
artifact from the source. They embed the Qt 6 libraries (LGPL v3) and the
PyQt6 bindings, and PyQt6 is **GPL-3.0-only** — so a bundle may only be
*redistributed* under GPL v3. That falls on whoever redistributes one, not on
anyone who merely uses one, and not on the MIT source: installing from source
with `pip` makes the combination on your own machine, which is not
distribution.

The notices, the offer of corresponding source and the full third-party
licence texts are in
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) and `licenses/`, both
shipped inside every bundle.
