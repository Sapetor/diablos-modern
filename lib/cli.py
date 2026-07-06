"""Headless command-line runner for DiaBloS diagrams.

Runs a ``.diablos`` diagram without the GUI and writes the Scope traces to CSV
or NPZ, so diagrams can be simulated from scripts, CI, or an AI agent without
touching the Qt UI. Reuses the same headless path the tuning/experiment tools
use (``DSim.run_tuning_simulation`` + ``harvest_scope_signals``).

Usage::

    python diablos_modern.py run diagram.diablos -o out.csv
    python diablos_modern.py run diagram.diablos --time 20 --dt 0.005 -o out.csv
    python diablos_modern.py run diagram.diablos --solver interpreter -o out.npz

Or as a module: ``python -m lib.cli run diagram.diablos -o out.csv``.

The default solver is the compiled fast path (scipy ``solve_ivp``), which is the
numerically accurate engine; pass ``--solver interpreter`` for the fixed-step
interpreter. See ``docs/ARCHITECTURE.md`` ("Compiled vs interpreter semantics")
for how the two paths differ.
"""

import argparse
import json
import os
import sys

import numpy as np

# Strong process-lifetime reference to the QApplication. Without this the Python
# wrapper is collected when run_diagram returns while the DSim's QWidgets still
# exist, and PyQt crashes tearing the C++ objects down at interpreter shutdown
# (exit code 9 on the Windows venv).
_QAPP = None


def _ensure_headless_qapp():
    """Create an offscreen QApplication if none exists.

    DSim construction builds real QWidgets/QPixmaps, which require a
    QApplication even headless. Force the offscreen platform so no window ever
    appears (mirrors tests/conftest.py). Must run before QApplication is built.
    """
    global _QAPP
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("MPLBACKEND", "Agg")
    from PyQt5.QtWidgets import QApplication

    _QAPP = QApplication.instance() or QApplication(["diablos-run"])
    return _QAPP


def _file_sim_params(filepath):
    """Read (sim_time, sim_dt) from a diagram's sim_data, with sane defaults.

    apply_loaded_data does not push the loaded sim params onto the DSim facade,
    so read them straight from the file to use as CLI defaults.
    """
    try:
        with open(filepath) as f:
            sim = json.load(f).get("sim_data", {})
        return float(sim.get("sim_time", 10.0)), float(sim.get("sim_dt", 0.01))
    except (OSError, ValueError, TypeError):
        return 10.0, 0.01


def run_diagram(filepath, sim_time=None, sim_dt=None, use_fast_solver=True):
    """Load and simulate ``filepath`` headlessly; return the finished DSim.

    ``sim_time`` / ``sim_dt`` default to the diagram's own sim_data when not
    given. Raises FileNotFoundError if the diagram is missing and RuntimeError
    if the simulation fails.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(filepath)

    _ensure_headless_qapp()

    from lib.lib import DSim
    from lib.workspace import WorkspaceManager

    # Reset the workspace singleton so a CLI run never inherits stale variables.
    WorkspaceManager._instance = None

    file_time, file_dt = _file_sim_params(filepath)
    sim_time = file_time if sim_time is None else float(sim_time)
    sim_dt = file_dt if sim_dt is None else float(sim_dt)

    dsim = DSim()
    data = dsim.file_service.load(filepath=filepath)
    if data is None:
        raise RuntimeError(f"Failed to load diagram: {filepath}")
    dsim.file_service.apply_loaded_data(data)
    dsim.use_fast_solver = use_fast_solver

    ok, err = dsim.run_tuning_simulation(sim_time, sim_dt)
    if not ok:
        raise RuntimeError(err or "Simulation failed")
    return dsim


def write_output(result, out_path, out_format=None):
    """Write ``{'timeline', 'signals'}`` to CSV or NPZ.

    Format is inferred from the extension unless ``out_format`` is given.
    Returns the number of signal columns written.
    """
    timeline = np.asarray(result["timeline"], dtype=float).reshape(-1)
    signals = result["signals"]
    labels = list(signals.keys())

    if out_format is None:
        out_format = "npz" if out_path.lower().endswith(".npz") else "csv"

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    if out_format == "npz":
        np.savez(out_path, t=timeline, **{k: np.asarray(v).reshape(-1) for k, v in signals.items()})
    else:
        n = len(timeline)
        columns = [timeline]
        for label in labels:
            col = np.asarray(signals[label], dtype=float).reshape(-1)
            # Guard against a scope buffer that is off-by-one vs the timeline.
            if len(col) != n:
                col = col[:n] if len(col) > n else np.pad(col, (0, n - len(col)))
            columns.append(col)
        data = np.column_stack(columns) if columns else timeline.reshape(-1, 1)
        header = ",".join(["t"] + [str(label) for label in labels])
        np.savetxt(out_path, data, delimiter=",", header=header, comments="")

    return len(labels)


def build_parser():
    parser = argparse.ArgumentParser(prog="diablos", description="DiaBloS headless diagram tools.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Simulate a .diablos diagram headlessly.")
    run.add_argument("diagram", help="Path to a .diablos diagram file.")
    run.add_argument(
        "-o",
        "--out",
        help="Output file (.csv or .npz). Defaults to the diagram name with a .csv extension.",
    )
    run.add_argument(
        "-t",
        "--time",
        type=float,
        default=None,
        help="Simulation duration (s). Default: diagram's sim_time.",
    )
    run.add_argument(
        "--dt", type=float, default=None, help="Time step (s). Default: diagram's sim_dt."
    )
    run.add_argument(
        "--solver",
        choices=("compiled", "interpreter"),
        default="compiled",
        help="Integration path (default: compiled fast solver).",
    )
    run.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress the per-run summary on stdout."
    )
    return parser


def main(argv=None):
    """CLI entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)

    if args.command != "run":  # argparse enforces this, but be explicit.
        return 2

    out_path = args.out or (os.path.splitext(args.diagram)[0] + ".csv")

    try:
        dsim = run_diagram(
            args.diagram,
            sim_time=args.time,
            sim_dt=args.dt,
            use_fast_solver=(args.solver == "compiled"),
        )
    except FileNotFoundError as e:
        print(f"error: diagram not found: {e}", file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001 -- surface any run failure to the shell
        print(f"error: {e}", file=sys.stderr)
        return 1

    from lib.analysis.resim import harvest_scope_signals

    result = harvest_scope_signals(dsim)
    if result is None or not result["signals"]:
        print("error: simulation produced no Scope signals to export", file=sys.stderr)
        return 1

    n_cols = write_output(result, out_path)
    if not args.quiet:
        n_rows = len(np.atleast_1d(result["timeline"]))
        print(
            f"ran {args.diagram} [{args.solver}] -> {out_path} ({n_rows} samples, {n_cols} signals)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
