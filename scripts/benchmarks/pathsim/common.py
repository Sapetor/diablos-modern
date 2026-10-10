"""Shared helpers for the DiaBloS vs PathSim head-to-head benchmark.

Timing scope: only the simulate call is timed (DiaBloS ``run_tuning_simulation``
after a fresh load, PathSim ``Simulation.run``). Diagram loading, QApplication
start-up and model construction are excluded. Each measurement is one warm-up
plus ``REPEATS`` timed runs; median and min are reported.
"""

import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OUT_DIR = Path(
    os.environ.get("DIABLOS_BENCH_OUT", REPO / "scripts" / "benchmarks" / "pathsim" / "out")
)
REPEATS = int(os.environ.get("DIABLOS_BENCH_REPEATS", "5"))


def env_info():
    """Versions and host facts recorded alongside every result."""
    import scipy

    try:
        import pathsim

        ps_ver = pathsim.__version__
    except ImportError:
        ps_ver = None
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    uname = platform.uname()
    return {
        "diablos_commit": commit,
        "pathsim": ps_ver,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "python": platform.python_version(),
        "machine": uname.machine,
        "system": f"{uname.system} {uname.release}",
        "processor": platform.processor() or uname.machine,
        "wsl": "microsoft" in uname.release.lower(),
        "repeats": REPEATS,
    }


def time_call(run_once, repeats=None):
    """Time ``run_once() -> (elapsed_seconds, payload)``; warm-up excluded.

    ``run_once`` measures its own elapsed time so that per-run setup (diagram
    reload, model construction) stays outside the timed region.
    """
    repeats = REPEATS if repeats is None else repeats
    run_once()  # warm-up
    times, payload = [], None
    for _ in range(repeats):
        dt, payload = run_once()
        times.append(dt)
    return {"median_s": statistics.median(times), "min_s": min(times), "n": repeats}, payload


# ---------------------------------------------------------------- DiaBloS side


def diablos_run(
    path, sim_time, sim_dt, use_fast_solver=True, solver_method="RK45", rtol=1e-9, atol=1e-12
):
    """Load ``path`` fresh and run it once; return ``(elapsed, result)``.

    ``result`` holds the harvested Scope signals, the solver path that actually
    ran (``last_solver_type``), and an error string when the run was rejected.
    """
    from lib.analysis.resim import harvest_scope_signals
    from lib.cli import load_diagram

    dsim, _ = load_diagram(str(path))
    dsim.use_fast_solver = use_fast_solver
    dsim.solver_method = solver_method
    dsim.rtol = rtol
    dsim.atol = atol
    dsim.zero_crossing = True
    t0 = time.perf_counter()
    ok, err = dsim.run_tuning_simulation(sim_time, sim_dt)
    elapsed = time.perf_counter() - t0
    result = {
        "ok": bool(ok),
        "error": None if ok else str(err),
        # run_tuning_simulation does not set last_solver_type; the compiled
        # path leaves solver diagnostics behind, the interpreter path does not.
        "solver_path": (
            ("compiled" if dsim.last_solver_diagnostics_summary else "interpreter") if ok else None
        ),
        "fallback_reason": dsim.engine.get_compile_fallback_reason(),
        "data": harvest_scope_signals(dsim) if ok else None,
    }
    return elapsed, result


def diablos_signal(result, label):
    """``(t, y)`` for the Scope channel ``label`` of a ``diablos_run`` result."""
    data = result["data"]
    return np.asarray(data["timeline"], float), np.asarray(data["signals"][label], float)


# ---------------------------------------------------------------- output


def write_result(name, payload):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"scenario": name, "env": env_info(), **payload}
    path = OUT_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=_jsonable))
    print(f"wrote {path}")
    return path


def _jsonable(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj).__name__)
