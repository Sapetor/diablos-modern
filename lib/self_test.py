"""``DiaBloS self-test``: check that an installed (or packaged) app works.

The release workflow runs this through each PyInstaller bundle before
publishing it, because source-tree tests cannot see what the bundle is
missing: a data file left out of ``diablos.spec``, a hidden import, a Qt
platform plugin, or a process pool that cannot spawn the frozen executable.

Checks, each recorded pass/fail with a detail line:

* ``resources`` -- bundled examples and translation catalogs are present.
* ``simulate-compiled`` / ``simulate-interpreter`` -- the first-order step
  example matches its closed form ``y = 1 - exp(-(t - 1) / 2)`` (t >= 1).
* ``monte-carlo-workers`` -- a 2-process ensemble returns exactly what the
  in-process one does (worker processes start from the frozen executable).
* ``gui`` -- the main window builds, opens an example and paints its canvas
  (offscreen unless a platform is already chosen); skipped by ``--no-gui``.

The report goes to stdout and, with ``--report PATH``, to a JSON file: a
windowed Windows build has no stdout, so CI reads the file. Exit status is 0
only when every check passed.
"""

import argparse
import json
import os
import platform
import sys
import time
import traceback

import numpy as np

STEP_EXAMPLE = "first_order_step_response.diablos"
ENSEMBLE_EXAMPLE = "monte_carlo_robustness.diablos"
GUI_EXAMPLE = "pid_second_order.diablos"

#: Max |error| against the closed form, per solver (the interpreter is a
#: fixed-step method at the example's dt = 0.01).
TOLERANCE = {"compiled": 1e-6, "interpreter": 5e-3}


def _example(name):
    from lib.app_paths import resource_path

    return resource_path(os.path.join("examples", name))


def _check_resources():
    from lib.app_paths import locales_path, resource_path

    missing = [
        p
        for p in (
            _example(STEP_EXAMPLE),
            _example(ENSEMBLE_EXAMPLE),
            _example(GUI_EXAMPLE),
            locales_path("es.json"),
            resource_path(os.path.join("config", "default_config.json")),
        )
        if not os.path.isfile(p)
    ]
    if missing:
        raise AssertionError("missing bundled files: " + ", ".join(missing))
    return "examples, locales and config present"


def _check_simulation(solver):
    from lib import api

    result = api.load(_example(STEP_EXAMPLE)).run(solver=solver)
    t, y = result.time, result["output"]
    expected = np.where(t >= 1.0, 1.0 - np.exp(-(t - 1.0) / 2.0), 0.0)
    # The step itself is a discontinuity; compare away from it.
    mask = np.abs(t - 1.0) > 0.05
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(y))):
        raise AssertionError("non-finite time or output values")
    err = float(np.max(np.abs(y[mask] - expected[mask])))
    # Written so a NaN error fails too (any comparison with NaN is False).
    if not (len(t) > 100 and err <= TOLERANCE[solver]):
        raise AssertionError(f"{len(t)} samples, max error {err:.3g} > {TOLERANCE[solver]}")
    if solver == "compiled" and result.engine != "compiled":
        raise AssertionError(f"ran on the {result.engine} ({result.fallback_reason})")
    return f"{len(t)} samples on the {result.engine}, max error {err:.2e}"


def _check_workers():
    from lib import api

    diagram = api.load(_example(ENSEMBLE_EXAMPLE))
    serial = diagram.monte_carlo(4, seed=7, t=2.0, keep_traces=False, workers=1)
    pooled = diagram.monte_carlo(4, seed=7, t=2.0, keep_traces=False, workers=2)
    for label, ens in (("1 worker", serial), ("2 workers", pooled)):
        if ens["n_ok"] != 4:
            raise AssertionError(f"{label}: {ens['n_ok']} of 4 runs succeeded")
    names = sorted(serial["signals"])
    if not names or names != sorted(pooled["signals"]):
        raise AssertionError(f"signals {names} vs {sorted(pooled['signals'])}")
    for name in names:
        for stat in ("mean", "std"):
            if not np.array_equal(serial["signals"][name][stat], pooled["signals"][name][stat]):
                raise AssertionError(f"signal {name!r}: 2-worker {stat} differs from 1-worker")
    return f"4 runs x {len(names)} signals identical with 1 and 2 worker processes"


def _check_gui():
    from lib.qt_setup import configure_qt_env

    if not os.environ.get("QT_QPA_PLATFORM"):
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
    configure_qt_env()
    from PyQt6.QtWidgets import QApplication

    from modern_ui.main_window import ModernDiaBloSWindow

    app = QApplication.instance() or QApplication([sys.argv[0]])
    window = ModernDiaBloSWindow()
    window.show()
    app.processEvents()
    # The Examples-menu route: no recent-files entry, no trust prompt.
    if window.open_example(_example(GUI_EXAMPLE)) is None:
        raise AssertionError(f"could not open {GUI_EXAMPLE}")
    app.processEvents()
    n_blocks = len(window.dsim.blocks_list)
    image = window.canvas.grab()
    if n_blocks == 0 or image.isNull():
        raise AssertionError(f"{n_blocks} blocks, canvas image null={image.isNull()}")
    # hide(), not close(): closeEvent deletes the user's autosave file.
    window.dsim.dirty = False
    window.hide()
    app.processEvents()
    return f"main window opened {GUI_EXAMPLE} ({n_blocks} blocks) and painted the canvas"


def run_checks(gui=True):
    checks = [
        ("resources", _check_resources),
        ("simulate-compiled", lambda: _check_simulation("compiled")),
        ("simulate-interpreter", lambda: _check_simulation("interpreter")),
        ("monte-carlo-workers", _check_workers),
    ]
    if gui:
        checks.append(("gui", _check_gui))
    results = []
    for name, check in checks:
        start = time.perf_counter()
        try:
            detail, ok = check(), True
        except Exception as e:  # noqa: BLE001 -- every failure goes in the report
            detail, ok = f"{type(e).__name__}: {e}", False
            traceback.print_exc()
        results.append(
            {
                "name": name,
                "ok": ok,
                "detail": detail,
                "seconds": round(time.perf_counter() - start, 3),
            }
        )
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="diablos self-test", description="Check that this DiaBloS installation works."
    )
    parser.add_argument("--report", help="also write the results to this JSON file")
    parser.add_argument("--no-gui", action="store_true", help="skip the main-window check")
    args = parser.parse_args(argv)

    results = run_checks(gui=not args.no_gui)
    passed = all(r["ok"] for r in results)
    report = {
        "passed": passed,
        "frozen": bool(getattr(sys, "frozen", False)),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "checks": results,
    }
    for r in results:
        print(f"[{'PASS' if r['ok'] else 'FAIL'}] {r['name']}: {r['detail']}")
    print("self-test " + ("passed" if passed else "FAILED"))
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fp:
            json.dump(report, fp, indent=2)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
