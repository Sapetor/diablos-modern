#!/usr/bin/env python3
"""Run the numerical validation cases and print the results as Markdown.

Same cases, same tolerances and same code as ``pytest tests/validation`` -- both
read the registry in ``tests/validation/_cases.py`` -- but this entry point
needs no pytest and prints a table instead of asserting, so the numbers in
``docs/VALIDATION.md`` can be regenerated and pasted after any change to the
solver.

    python scripts/validation_report.py                 # the whole table
    python scripts/validation_report.py --case heat_eigenmode advection_pulse
    python scripts/validation_report.py --list

Exit status is 0 when every case is inside its tolerance and 1 otherwise, so it
can also be used as a coarse smoke check in a build script.
"""

import argparse
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Headless before anything imports Qt or matplotlib.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")


def _format_step(step):
    """Render the step / grid column, which is a number for most cases."""
    if isinstance(step, float):
        return "{0:g}".format(step)
    return str(step)


def _escape(text):
    """Markdown cells are pipe-delimited, and several case names contain |x|."""
    return str(text).replace("|", "\\|")


def _render_table(rows):
    """The Markdown table: one line per measured row."""
    header = (
        "| Case | Path | Method | dt / grid | Max error | Tolerance | Pass |",
        "| --- | --- | --- | --- | ---: | ---: | :---: |",
    )
    lines = list(header)
    for row in rows:
        lines.append(
            "| {name} | {path} | {method} | {step} | {error:.2e} | {tol:.1e} | {mark} |".format(
                name=_escape(row.name),
                path=row.path,
                method=_escape(row.method),
                step=_format_step(row.step),
                error=row.error,
                tol=row.tol,
                mark="yes" if row.passed else "**NO**",
            )
        )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--case",
        nargs="+",
        metavar="NAME",
        help="run only these cases (default: all); see --list for the names",
    )
    parser.add_argument("--list", action="store_true", help="list the case names and exit")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="print only the table, not the progress lines or the summary",
    )
    args = parser.parse_args(argv)

    from tests.validation import _cases as cases

    registry = dict(cases.ALL_CASES)
    if args.list:
        for name, _fn in cases.ALL_CASES:
            print(name)
        return 0

    selected = [name for name, _fn in cases.ALL_CASES]
    if args.case:
        unknown = [name for name in args.case if name not in registry]
        if unknown:
            parser.error("unknown case(s): {0}".format(", ".join(unknown)))
        selected = list(args.case)

    rows = []
    started = time.time()
    for name in selected:
        if not args.quiet:
            print("running {0} ...".format(name), file=sys.stderr)
        rows.extend(registry[name]())

    print(_render_table(rows))

    failed = [row for row in rows if not row.passed]
    if not args.quiet:
        print("")
        print(
            "{0} case rows, {1} outside tolerance, {2:.1f}s".format(
                len(rows), len(failed), time.time() - started
            )
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
