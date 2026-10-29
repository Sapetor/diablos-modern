#!/usr/bin/env bash
# Run a packaged DiaBloS executable before it is published.
#
#   tools/smoke_test_bundle.sh <path to the bundled executable> [report dir]
#
# Source-tree tests cannot see what a PyInstaller bundle is missing (a data
# file left out of diablos.spec, a hidden import, a Qt plugin, a process pool
# that cannot re-launch the frozen executable), so the release workflow runs
# the built app itself:
#
#   1. `DiaBloS self-test`: bundled resources, both solvers against a closed
#      form, a 2-process Monte-Carlo ensemble, and the main window built
#      offscreen (lib/self_test.py). Its JSON report is checked, not only the
#      exit code -- a windowed Windows build has no stdout.
#   2. `DiaBloS run`: the headless CLI simulates an example to CSV, and the CSV
#      is checked against y = 1 - exp(-(t - 1) / 2).
set -euo pipefail

EXE="$1"
OUT="${2:-smoke-report}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python}"
mkdir -p "$OUT"
export QT_QPA_PLATFORM=offscreen

echo "== self-test: $EXE"
status=0
"$EXE" self-test --report "$OUT/self-test.json" || status=$?
"$PYTHON" - "$OUT/self-test.json" "$status" <<'PY'
import json, sys

path, status = sys.argv[1], int(sys.argv[2])
try:
    report = json.load(open(path, encoding="utf-8"))
except (OSError, ValueError) as e:
    sys.exit(f"no self-test report ({e}); exit status {status}")
for c in report["checks"]:
    print(f"  [{'PASS' if c['ok'] else 'FAIL'}] {c['name']}: {c['detail']}")
if not report["frozen"]:
    sys.exit("the self-test did not run from a frozen build")
if not report["passed"] or status != 0:
    sys.exit(f"self-test failed (exit status {status})")
PY

echo "== run: first_order_step_response.diablos"
"$EXE" run "$HERE/examples/first_order_step_response.diablos" --out "$OUT/run.csv"
"$PYTHON" - "$OUT/run.csv" <<'PY'
import sys

import numpy as np

data = np.genfromtxt(sys.argv[1], delimiter=",", names=True)
t, y = data["t"], data["output"]
expected = np.where(t >= 1.0, 1.0 - np.exp(-(t - 1.0) / 2.0), 0.0)
mask = np.abs(t - 1.0) > 0.05
err = float(np.max(np.abs(y[mask] - expected[mask])))
print(f"  {len(t)} samples, max error {err:.2e}")
if len(t) < 100 or err > 1e-6:
    sys.exit("the bundled CLI run does not match the closed form")
PY
echo "== bundle smoke test passed"
