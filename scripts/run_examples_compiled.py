"""
Headless compiled/fast-solver smoke runner for every examples/*.diablos.

Mirrors the load -> deserialize -> resolve_params -> initialize_execution
pipeline from tests/integration/test_example_runs.py, then drives the COMPILED
fast solver directly (engine.run_compiled_simulation) so that any exception in
compile_system or the per-step replay loop propagates with a full traceback.

Unlike DSim.run_tuning_simulation (which silently falls back to the interpreter
when the compiled path raises), this script forces the compiled path and records
the exception type, message, and the block/fn implicated by the traceback.

Run (offscreen Qt required):
    QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python scripts/run_examples_compiled.py \
        [run | --orchestrate | --summary | --reset]

``run`` (default) is a fast in-process loop; ``--orchestrate`` runs each example
in an isolated subprocess so a native crash in one cannot abort the whole sweep.
On WSL the bundled .venv-win interpreter can fast-fail under bulk headless Qt;
the system python3 (with PyQt5/numpy/scipy) is more reliable for this script.
"""

import json
import re
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO / "examples"
sys.path.insert(0, str(REPO))


# Module-level reference to the QApplication. Without this, the app the
# callers create is only a local; once it is GC'd the underlying QApplication
# is destroyed and the next QPixmap aborts ("Must construct a QGuiApplication
# before a QPixmap"). Holding it here keeps it alive for the whole process.
_QAPP = None


def ensure_qapp():
    """DSim builds real PyQt5 objects; a QApplication must exist (and stay
    alive) first or the interpreter fast-fails under offscreen Qt."""
    global _QAPP
    from PyQt5.QtWidgets import QApplication
    _QAPP = QApplication.instance() or QApplication(sys.argv)
    return _QAPP

# Cap runtime per example so the replay loop is exercised but stays fast.
MAX_SIM_TIME = 0.5  # seconds of sim time (clamped from the file's sim_time)

# Tester-bug runtime signatures to flag specially.
SIGNATURES = [
    ("length-1 arrays", re.compile(r"only length-1 arrays", re.I)),
    ("ambiguous truth value", re.compile(r"truth value of an array is ambiguous", re.I)),
    ("KeyError (dropdown?)", re.compile(r"^KeyError", re.I)),
    ("ZeroDivision", re.compile(r"ZeroDivisionError|division by zero|divide by zero", re.I)),
    ("overflow", re.compile(r"OverflowError|overflow encountered", re.I)),
]


def classify_signature(exc_type, message):
    blob = f"{exc_type}: {message}"
    hits = [name for name, rx in SIGNATURES if rx.search(blob)]
    return ", ".join(hits) if hits else ""


def implicated_block(tb_text):
    """Best-effort: pull the deepest blocks/ frame and any block name/fn."""
    block_file = ""
    for line in tb_text.splitlines():
        m = re.search(r'File "([^"]*[/\\]blocks[/\\][^"]+\.py)", line \d+, in (\w+)', line)
        if m:
            block_file = f"{Path(m.group(1)).name}:{m.group(2)}"
    return block_file


def run_one(example_file):
    """Returns dict with keys: name, status, exc_type, message, signature, block,
    solver, n_states, block_fns."""
    from lib.lib import DSim
    from lib.workspace import WorkspaceManager

    prev_instance = WorkspaceManager._instance
    WorkspaceManager._instance = None

    result = {
        "name": example_file.name,
        "status": "PASS",
        "exc_type": "",
        "message": "",
        "signature": "",
        "block": "",
        "solver": "",
        "n_states": None,
        "block_fns": [],
    }

    try:
        dsim = DSim()
        data = dsim.file_service.load(filepath=str(example_file))
        if data is None:
            result["status"] = "FAIL"
            result["exc_type"] = "LoadError"
            result["message"] = "file_service.load returned None"
            return result

        dsim.deserialize(data)

        # Record block fn coverage from the raw file (flat, pre-flatten).
        try:
            result["block_fns"] = sorted({
                b.get("block_fn", "") for b in data.get("blocks_data", [])
            } - {""})
        except Exception:
            pass

        workspace_manager = WorkspaceManager()

        def resolve_recursive(blocks):
            for block in blocks:
                block.exec_params = workspace_manager.resolve_params(block.params)
                block.exec_params.update(
                    {k: v for k, v in block.params.items() if k.startswith("_")}
                )
                block.exec_params["dtime"] = dsim.sim_dt
                dsim.engine.set_block_type(block)
                if getattr(block, "block_type", "") == "Subsystem":
                    resolve_recursive(block.sub_blocks)

        root_blocks, root_lines = dsim.get_root_context()
        resolve_recursive(root_blocks)

        sim_dt = dsim.sim_dt
        sim_time = min(dsim.sim_time, MAX_SIM_TIME)
        # Need at least a couple of steps to exercise the replay loop.
        if sim_time < 2 * sim_dt:
            sim_time = max(2 * sim_dt, sim_time)

        dsim.engine.update_sim_params(sim_time, sim_dt)
        ok = dsim.engine.initialize_execution(root_blocks, root_lines)
        if not ok:
            result["status"] = "FAIL"
            result["exc_type"] = "InitError"
            result["message"] = dsim.error_msg or getattr(dsim.engine, "error_msg", "") or "initialize_execution returned False"
            return result

        # Match the live path: reset data, identify memory blocks.
        dsim.reset_execution_data()
        dsim.engine.identify_memory_blocks()

        # Force the COMPILED path. If the system is not compilable, note it and
        # skip (the compiled solver genuinely cannot run it).
        flat_blocks = dsim.engine.active_blocks_list or dsim.blocks_list
        flat_lines = dsim.engine.active_line_list or dsim.line_list
        compilable = dsim.engine.check_compilability(flat_blocks)
        if not compilable:
            result["status"] = "SKIP-INTERP"
            result["solver"] = "Interpreter (not compilable)"
            result["message"] = "System not fully compilable; compiled solver does not apply"
            return result

        result["solver"] = "Compiled"
        t_span = (0.0, sim_time)
        success = dsim.engine.run_compiled_simulation(
            flat_blocks, flat_lines, t_span, sim_dt
        )

        try:
            result["n_states"] = int(getattr(dsim.engine, "outs", []).shape[0])
        except Exception:
            pass

        if not success:
            # run_compiled_simulation returned False without raising (solver
            # failure / non-compilable flattened system).
            result["status"] = "FAIL"
            result["exc_type"] = "CompiledReturnedFalse"
            result["message"] = (
                "run_compiled_simulation returned False "
                "(solver failed or flattened system uncompilable)"
            )
        return result

    except Exception as e:
        # Capture any block/solver error so the sweep never stops on one example.
        # Native fast-fails (0xC0000409) are not Python-catchable; the marker /
        # subprocess mechanism records those instead.
        tb_text = traceback.format_exc()
        result["status"] = "FAIL"
        result["exc_type"] = type(e).__name__
        result["message"] = str(e).strip().replace("\n", " ")[:400]
        result["signature"] = classify_signature(result["exc_type"], str(e))
        result["block"] = implicated_block(tb_text)
        result["_tb"] = tb_text
        return result
    finally:
        WorkspaceManager._instance = prev_instance


# Incremental result store. Each completed example is appended as one JSON line
# (JSONL) and fsync'd immediately, so a hard native crash (0xC0000409 fast-fail)
# on a later example never loses already-completed results. A separate "current"
# marker records the example about to run; if the process dies mid-example, the
# next launch reads the marker, records that example as a HardCrash, and resumes.
RESULTS_PATH = REPO / "scripts" / ".examples_compiled_results.jsonl"
MARKER_PATH = REPO / "scripts" / ".examples_compiled_current.txt"


def _append_result(rec):
    import os
    rec.pop("_tb", None)
    with open(RESULTS_PATH, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _load_results():
    if not RESULTS_PATH.exists():
        return []
    out = []
    with open(RESULTS_PATH) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out


def _set_marker(name):
    import os
    with open(MARKER_PATH, "w") as fh:
        fh.write(name)
        fh.flush()
        os.fsync(fh.fileno())


def _clear_marker():
    MARKER_PATH.unlink(missing_ok=True)


def run_loop():
    """Run every example in THIS process, persisting each result incrementally.

    Resumes from prior partial runs. If a marker survived from a previous launch
    that example crashed the interpreter natively last time -> record HardCrash
    and skip it. A relaunch wrapper re-invokes until all examples are recorded.
    """
    ensure_qapp()
    files = sorted(EXAMPLES_DIR.glob("*.diablos"))

    done = {r["name"] for r in _load_results()}

    if MARKER_PATH.exists():
        crashed = MARKER_PATH.read_text().strip()
        if crashed and crashed not in done:
            _append_result({
                "name": crashed, "status": "FAIL",
                "exc_type": "HardCrash(native fast-fail)",
                "message": "interpreter died mid-example (native crash, no Python traceback)",
                "signature": "", "block": "", "solver": "", "n_states": None,
                "block_fns": [],
            })
            done.add(crashed)
        _clear_marker()

    for f in files:
        if f.name in done:
            continue
        _set_marker(f.name)
        rec = run_one(f)
        _append_result(rec)
        _clear_marker()
        sys.stderr.write(f"[done] {f.name} -> {rec['status']} {rec['exc_type']}\n")
        sys.stderr.flush()


def summarize():
    results = _load_results()
    # De-dup by name keeping the last record.
    by_name = {}
    for r in results:
        by_name[r["name"]] = r
    results = list(by_name.values())

    passed = [r for r in results if r["status"] == "PASS"]
    skipped = [r for r in results if r["status"] == "SKIP-INTERP"]
    failed = [r for r in results if r["status"] == "FAIL"]

    # Coverage: union of block_fns across all examples.
    all_fns = set()
    for r in results:
        all_fns.update(r["block_fns"])

    out = {
        "total": len(results),
        "passed": len(passed),
        "skipped_not_compilable": len(skipped),
        "failed": len(failed),
        "failures": [
            {
                "name": r["name"],
                "exc_type": r["exc_type"],
                "message": r["message"],
                "signature": r["signature"],
                "block": r["block"],
                "solver": r["solver"],
            }
            for r in failed
        ],
        "skipped": [
            {"name": r["name"], "message": r["message"]} for r in skipped
        ],
        "passed_names": [r["name"] for r in passed],
        "block_fn_coverage": sorted(all_fns),
    }
    out["all_recorded"] = len(results) == len(list(EXAMPLES_DIR.glob("*.diablos")))
    print("=== JSON_RESULT_START ===")
    print(json.dumps(out, indent=2))
    print("=== JSON_RESULT_END ===")


def _single_child(name, out_path):
    """Run ONE example, write its JSON result to out_path, then os._exit(0) to
    skip Qt/native teardown. Launched as an isolated subprocess so a native
    crash (QPixmap-before-QGuiApplication, SIGABRT, fast-fail) on one example
    cannot take down the whole run."""
    import os
    ensure_qapp()  # QApplication first — before any block import constructs a QPixmap
    rec = run_one(EXAMPLES_DIR / name)
    rec.pop("_tb", None)
    with open(out_path, "w") as fh:
        json.dump(rec, fh)
        fh.flush()
        os.fsync(fh.fileno())
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def orchestrate():
    """Launch one isolated child subprocess per example, collecting results.
    Survives native crashes: a crashed child is recorded as HardCrash and the
    run continues to the next example."""
    import subprocess
    import tempfile

    files = sorted(EXAMPLES_DIR.glob("*.diablos"))
    done = {r["name"] for r in _load_results()}

    for f in files:
        if f.name in done:
            continue
        tmp = tempfile.NamedTemporaryFile(
            prefix="diablos_ex_", suffix=".json", delete=False
        )
        tmp.close()
        rec = None
        try:
            proc = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()),
                 "--single", f.name, tmp.name],
                capture_output=True, text=True, timeout=120,
            )
            rc, err = proc.returncode, (proc.stderr or "")
        except subprocess.TimeoutExpired:
            rc, err = "timeout", "exceeded 120s"
        try:
            txt = Path(tmp.name).read_text().strip()
            if txt:
                rec = json.loads(txt)
        except Exception:
            rec = None
        finally:
            Path(tmp.name).unlink(missing_ok=True)

        if rec is None:
            err_tail = [l for l in err.strip().splitlines()
                        if "Symbolic features" not in l]
            rec = {
                "name": f.name, "status": "FAIL",
                "exc_type": f"HardCrash(exit={rc})",
                "message": ("child crashed natively before writing result; "
                            f"stderr: {err_tail[-1] if err_tail else ''}")[:400],
                "signature": "", "block": "", "solver": "", "n_states": None,
                "block_fns": [],
            }
        _append_result(rec)
        sys.stderr.write(f"[orch] {f.name} -> {rec['status']} {rec['exc_type']}\n")
        sys.stderr.flush()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    # Create the QApplication as the very first runtime action (before any block
    # import can construct a QPixmap without a QGuiApplication present).
    if mode in ("run", "--orchestrate", "--single"):
        ensure_qapp()
    if mode == "--reset":
        RESULTS_PATH.unlink(missing_ok=True)
        MARKER_PATH.unlink(missing_ok=True)
        print("reset done")
    elif mode == "--summary":
        summarize()
    elif mode == "--single":
        _single_child(sys.argv[2], sys.argv[3])
    elif mode == "--orchestrate":
        orchestrate()
        summarize()
    else:
        run_loop()
        summarize()
