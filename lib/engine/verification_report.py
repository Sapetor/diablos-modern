"""Post-run verification report.

After a run, the GUI (``SimulationController``) and the headless CLI
(``run --verify``) summarise what the diagram's *verification* blocks hold:

* **Display** blocks: the value shown on the block.
* **StateVariable** blocks: the final state versus its initial value, judged
  as converged when the state moved away from the initial value or ended near
  zero (the usual optimisation demos minimise a quadratic).
* **Scope** blocks: first versus last sample. A Scope's ``verify_mode`` says
  how to judge it -- ``objective`` must drop by 90 % (or end below 1e-6),
  ``trajectory`` must change, ``comparison`` is shown without a verdict,
  ``none`` is skipped, and ``auto`` picks a mode from keywords in the name.

The report is plain text; :func:`build_verification_report` returns it with a
pass/fail verdict so a caller can log it, print it or turn it into an exit code.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from lib.engine.block_params import runtime_params

OBJECTIVE_KEYWORDS = ("f_", "cost", "obj", "norm", "value")
STATE_KEYWORDS = ("x_", "state", "traj", "position")

_RULE = "=" * 60
_THIN_RULE = "-" * 60


@dataclass
class VerificationReport:
    """The assembled report.

    ``has_data`` is False when the diagram holds no Display, StateVariable or
    Scope data to report on; ``text`` is then empty and ``passed`` is True.
    """

    text: str
    passed: bool
    has_data: bool


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


def _label(block) -> str:
    username = getattr(block, "username", "")
    return username if username else block.name


def collect_display_values(blocks: Iterable[Any]) -> Dict[str, Any]:
    """``{label or username: shown value}`` for every Display block.

    Read through ``runtime_params`` like the block renderer does: the block
    writes ``_display_value_`` into the dict ``execute()`` receives
    (``exec_params`` once a run happened), so reading ``block.params`` showed
    ``---`` for every Display after a headless run.
    """
    values = {}
    for block in blocks:
        if block.block_fn == "Display":
            params = runtime_params(block) or {}
            name = params.get("label", "") or block.username
            values[name] = params.get("_display_value_", "---")
    return values


def collect_state_variables(blocks: Iterable[Any]) -> Dict[str, Dict[str, Any]]:
    """``{name: {"final": array, "initial": array | None}}`` for StateVariable blocks."""
    states = {}
    for block in blocks:
        if block.block_fn != "StateVariable":
            continue
        exec_params = getattr(block, "exec_params", {}) or {}
        state = exec_params.get("_state_")
        if state is None:
            continue
        initial = exec_params.get("initial_value")
        states[_label(block)] = {
            "final": np.atleast_1d(state),
            "initial": np.atleast_1d(initial) if initial is not None else None,
        }
    return states


def _scope_samples(exec_params: Dict[str, Any]) -> Optional[np.ndarray]:
    """The Scope's recorded samples as a 1-D or (samples, vec_dim) array."""
    vec = exec_params.get("vector")
    if vec is None or not hasattr(vec, "__len__") or len(vec) == 0:
        return None
    arr = np.array(vec)
    vec_dim = exec_params.get("vec_dim", 1)
    if arr.ndim == 1 and vec_dim > 1 and len(arr) >= vec_dim:
        num_samples = len(arr) // vec_dim
        arr = arr[: num_samples * vec_dim].reshape(num_samples, vec_dim)
    return arr


def collect_scope_convergence(blocks: Iterable[Any]) -> Dict[str, Dict[str, Any]]:
    """First/last sample, sample count and ``verify_mode`` for every Scope with data."""
    scopes = {}
    for block in blocks:
        if block.block_fn != "Scope":
            continue
        exec_params = getattr(block, "exec_params", {}) or {}
        arr = _scope_samples(exec_params)
        if arr is None:
            continue
        if arr.ndim == 2:
            first, last = arr[0, :], arr[-1, :]
        else:
            first, last = arr[0], arr[-1]
        scopes[_label(block)] = {
            "labels": exec_params.get("vec_labels", block.username),
            "first": first,
            "last": last,
            "samples": len(arr),
            "data": arr,
            "verify_mode": exec_params.get("verify_mode", "auto"),
        }
    return scopes


# ---------------------------------------------------------------------------
# Judgement and formatting
# ---------------------------------------------------------------------------


def classify_scope(name: str, verify_mode: str) -> Tuple[bool, bool]:
    """``(is_objective, is_state)`` for a Scope; ``auto`` uses keywords in the name."""
    if verify_mode == "auto":
        lower = name.lower()
        return (
            any(kw in lower for kw in OBJECTIVE_KEYWORDS),
            any(kw in lower for kw in STATE_KEYWORDS),
        )
    if verify_mode == "objective":
        return True, False
    if verify_mode == "trajectory":
        return False, True
    return False, False  # "comparison" or unknown


def format_value(v) -> str:
    """Compact rendering of a scalar or short vector sample."""
    if v is None:
        return "N/A"
    v = np.atleast_1d(v)
    if len(v) == 1:
        return f"{float(v[0]):.6g}"
    if len(v) <= 3:
        return np.array2string(v, precision=4, suppress_small=True)
    return f"[{v[0]:.4g}, {v[1]:.4g}, ...]"


def _format_state(final: np.ndarray) -> str:
    if len(final) <= 4:
        return np.array2string(final, precision=6, suppress_small=True)
    return f"[{final[0]:.4g}, ..., {final[-1]:.4g}]"


def display_lines(display_values: Dict[str, Any]) -> List[str]:
    lines = ["", "📊 Display Values:"]
    for name, value in display_values.items():
        text = str(value)
        # A labelled Display already renders itself as "label: value".
        lines.append(f"   {text}" if text.startswith(f"{name}: ") else f"   {name}: {text}")
    return lines


def state_variable_lines(state_values: Dict[str, Dict[str, Any]]) -> Tuple[List[str], bool]:
    """Report lines for the StateVariable section and whether every check passed."""
    lines = ["", "🎯 Optimization Convergence:"]
    all_passed = True
    for name, info in state_values.items():
        final, initial = info["final"], info["initial"]
        final_norm = np.linalg.norm(final)
        converged_to_zero = final_norm < 1e-3

        if initial is not None:
            initial_norm = np.linalg.norm(initial)
            state_changed = not np.allclose(final, initial, rtol=1e-2)
            reduction = (initial_norm - final_norm) / initial_norm if initial_norm > 0 else 0
        else:
            state_changed = True
            reduction = None

        ok = bool(converged_to_zero or state_changed)
        all_passed &= ok
        lines.append(f"   {'✓' if ok else '✗'} {name}: {_format_state(final)}")
        if reduction is not None and reduction > 0:
            lines.append(f"      ‖x‖ reduced by {reduction * 100:.1f}%")
        if converged_to_zero:
            lines.append(f"      Converged to ‖x‖ = {final_norm:.2e}")
    return lines, all_passed


def scope_lines(scope_convergence: Dict[str, Dict[str, Any]]) -> Tuple[List[str], bool]:
    """Report lines for the Scope section and whether every check passed."""
    lines = ["", "📈 Signal Convergence:"]
    all_passed = True
    for name, info in scope_convergence.items():
        verify_mode = info.get("verify_mode", "auto")
        if verify_mode == "none":
            continue
        first, last = info["first"], info["last"]
        first_norm = np.linalg.norm(np.atleast_1d(first))
        last_norm = np.linalg.norm(np.atleast_1d(last))
        is_objective, is_state = classify_scope(name, verify_mode)
        transition = f"{format_value(first)} → {format_value(last)}"

        if is_objective and first_norm > 0:
            reduction = (first_norm - last_norm) / first_norm
            converged = bool(reduction > 0.9 or last_norm < 1e-6)
            all_passed &= converged
            lines.append(f"   {'✓' if converged else '✗'} {name}: {transition}")
            if reduction > 0:
                lines.append(f"      Reduced by {reduction * 100:.1f}%")
        elif is_state:
            changed = not bool(np.allclose(first, last, rtol=0.01))
            all_passed &= changed
            lines.append(f"   {'✓' if changed else '✗'} {name}: {transition}")
        else:
            lines.append(f"   • {name} ({info['samples']} pts): {transition}")
    return lines, all_passed


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def report_blocks(dsim) -> Sequence[Any]:
    """The blocks a report should read: the engine's flattened active list when
    a run populated it, else the diagram's top-level blocks."""
    engine = getattr(dsim, "engine", None)
    active = getattr(engine, "active_blocks_list", None) if engine is not None else None
    return active if active else dsim.blocks_list


def build_verification_report(blocks: Iterable[Any]) -> VerificationReport:
    """Assemble the report for ``blocks`` (see the module docstring)."""
    blocks = list(blocks)
    display_values = collect_display_values(blocks)
    state_values = collect_state_variables(blocks)
    scope_convergence = collect_scope_convergence(blocks)

    if not (display_values or state_values or scope_convergence):
        return VerificationReport(text="", passed=True, has_data=False)

    lines: List[str] = ["", _RULE, "VERIFICATION RESULTS", _RULE]
    all_passed = True
    if display_values:
        lines.extend(display_lines(display_values))
    if state_values:
        section, ok = state_variable_lines(state_values)
        lines.extend(section)
        all_passed &= ok
    if scope_convergence:
        section, ok = scope_lines(scope_convergence)
        lines.extend(section)
        all_passed &= ok

    lines.extend(["", _THIN_RULE])
    lines.append(
        "✓ VERIFICATION PASSED" if all_passed else "✗ VERIFICATION FAILED - Check values above"
    )
    lines.append(_RULE)
    return VerificationReport(text="\n".join(lines), passed=all_passed, has_data=True)
