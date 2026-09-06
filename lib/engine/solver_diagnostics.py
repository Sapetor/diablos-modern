"""Compiled-solver run diagnostics.

Pure helpers extracted from :class:`~lib.engine.simulation_engine.SimulationEngine`:
they turn a scipy ``solve_ivp`` result (or a fixed-step stand-in) plus timing and
configuration numbers into a compact, UI/log-friendly dict, and format that dict
as a one-line summary. Kept free of engine state so they can be unit-tested in
isolation — the engine keeps thin wrappers that supply its running cache-hit
counters and stash the result on ``self.last_solver_diagnostics``.

This module also owns the **stiffness heuristic** (:func:`estimate_stiffness`):
after a run with an *explicit* method it decides whether the diagram looks stiff
enough that an implicit solver (Radau / BDF / LSODA) would be the better tool.
See that function for the two measurements and why both are needed.
"""

import logging
import math
from typing import Any, Callable, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

#: RHS evaluations one accepted step costs, per explicit scipy method. Used to
#: turn the only work counter a scipy result exposes (``nfev``) into an estimate
#: of how many internal steps the solver actually took. The implicit methods are
#: deliberately absent: their ``nfev`` also covers Jacobian/Newton work, and the
#: heuristic has nothing to suggest to someone already using one.
EXPLICIT_METHOD_STAGES = {"RK45": 6, "RK23": 3, "DOP853": 12}

#: Estimated internal solver steps per *output* sample above which the solver is
#: doing markedly more work than the requested output resolution needs.
STIFFNESS_WORK_RATIO = 4.0

#: ``max|Re lambda| * dt`` above which the fastest decaying mode lives far below
#: one output step -- the regime where an explicit method is limited by its
#: stability region rather than by accuracy.
STIFFNESS_INDEX = 20.0

#: Skip the Jacobian probe above this state count: it costs ``n+1`` RHS
#: evaluations per sample point, which stops being "cheap" for a 2-D PDE.
STIFFNESS_MAX_STATES = 64

#: Trajectory points at which the Jacobian is sampled (evenly spaced).
STIFFNESS_SAMPLE_POINTS = 5

#: What the heuristic recommends, and what the ``"auto"`` solver setting picks:
#: LSODA switches between an Adams (non-stiff) and a BDF (stiff) formula on its
#: own, so it is the safe suggestion when we only *suspect* stiffness.
STIFFNESS_SUGGESTED_METHOD = "LSODA"


def solver_attr(sol, name, default=None):
    """Read an attribute off a solver result object, tolerating stand-ins that
    omit it (fixed-step integrators don't expose ``nfev``/``njev``/``nlu``)."""
    return getattr(sol, name, default)


def _finite_difference_jacobian(model_func: Callable, t: float, y: np.ndarray) -> np.ndarray:
    """Forward-difference Jacobian of ``model_func`` at ``(t, y)``.

    ``n + 1`` RHS evaluations. Only ever used for an order-of-magnitude read on
    the eigenvalues, so the plain forward difference (rather than a central one)
    is accurate enough and half the cost.
    """
    y = np.asarray(y, dtype=float)
    f0 = np.asarray(model_func(t, y), dtype=float).ravel()
    n = y.size
    jac = np.empty((f0.size, n), dtype=float)
    for i in range(n):
        # sqrt(machine eps), scaled by the component's own magnitude.
        h = 1.4901161193847656e-08 * max(1.0, abs(float(y[i])))
        y_pert = y.copy()
        y_pert[i] += h
        jac[:, i] = (np.asarray(model_func(t, y_pert), dtype=float).ravel() - f0) / h
    return jac


def _spectrum_at(model_func: Callable, t: float, y: np.ndarray):
    """``(max|Re lambda|, stiffness ratio)`` of the Jacobian at ``(t, y)``.

    The ratio is over the *non-negligible* real parts only; a purely imaginary
    spectrum (an undamped oscillator) has no decay to be stiff about and yields
    ``(0.0, None)``.
    """
    jac = _finite_difference_jacobian(model_func, t, y)
    if jac.shape[0] != jac.shape[1] or not np.all(np.isfinite(jac)):
        return None, None
    eigenvalues = np.linalg.eigvals(jac)
    real_parts = np.abs(eigenvalues.real)
    if not np.all(np.isfinite(real_parts)):
        return None, None
    fastest = float(real_parts.max()) if real_parts.size else 0.0
    # "Negligible" is relative to the fastest mode: an eigenvalue 12 orders down
    # is numerical noise from the finite difference, not a slow physical mode.
    significant = real_parts[real_parts > max(fastest * 1e-12, 1e-300)]
    ratio = float(fastest / significant.min()) if significant.size else None
    return fastest, ratio


def estimate_stiffness(
    *,
    sol,
    method: str,
    dt: float,
    n_states: int,
    model_func: Optional[Callable] = None,
) -> Optional[Dict[str, Any]]:
    """Decide whether an explicit-solver run looks stiff. ``None`` if not applicable.

    Two independent measurements have to agree, because either one alone
    misfires:

    ``work_ratio``
        Estimated internal solver steps per output sample,
        ``nfev / (stages * n_output_intervals)``. High means the solver stepped
        far finer than the output grid the user asked for -- the *cost* symptom.
        On its own it also fires for a perfectly smooth system sampled coarsely
        (a 20 s sine written out every 0.5 s), where an implicit solver would
        not help at all.

    ``stiffness_index``
        ``max|Re lambda| * dt``, from a finite-difference Jacobian sampled at a
        few points of the returned trajectory. High means the fastest *decaying*
        mode has a time constant far below one output step -- the regime where
        an explicit method's step is bounded by its stability region rather than
        by accuracy, which is precisely what an implicit method fixes. On its
        own it also fires for a fast mode that has already died out, where the
        solver has long since stretched its step back out and costs nothing.

        The real part is what counts: a fast *oscillation* (purely imaginary
        eigenvalues) is not stiffness, and switching to Radau/LSODA buys nothing
        there -- the step is small because the answer genuinely moves that fast.

    Args:
        sol: The solver result (needs ``t``, ``y``, ``nfev``).
        method: The method actually used. Anything outside
            :data:`EXPLICIT_METHOD_STAGES` returns ``None``.
        dt: The output step (the simulation ``dt``).
        n_states: Size of the ODE state vector.
        model_func: The compiled RHS ``f(t, y)``. Without it only ``work_ratio``
            is measured, and nothing is ever flagged (one symptom is not enough).

    Returns:
        ``{"suspected", "work_ratio", "n_steps_est", "stiffness_index",
        "eig_ratio", "suggested_method"}`` or ``None``.
    """
    stages = EXPLICIT_METHOD_STAGES.get(str(method))
    if stages is None or int(n_states) <= 0:
        return None

    times = np.asarray(solver_attr(sol, "t", None) if sol is not None else None, dtype=float)
    n_intervals = max(1, times.size - 1)
    nfev = solver_attr(sol, "nfev", None)
    if nfev is None or times.size < 2 or not (float(dt) > 0):
        return None

    n_steps_est = float(nfev) / float(stages)
    work_ratio = n_steps_est / float(n_intervals)

    result = {
        "suspected": False,
        "work_ratio": float(work_ratio),
        "n_steps_est": int(round(n_steps_est)),
        "stiffness_index": None,
        "eig_ratio": None,
        "suggested_method": STIFFNESS_SUGGESTED_METHOD,
    }

    # The Jacobian probe is the expensive half, so it only runs once the cheap
    # half has already said the solver worked harder than the grid needed.
    if work_ratio < STIFFNESS_WORK_RATIO:
        return result
    if model_func is None or int(n_states) > STIFFNESS_MAX_STATES:
        return result

    states = np.asarray(solver_attr(sol, "y", None), dtype=float)
    if states.ndim != 2 or states.shape[1] == 0:
        return result

    sample_idx = np.unique(np.linspace(0, states.shape[1] - 1, STIFFNESS_SAMPLE_POINTS).astype(int))
    fastest = 0.0
    eig_ratio = None
    try:
        for i in sample_idx:
            rate, ratio = _spectrum_at(model_func, float(times[i]), states[:, i])
            if rate is None:
                continue
            if rate > fastest:
                fastest = rate
            if ratio is not None and (eig_ratio is None or ratio > eig_ratio):
                eig_ratio = ratio
    except Exception:  # noqa: BLE001 - a diagnostic must never break a finished run
        logger.debug("Stiffness Jacobian probe failed", exc_info=True)
        return result

    if not math.isfinite(fastest):
        return result

    index = fastest * float(dt)
    result["stiffness_index"] = float(index)
    result["eig_ratio"] = None if eig_ratio is None else float(eig_ratio)
    result["suspected"] = bool(index >= STIFFNESS_INDEX)
    return result


def format_stiffness_for_log(stiffness: Optional[Dict[str, Any]], method_used: str) -> str:
    """Warning text for a run the heuristic flagged. ``''`` when it did not.

    Developer-facing (log) text, so it stays English; the GUI phrases its own
    translated version from the same numbers.
    """
    if not stiffness or not stiffness.get("suspected"):
        return ""
    return (
        "This diagram looks stiff: {method} took about {ratio:.0f} internal steps per "
        "output sample and the fastest decaying mode is ~{index:.0f}x faster than one "
        "output step. An implicit solver ({suggested}, Radau or BDF) will usually be "
        "far quicker and no less accurate -- change it in Simulation settings, or pick "
        "the 'auto' solver.".format(
            method=method_used,
            ratio=stiffness.get("work_ratio", 0.0),
            index=stiffness.get("stiffness_index", 0.0),
            suggested=stiffness.get("suggested_method", STIFFNESS_SUGGESTED_METHOD),
        )
    )


def build_diagnostics(
    *,
    sol,
    success,
    method_requested,
    method_used,
    backend,
    t_span,
    dt,
    rtol,
    atol,
    n_states,
    n_blocks,
    n_lines,
    compile_cache_hit,
    compile_cache_hits_total,
    compile_cache_misses_total,
    compile_time,
    solve_time,
    replay_time,
    total_time,
    fallback_reason=None,
    failure_stage=None,
    output_range=None,
    zero_crossing=None,
    stiffness=None,
) -> Dict[str, Any]:
    """Build the compact diagnostics dict for a single compiled run.

    ``compile_cache_hits_total`` / ``compile_cache_misses_total`` are the
    engine's running counters, passed in so this stays state-free.
    """
    times = getattr(sol, "t", None)
    n_time_points = 0 if times is None else len(times)
    return {
        "success": bool(success),
        "failure_stage": failure_stage,
        "message": str(solver_attr(sol, "message", "") or ""),
        "status": solver_attr(sol, "status", None),
        "backend": backend,
        "method_requested": method_requested,
        "method_used": method_used,
        "fallback_reason": fallback_reason,
        "rtol": rtol,
        "atol": atol,
        "t_start": float(t_span[0]),
        "t_end": float(t_span[1]),
        "dt": float(dt),
        "n_states": int(n_states),
        "n_blocks": int(n_blocks),
        "n_lines": int(n_lines),
        "n_time_points": int(n_time_points),
        "n_output_steps": max(0, int(n_time_points) - 1),
        "nfev": solver_attr(sol, "nfev", None),
        "njev": solver_attr(sol, "njev", None),
        "nlu": solver_attr(sol, "nlu", None),
        "compile_cache_hit": bool(compile_cache_hit),
        "compile_cache_hits_total": int(compile_cache_hits_total),
        "compile_cache_misses_total": int(compile_cache_misses_total),
        "compile_wall_time": float(compile_time),
        "solve_wall_time": float(solve_time),
        "replay_wall_time": float(replay_time),
        "total_wall_time": float(total_time),
        "output_range": output_range,
        # None on the interpreted / fixed-step / algebraic paths, which have no
        # event machinery; otherwise the summary dict from
        # lib.engine.zero_crossing (events located, segments, guard state).
        "zero_crossing": zero_crossing,
        # None whenever the heuristic does not apply (implicit or fixed-step
        # method, no states, failed solve); otherwise the dict from
        # estimate_stiffness. The flat bool is the one the UI/CLI branch on.
        "stiffness": stiffness,
        "stiffness_suspected": bool((stiffness or {}).get("suspected")),
    }


def format_diagnostics_for_log(diagnostics: Dict[str, Any]) -> str:
    """One-line human/log summary of a diagnostics dict."""
    cache = "hit" if diagnostics.get("compile_cache_hit") else "miss"
    nfev = diagnostics.get("nfev")
    nfev_text = "n/a" if nfev is None else str(nfev)
    zc = diagnostics.get("zero_crossing") or {}
    if zc.get("enabled"):
        zc_text = "events={}{} ".format(
            zc.get("n_events", 0), " (guard tripped)" if zc.get("guard_tripped") else ""
        )
    else:
        zc_text = ""
    stiff_text = "stiffness=suspected " if diagnostics.get("stiffness_suspected") else ""
    return (
        f"method={diagnostics.get('method_used')} "
        f"backend={diagnostics.get('backend')} "
        f"states={diagnostics.get('n_states')} "
        f"points={diagnostics.get('n_time_points')} "
        f"nfev={nfev_text} "
        f"{zc_text}"
        f"{stiff_text}"
        f"cache={cache} "
        f"compile={diagnostics.get('compile_wall_time', 0.0):.4f}s "
        f"solve={diagnostics.get('solve_wall_time', 0.0):.4f}s "
        f"replay={diagnostics.get('replay_wall_time', 0.0):.4f}s "
        f"total={diagnostics.get('total_wall_time', 0.0):.4f}s"
    )
