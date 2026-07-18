"""Compiled-solver run diagnostics.

Pure helpers extracted from :class:`~lib.engine.simulation_engine.SimulationEngine`:
they turn a scipy ``solve_ivp`` result (or a fixed-step stand-in) plus timing and
configuration numbers into a compact, UI/log-friendly dict, and format that dict
as a one-line summary. Kept free of engine state so they can be unit-tested in
isolation — the engine keeps thin wrappers that supply its running cache-hit
counters and stash the result on ``self.last_solver_diagnostics``.
"""

from typing import Any, Dict


def solver_attr(sol, name, default=None):
    """Read an attribute off a solver result object, tolerating stand-ins that
    omit it (fixed-step integrators don't expose ``nfev``/``njev``/``nlu``)."""
    return getattr(sol, name, default)


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
    }


def format_diagnostics_for_log(diagnostics: Dict[str, Any]) -> str:
    """One-line human/log summary of a diagnostics dict."""
    cache = "hit" if diagnostics.get("compile_cache_hit") else "miss"
    nfev = diagnostics.get("nfev")
    nfev_text = "n/a" if nfev is None else str(nfev)
    return (
        f"method={diagnostics.get('method_used')} "
        f"backend={diagnostics.get('backend')} "
        f"states={diagnostics.get('n_states')} "
        f"points={diagnostics.get('n_time_points')} "
        f"nfev={nfev_text} "
        f"cache={cache} "
        f"compile={diagnostics.get('compile_wall_time', 0.0):.4f}s "
        f"solve={diagnostics.get('solve_wall_time', 0.0):.4f}s "
        f"replay={diagnostics.get('replay_wall_time', 0.0):.4f}s "
        f"total={diagnostics.get('total_wall_time', 0.0):.4f}s"
    )
