"""Run indexed re-simulation tasks in-process or on a spawn-based process pool.

Both experiment runners reduce to the same shape: run ``n`` simulations of one
diagram, each with its own ``{(qualified_block, param): value}`` assignment, and
fold every run's compact record (:func:`lib.analysis.streaming.reduce_run`) into
an accumulator *in run-index order*. :func:`run_indexed` does exactly that,
either in the current process (``workers <= 1``: the historical behaviour, on
the caller's ``DSim``) or on a ``ProcessPoolExecutor`` with the ``spawn``
context (``workers > 1``).

Pool design:

  * Each worker process rebuilds the diagram **once** in its initializer from
    the parent's serialized root diagram (``DSim.serialize`` -> a fresh headless
    ``DSim`` via ``lib.cli.new_headless_dsim`` + ``deserialize`` -- the same path
    as ``DSim.clone_for_analysis``) and then reuses it for every task.
  * Assignments (per-run seeds, parameter samples, sweep values) are computed in
    the parent and shipped as plain floats/ints, so samplers never need to be
    picklable and every run sees exactly the values the in-process loop would.
  * Records are released to the accumulator strictly in index order (a bounded
    reorder window), so floating-point accumulation is identical for any
    worker count: ``workers=1`` and ``workers=N`` are bit-identical.
  * Cancellation: ``cancel_cb`` is polled while waiting; on cancel a shared
    ``multiprocessing.Event`` (which each worker forwards as its run's
    ``cancel_cb``) aborts in-flight runs, pending futures are cancelled, and
    already completed records are still released (in index order).
  * Workspace variables are shipped explicitly
    (:func:`workspace_variables_snapshot` / :func:`apply_workspace_variables`).
"""

import concurrent.futures as cf
import logging
import multiprocessing
import os

from lib.analysis.resim import harvest_scope_signals, iter_blocks_qualified, root_blocks_of
from lib.analysis.streaming import reduce_run

logger = logging.getLogger(__name__)

__all__ = [
    "run_indexed",
    "default_workers",
    "workspace_variables_snapshot",
    "apply_workspace_variables",
    "serialize_root",
]

_CANCELLED = "cancelled"


def default_workers():
    """A sensible GUI default: ``min(4, cpu_count - 1)``, at least 1."""
    return max(1, min(4, (os.cpu_count() or 2) - 1))


# --------------------------------------------------------------- workspace
def workspace_variables_snapshot(dsim):
    """``dsim``'s workspace as a picklable dict (see ``Workspace.to_dict``)."""
    return dsim.workspace.to_dict()


def apply_workspace_variables(snapshot, dsim):
    """Give the worker's ``dsim`` the parent diagram's workspace."""
    from lib.workspace import Workspace

    dsim.workspace = Workspace.from_dict(snapshot)


# --------------------------------------------------------------- diagram
def serialize_root(dsim):
    """Serialize the whole diagram, even while navigated inside a subsystem."""
    return dsim.serialize_root()


def set_block_param(block, name, value):
    """Write ``value`` to ``block.params`` and (when present) ``exec_params``."""
    block.params[name] = value
    if getattr(block, "exec_params", None):
        block.exec_params[name] = value


def apply_assignments(blocks_by_name, assignments):
    """Apply ``{(qualified_block, param): value}``; unknown blocks are skipped."""
    for (bn, pn), val in assignments.items():
        b = blocks_by_name.get(bn)
        if b is not None:
            set_block_param(b, pn, val)


def simulate_and_reduce(dsim, sim_time, sim_dt, keep_signals, cancel_cb):
    """Run once and reduce. Returns ``(status, record, err)``.

    ``status`` is ``"ok"``, ``"failed"`` or ``"cancelled"``; ``record`` is the
    :func:`reduce_run` record for ``"ok"`` (``None`` if no Scope data).
    """
    ok, err = dsim.run_tuning_simulation(sim_time, sim_dt, cancel_cb=cancel_cb)
    if cancel_cb is not None and cancel_cb():
        return _CANCELLED, None, err
    if not ok:
        return "failed", None, err
    return "ok", reduce_run(harvest_scope_signals(dsim), keep_signals), ""


# --------------------------------------------------------------- worker side
_WORKER = {}


def _worker_init(data, use_fast_solver, workspace_vars, cancel_event):
    """Build this worker process's private headless DSim (once per process)."""
    from lib.cli import new_headless_dsim

    dsim = new_headless_dsim()
    apply_workspace_variables(workspace_vars, dsim)
    dsim.deserialize(data)
    dsim.use_fast_solver = use_fast_solver
    dsim.dynamic_plot = False
    _WORKER["dsim"] = dsim
    _WORKER["blocks"] = dict(iter_blocks_qualified(root_blocks_of(dsim)))
    _WORKER["cancel"] = cancel_event


def _worker_run(index, assignments, sim_time, sim_dt, keep_signals):
    apply_assignments(_WORKER["blocks"], assignments)
    status, rec, err = simulate_and_reduce(
        _WORKER["dsim"], sim_time, sim_dt, keep_signals, _WORKER["cancel"].is_set
    )
    return index, status, rec, err


# --------------------------------------------------------------- driver
def run_indexed(
    dsim,
    n,
    assignments_for,
    on_record,
    sim_time,
    sim_dt,
    keep_signals,
    workers=1,
    progress_cb=None,
    cancel_cb=None,
    label="run",
):
    """Run ``n`` simulations; feed ``on_record(index, record|None)`` in index order.

    Args:
        dsim: the diagram (used directly when ``workers <= 1``; serialized for
            the pool otherwise -- the parent's ``dsim`` is then never mutated).
        assignments_for: ``callable(index) -> {(block, param): value}``.
        on_record: called with ``(index, record)`` for every completed run, in
            increasing index order; ``record`` is ``None`` for a failed run.
        keep_signals: forwarded to :func:`reduce_run`.
        workers: ``<= 1`` in-process; ``> 1`` spawn process pool.
        progress_cb: ``callable(done, total)`` as runs complete.
        cancel_cb: ``callable() -> bool``; when true, stop early.

    In-process mode applies assignments to ``dsim``'s blocks directly; the
    caller is responsible for snapshotting/restoring them.

    Returns the number of runs completed (released to ``on_record``).
    """
    workers = int(workers or 1)
    if workers <= 1 or n <= 1:
        return _run_inprocess(
            dsim,
            n,
            assignments_for,
            on_record,
            sim_time,
            sim_dt,
            keep_signals,
            progress_cb,
            cancel_cb,
            label,
        )
    return _run_pool(
        dsim,
        n,
        assignments_for,
        on_record,
        sim_time,
        sim_dt,
        keep_signals,
        min(workers, n),
        progress_cb,
        cancel_cb,
        label,
    )


def _run_inprocess(
    dsim,
    n,
    assignments_for,
    on_record,
    sim_time,
    sim_dt,
    keep_signals,
    progress_cb,
    cancel_cb,
    label,
):
    blocks_by_name = dict(iter_blocks_qualified(root_blocks_of(dsim)))
    done = 0
    for i in range(n):
        if cancel_cb is not None and cancel_cb():
            logger.info("Experiment cancelled after %d/%d %ss", i, n, label)
            break
        apply_assignments(blocks_by_name, assignments_for(i))
        # cancel_cb is forwarded so a cancel lands mid-run rather than only
        # between runs.
        status, rec, err = simulate_and_reduce(dsim, sim_time, sim_dt, keep_signals, cancel_cb)
        if status == _CANCELLED:
            logger.info("Experiment cancelled during %s %d/%d", label, i + 1, n)
            break
        if status == "failed":
            logger.warning("%s %d failed: %s", label.capitalize(), i, err)
        on_record(i, rec)
        done += 1
        if progress_cb:
            progress_cb(i + 1, n)
    return done


def _run_pool(
    dsim,
    n,
    assignments_for,
    on_record,
    sim_time,
    sim_dt,
    keep_signals,
    workers,
    progress_cb,
    cancel_cb,
    label,
):
    ctx = multiprocessing.get_context("spawn")
    cancel_event = ctx.Event()
    initargs = (
        serialize_root(dsim),
        bool(getattr(dsim, "use_fast_solver", True)),
        workspace_variables_snapshot(dsim),
        cancel_event,
    )
    # Bounded reorder window: at most this many runs submitted but not yet
    # released, so the buffer of out-of-order records stays O(workers).
    window = 2 * workers
    buffered = {}
    next_release = 0
    next_submit = 0
    completed = 0
    released = 0
    pending = set()
    cancelled = False

    executor = cf.ProcessPoolExecutor(
        max_workers=workers, mp_context=ctx, initializer=_worker_init, initargs=initargs
    )
    try:
        while next_release < n:
            if cancel_cb is not None and cancel_cb():
                cancelled = True
                break
            while next_submit < n and next_submit - next_release < window:
                pending.add(
                    executor.submit(
                        _worker_run,
                        next_submit,
                        assignments_for(next_submit),
                        sim_time,
                        sim_dt,
                        keep_signals,
                    )
                )
                next_submit += 1
            finished, pending = cf.wait(pending, timeout=0.1, return_when=cf.FIRST_COMPLETED)
            for fut in finished:
                idx, status, rec, err = fut.result()  # re-raises worker exceptions
                if status == _CANCELLED:
                    continue
                if status == "failed":
                    logger.warning("%s %d failed: %s", label.capitalize(), idx, err)
                buffered[idx] = rec
                completed += 1
                if progress_cb:
                    progress_cb(completed, n)
            while next_release in buffered:
                on_record(next_release, buffered.pop(next_release))
                next_release += 1
                released += 1
    finally:
        if cancelled or next_release < n:
            cancel_event.set()
            for fut in pending:
                fut.cancel()
        executor.shutdown(wait=True, cancel_futures=True)

    if cancelled:
        logger.info("Experiment cancelled after %d/%d %ss", completed, n, label)
        # Runs that finished before the cancel are still reported, in order.
        for idx in sorted(buffered):
            on_record(idx, buffered[idx])
            released += 1
    return released
