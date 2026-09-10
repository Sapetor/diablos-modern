"""Shared harness for the numerical validation suite.

Builds small diagrams headlessly and runs them through *both* DiaBloS
execution paths, so a validation case can state one tolerance per path:

* the **compiled** path -- ``SystemCompiler`` folds the diagram into a single
  ODE right-hand side that ``scipy.integrate.solve_ivp`` integrates adaptively
  (see ``docs/FAST_SOLVER.md``);
* the **interpreter** path -- a fixed-step loop that calls each block's
  ``execute()`` once per ``sim_dt``, with per-block integration methods.

The module deliberately imports no pytest: ``scripts/validation_report.py``
reuses it to regenerate the table in ``docs/VALIDATION.md``.
"""

import gc
import os
import shutil
import tempfile

import numpy as np

__all__ = [
    "ensure_qapp",
    "build",
    "add",
    "run",
    "release",
    "read_scopes",
    "RunResult",
    "max_abs_error",
    "rel_error",
    "observed_order",
]


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
_QAPP = None


def ensure_qapp():
    """A live ``QApplication``, created headless if the process has none.

    ``DSim`` owns pyqtgraph-backed plotting state and blocks build Qt paint
    objects for their icons, so a Qt application object has to exist before
    either is constructed. Under pytest the session ``qapp`` fixture supplies
    it; the report script has no fixtures, so it calls this. The instance is
    parked in a module global: PyQt6 destroys a QApplication whose last Python
    reference goes away, and the next QPixmap then aborts the process.
    """
    global _QAPP
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("MPLBACKEND", "Agg")
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    _QAPP = app
    return app


# --------------------------------------------------------------------------- #
# Diagram construction
# --------------------------------------------------------------------------- #
_PARAM_DEFAULTS = {}


def _defaults(block_name):
    """Every declared default of a block, so builder diagrams are complete.

    ``DiagramBuilder`` writes only the params it is handed, while the engine's
    interpreted initialization reads a block's whole parameter set (an
    Integrator wants ``method``, a Step wants ``type``). Pull the defaults from
    the block classes rather than restating them here.
    """
    if not _PARAM_DEFAULTS:
        # Constructing a block builds Qt paint objects for its icon, so the
        # QApplication has to exist first.
        ensure_qapp()
        from lib.block_loader import load_blocks

        for cls in load_blocks():
            try:
                block = cls()
                _PARAM_DEFAULTS[block.block_name] = {
                    name: meta["default"]
                    for name, meta in (block.params or {}).items()
                    if isinstance(meta, dict) and "default" in meta
                }
            except Exception:  # noqa: BLE001 - not every block is bare-constructible
                continue
    return dict(_PARAM_DEFAULTS.get(block_name, {}))


def build(sim_time, sim_dt):
    """A fresh :class:`~lib.diagram_builder.DiagramBuilder` for the given run."""
    ensure_qapp()
    from lib.diagram_builder import DiagramBuilder

    return DiagramBuilder(sim_time=sim_time, sim_dt=sim_dt)


def add(builder, block_type, name, params=None, **kwargs):
    """Add ``block_type`` with its declared defaults merged under ``params``."""
    merged = _defaults(block_type)
    merged.update(params or {})
    x = 60 + 120 * len(builder.blocks)
    return builder.add_block(block_type, x, 120, name=name, params=merged, **kwargs)


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #
def read_scopes(dsim):
    """Scope captures of a finished run, as ``(signals, fields)``.

    ``signals`` maps each channel's label to its 1-D trace; ``fields`` maps each
    Scope's *username* to the whole ``(n_samples, n_channels)`` capture, which
    is how a PDE field arrives.

    ``lib.analysis.resim.harvest_scope_signals`` applies the same naming rule
    for the ensemble/sweep UI (label first, block name as the fallback); this
    reader additionally returns the whole-capture ``fields`` view and
    disambiguates duplicate labels with a trailing ``'`` rather than ``#n``.
    """
    from lib.engine.block_params import runtime_params

    blocks = getattr(dsim.engine, "active_blocks_list", None) or dsim.blocks_list
    signals = {}
    fields = {}
    for block in blocks:
        if block.block_fn != "Scope":
            continue
        params = runtime_params(block)
        vec = params.get("vector")
        if vec is None:
            continue
        arr = np.asarray(vec, dtype=float)
        vec_dim = int(params.get("vec_dim", 1) or 1)
        if arr.ndim == 1 and vec_dim > 1 and arr.size % vec_dim == 0:
            arr = arr.reshape(-1, vec_dim)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        fields[getattr(block, "username", None) or block.name] = arr
        labels = params.get("vec_labels") or []
        for j in range(arr.shape[1]):
            label = labels[j] if j < len(labels) and isinstance(labels[j], str) else None
            name = label or "{}[{}]".format(block.name, j)
            while name in signals:
                name += "'"
            signals[name] = arr[:, j]
    return signals, fields


class RunResult(object):
    """One completed run: timeline, named Scope traces, solver diagnostics."""

    def __init__(self, timeline, signals, diagnostics, compiled, dsim=None, fields=None):
        self.t = np.asarray(timeline, dtype=float)
        self.signals = signals
        self.fields = fields or {}
        self.diagnostics = diagnostics
        self.compiled = compiled
        self.dsim = dsim

    def field(self, scope_name):
        """The ``(n_samples, n_nodes)`` capture of a vector Scope, with its time."""
        if scope_name not in self.fields:
            raise AssertionError("no Scope {!r}; have {}".format(scope_name, sorted(self.fields)))
        arr = self.fields[scope_name]
        n = min(len(arr), len(self.t))
        return self.t[:n], arr[:n]

    def signal(self, name=None):
        """One trace, trimmed to the timeline length, with its time vector.

        Returns ``(t, y)``. ``name`` selects a Scope label; with no name the
        diagram's only trace is returned.
        """
        if name is None:
            if len(self.signals) != 1:
                raise AssertionError(
                    "diagram has {} traces {}; name one".format(
                        len(self.signals), sorted(self.signals)
                    )
                )
            y = list(self.signals.values())[0]
        else:
            if name not in self.signals:
                raise AssertionError("no trace {!r}; have {}".format(name, sorted(self.signals)))
            y = self.signals[name]
        n = min(len(y), len(self.t))
        return self.t[:n], np.asarray(y, dtype=float)[:n]


def run(
    builder,
    tmp_path=None,
    compiled=True,
    solver_method=None,
    rtol=None,
    atol=None,
    zero_crossing=True,
    require_compiled=None,
):
    """Save ``builder``'s diagram, load it into a fresh ``DSim`` and run it.

    Args:
        builder: a populated :class:`DiagramBuilder`; its ``sim_time`` /
            ``sim_dt`` set the run horizon and output grid.
        tmp_path: directory for the intermediate ``.diablos``; a private temp
            directory is used when omitted.
        compiled: ``True`` runs the compiled fast-solver path, ``False`` forces
            the fixed-step interpreter.
        solver_method / rtol / atol: compiled-solver settings (ignored by the
            interpreter, which steps each block with its own ``method``).
        zero_crossing: compiled-path event detection.
        require_compiled: assert afterwards that the run really took the
            compiled path. Defaults to ``compiled`` -- a diagram that silently
            falls back to the interpreter (an unsupported block, a discrete
            rate, a failed compile) would otherwise be validated against the
            wrong tolerance. Pass ``False`` for the cases that are expected to
            fall back.

    Returns:
        :class:`RunResult`.
    """
    from lib.lib import DSim
    from lib.workspace import WorkspaceManager

    ensure_qapp()
    if require_compiled is None:
        require_compiled = compiled

    scratch = None
    if tmp_path is None:
        scratch = tempfile.mkdtemp(prefix="diablos-validation-")
        path = os.path.join(scratch, "diagram.diablos")
    else:
        path = os.path.join(str(tmp_path), "diagram.diablos")
    builder.save(path)

    WorkspaceManager._instance = None
    dsim = DSim()
    try:
        data = dsim.file_service.load(filepath=path)
        if data is None:
            raise AssertionError("failed to load the generated diagram")
        dsim.file_service.apply_loaded_data(data)
    finally:
        # The diagram is in memory from here on; nothing reads the file again.
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)
    dsim.use_fast_solver = bool(compiled)
    dsim.zero_crossing = bool(zero_crossing)
    if solver_method is not None:
        dsim.solver_method = solver_method
    if rtol is not None:
        dsim.rtol = rtol
    if atol is not None:
        dsim.atol = atol

    sim_time, sim_dt = builder.sim_time, builder.sim_dt
    ok, err = dsim.run_tuning_simulation(sim_time, sim_dt)
    if not ok:
        raise AssertionError("run failed: {}".format(err))

    # Only the compiled runner records diagnostics, and every run here gets a
    # fresh DSim -- so a non-empty dict is proof the compiled path ran, which
    # ``check_compilability`` alone would not be (it says the diagram *could*
    # compile, not that this run did).
    diagnostics = dsim.engine.get_solver_diagnostics()
    took_compiled = bool(diagnostics)
    if require_compiled and not took_compiled:
        raise AssertionError("diagram was expected to compile but fell back to the interpreter")
    if not compiled and took_compiled:
        raise AssertionError("interpreter run unexpectedly took the compiled path")

    timeline = getattr(dsim, "timeline", None)
    if timeline is None or len(np.atleast_1d(timeline)) == 0:
        raise AssertionError("run produced no timeline")
    signals, fields = read_scopes(dsim)
    return RunResult(timeline, signals, diagnostics, took_compiled, dsim, fields)


def release(*results):
    """Drop each run's ``DSim`` and collect, while Qt is idle.

    Every run owns pyqtgraph-backed plotting state; leaving a few dozen of them
    for CPython to free at an arbitrary later allocation can delete an item out
    from under a widget mid-construction and segfault the interpreter. Freeing
    them at a quiescent point keeps the deletions here.
    """
    for res in results:
        if isinstance(res, RunResult):
            res.dsim = None
    gc.collect()


# --------------------------------------------------------------------------- #
# Error measures
# --------------------------------------------------------------------------- #
def max_abs_error(y, reference):
    """``max |y - reference|`` over the overlapping prefix."""
    y = np.asarray(y, dtype=float)
    reference = np.asarray(reference, dtype=float)
    n = min(len(y), len(reference))
    return float(np.max(np.abs(y[:n] - reference[:n])))


def rel_error(y, reference):
    """Max absolute error normalized by the reference's peak magnitude."""
    reference = np.asarray(reference, dtype=float)
    scale = float(np.max(np.abs(reference)))
    if scale == 0.0:
        scale = 1.0
    return max_abs_error(y, reference) / scale


def observed_order(errors, refinement=2.0):
    """Empirical convergence order from errors at successively halved steps.

    ``errors`` is ordered coarse-to-fine; the returned list holds
    ``log(e_k / e_{k+1}) / log(refinement)`` for each consecutive pair.
    """
    orders = []
    for coarse, fine in zip(errors[:-1], errors[1:]):
        if fine <= 0.0:
            orders.append(float("inf"))
        else:
            orders.append(float(np.log(coarse / fine) / np.log(refinement)))
    return orders
