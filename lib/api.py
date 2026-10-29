"""Importable headless simulation API for DiaBloS diagrams.

Load (or build) a diagram, change block parameters, simulate it and read every
Scope signal back as a numpy array -- from plain Python, with no window::

    from lib.api import load

    sim = load("examples/feedback_control.diablos")
    sim.set_param("gain1", "gain", 4.0)
    res = sim.run(t=10.0, dt=0.01)
    res.time               # 1-D ndarray
    res["scope1-0"]        # one Scope channel, 1-D ndarray
    res.engine             # "compiled" or "interpreter"
    res.diagnostics        # solver stats of a compiled run (method, nfev, ...)

The same object drives the analysis tools: :meth:`Diagram.monte_carlo`,
:meth:`Diagram.sweep`, :meth:`Diagram.find_operating_point` and
:meth:`Diagram.linearize`. See ``docs/user-guide/python-api.md`` for the full reference.

Everything here is a thin layer over the headless path the CLI and the
experiment tools already use (``lib/cli.py``, ``DSim.run_tuning_simulation``,
``lib/analysis/``). DSim builds Qt objects even headless, so the first
:func:`load` creates an offscreen QApplication if the process has none; no
window is ever shown.

Run from the repository root (DiaBloS installs no importable package).
Workspace variables belong to the diagram: each :class:`Diagram` owns its own
(:attr:`Diagram.workspace`, set with :meth:`Diagram.set_variable`), so several
diagrams with different variables can live in one process.
"""

import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from lib import cli
from lib.analysis.resim import harvest_scope_signals, iter_blocks_qualified, root_blocks_of

__all__ = ["Diagram", "SimResult", "SimulationError", "from_builder", "load"]

SOLVERS = ("compiled", "interpreter")


class SimulationError(RuntimeError):
    """A simulation (or the load that precedes it) failed."""


@dataclass(frozen=True)
class SimResult:
    """The outcome of one :meth:`Diagram.run`.

    Attributes:
        time: the output time grid (1-D ndarray).
        signals: ``{name: 1-D ndarray}``, one entry per Scope channel, aligned
            with ``time``. Names follow the Scope's ``labels`` (default
            ``<scope>-<j>``), exactly as the CLI's CSV columns.
        engine: ``"compiled"`` (scipy ``solve_ivp`` fast path) or
            ``"interpreter"`` (fixed-step block-by-block loop).
        diagnostics: solver statistics of a compiled run (method, backend,
            nfev, events, stiffness hint, wall times; see
            ``lib/engine/solver_diagnostics.py``). Empty for an interpreter run.
        fallback_reason: why a run asked to compile went to the interpreter
            instead, or ``None``.
    """

    time: np.ndarray
    signals: Dict[str, np.ndarray]
    engine: str
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    fallback_reason: Optional[str] = None

    def __getitem__(self, name: str) -> np.ndarray:
        try:
            return self.signals[name]
        except KeyError:
            raise KeyError(f"No signal {name!r}; available: {sorted(self.signals)}") from None

    @property
    def names(self) -> List[str]:
        """The signal names, in Scope order."""
        return list(self.signals)

    def to_dict(self) -> Dict[str, Any]:
        """``{"timeline", "signals"}``, the shape ``lib.cli.write_output`` takes."""
        return {"timeline": self.time, "signals": dict(self.signals)}

    def save(self, path: str, fmt: Optional[str] = None) -> int:
        """Write the signals to CSV or NPZ (by extension, or ``fmt``).

        Returns the number of signal columns written.
        """
        return cli.write_output(self.to_dict(), path, fmt)


class Diagram:
    """A loaded diagram, ready to simulate headlessly.

    Create one with :func:`load` or :func:`from_builder`. Blocks are addressed
    by name; a block inside a Subsystem by its qualified ``"<Subsystem>/<block>"``
    name (see :attr:`blocks`).
    """

    def __init__(self, dsim, sim_params: Dict[str, Any]):
        self._dsim = dsim
        self._sim_params = dict(sim_params or {})

    # ----------------------------------------------------------- inspection

    @property
    def dsim(self):
        """The underlying ``lib.lib.DSim`` (escape hatch; not a stable API)."""
        return self._dsim

    @property
    def sim_time(self) -> float:
        """The diagram's saved duration, the default for :meth:`run`."""
        return float(self._sim_params.get("sim_time", 10.0))

    @property
    def sim_dt(self) -> float:
        """The diagram's saved step, the default for :meth:`run`."""
        return float(self._sim_params.get("sim_dt", 0.01))

    @property
    def blocks(self) -> List[str]:
        """Every block name, Subsystem contents included as ``"Sub/block"``."""
        return [name for name, _ in iter_blocks_qualified(root_blocks_of(self._dsim))]

    def block_type(self, block: str) -> str:
        """The block's type (``block_fn``), e.g. ``"Gain"``."""
        return self._block(block).block_fn

    def params(self, block: str) -> Dict[str, Any]:
        """A copy of the block's user parameters (internal ``_x_`` keys omitted)."""
        return {k: v for k, v in self._block(block).params.items() if not k.startswith("_")}

    def get_param(self, block: str, param: str) -> Any:
        """The current value of one block parameter."""
        b = self._block(block)
        self._check_param(b, block, param)
        return b.params[param]

    def set_param(self, block: str, param: str, value: Any) -> None:
        """Set one block parameter for every later run.

        ``value`` may be a number, an array/list, or an expression string
        (resolved against the workspace at run time, as in the GUI).
        """
        b = self._block(block)
        self._check_param(b, block, param)
        b.params[param] = value
        if getattr(b, "exec_params", None):
            b.exec_params[param] = value

    @property
    def workspace(self):
        """This diagram's workspace variables (a ``lib.workspace.Workspace``).

        Parameters that name a variable (``gain = "K"``) or use one in an
        expression resolve against it at run time. It is private to this
        diagram: another :class:`Diagram` in the same process has its own.
        """
        return self._dsim.workspace

    @property
    def variables(self) -> Dict[str, Any]:
        """A copy of the workspace variables, ``{name: value}``."""
        return dict(self._dsim.workspace.variables)

    def set_variable(self, name: str, value: Any) -> None:
        """Set one workspace variable of this diagram for every later run."""
        self._dsim.workspace.set_variable(name, value)

    def save(self, path: str) -> None:
        """Save the diagram, parameter changes included, as a ``.diablos`` file."""
        fs = self._dsim.file_service
        data = fs.serialize(sim_params=self._sim_params)
        if not fs.save_to_file(data, path):
            raise SimulationError(f"Failed to save diagram: {path}")

    # ----------------------------------------------------------- simulation

    def run(
        self,
        t: Optional[float] = None,
        dt: Optional[float] = None,
        *,
        params: Optional[Dict[Any, Any]] = None,
        solver: str = "compiled",
        method: Optional[str] = None,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
        zero_crossing: Optional[bool] = None,
    ) -> SimResult:
        """Simulate the diagram and return its Scope signals.

        Args:
            t: duration in seconds (default: the diagram's saved ``sim_time``).
            dt: output step / interpreter step (default: the saved ``sim_dt``).
            params: optional ``{(block, param): value}`` overrides for this run
                only; the diagram is restored afterwards.
            solver: ``"compiled"`` (default; falls back to the interpreter when
                the diagram cannot compile, see ``SimResult.fallback_reason``)
                or ``"interpreter"``.
            method, rtol, atol, zero_crossing: compiled-solver settings; each
                defaults to the diagram's saved value. ``method`` is a
                ``solve_ivp`` name (``"RK45"``, ``"Radau"``, ...) or one of the
                DiaBloS settings (``"auto"``, ``"Euler"``, ``"RK4"``).

        Raises:
            SimulationError: the run failed (the message says why).
        """
        if solver not in SOLVERS:
            raise ValueError(f"solver must be one of {SOLVERS}, got {solver!r}")
        t, dt = self._horizon(t, dt)
        dsim = self._dsim
        dsim.use_fast_solver = solver == "compiled"
        cli.apply_solver_settings(
            dsim,
            self._sim_params,
            solver_method=method,
            rtol=rtol,
            atol=atol,
            zero_crossing=zero_crossing,
        )

        with self._overrides(params):
            ok, err = dsim.run_tuning_simulation(t, dt)
            if not ok:
                raise SimulationError(err or "Simulation failed")
            harvest = harvest_scope_signals(dsim)

        diagnostics = dsim.engine.get_solver_diagnostics()
        compiled = bool(diagnostics.get("success"))
        fallback = None
        if solver == "compiled" and not compiled:
            # A failed compiled attempt records why; a diagram that never
            # compiled leaves the compilability check's reason instead.
            fallback = diagnostics.get("message") or dsim.engine.get_compile_fallback_reason()
        if harvest is None:
            time, signals = np.zeros(0), {}
        else:
            time = np.array(harvest["timeline"], dtype=float)
            signals = {k: np.array(v, dtype=float) for k, v in harvest["signals"].items()}
        return SimResult(
            time=time,
            signals=signals,
            engine="compiled" if compiled else "interpreter",
            diagnostics=diagnostics if compiled else {},
            fallback_reason=fallback,
        )

    # ------------------------------------------------------------- analysis

    def monte_carlo(
        self,
        n_runs: int,
        seed: int = 12345,
        samplers: Optional[Dict[Any, Any]] = None,
        t: Optional[float] = None,
        dt: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run a seeded Monte-Carlo ensemble; see ``MonteCarloRunner.run``.

        Every block with a ``seed`` parameter gets a per-run sub-seed derived
        from ``seed``, so the ensemble is reproducible. ``samplers`` maps
        ``(block, param)`` to ``(low, high)`` (uniform draw) or ``f(rng)``.
        Returns the ensemble dict: ``timeline`` plus, per signal, ``runs``,
        ``mean``, ``std``, ``p5``/``p50``/``p95``, ``min``/``max`` and per-run
        ``metrics``. The diagram's parameters are restored afterwards.
        """
        from lib.analysis.monte_carlo import MonteCarloRunner

        t, dt = self._prepare_batch(t, dt)
        return MonteCarloRunner(self._dsim).run(
            int(n_runs), master_seed=seed, sim_time=t, sim_dt=dt, samplers=samplers
        )

    def sweep(
        self,
        axes: Iterable[Dict[str, Any]],
        t: Optional[float] = None,
        dt: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run a 1-D or 2-D parameter sweep; see ``ParameterSweepRunner.run``.

        ``axes`` is one or two ``{"block": name, "param": pname, "values": [...]}``
        dicts. Returns the sweep dict (1-D: a response family per signal plus
        per-value ``metrics``; 2-D: a metric grid). The diagram's parameters
        are restored afterwards.
        """
        from lib.analysis.parameter_sweep import ParameterSweepRunner

        t, dt = self._prepare_batch(t, dt)
        return ParameterSweepRunner(self._dsim).run(list(axes), sim_time=t, sim_dt=dt)

    def find_operating_point(
        self, t: float = 0.0, input_overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Solve for an equilibrium (dx/dt = 0) of the compiled diagram.

        Returns ``success``, ``y``, ``residual``, ``message``, ``state_names``
        and ``operating_point`` (ready for :meth:`linearize`). Raises
        ValueError when the diagram does not compile.
        """
        return self._linearizer().find_operating_point(t=t, input_overrides=input_overrides)

    def linearize(
        self,
        inputs: Optional[List[str]] = None,
        outputs: Optional[List[str]] = None,
        operating_point: Optional[Dict[str, Any]] = None,
        t: float = 0.0,
        eps: float = 1e-6,
    ) -> Dict[str, Any]:
        """Linearize the compiled diagram by finite differences.

        Always returns ``A`` (plus ``eigenvalues``, ``stable``, ``state_names``);
        with ``inputs`` (source-block names) and ``outputs`` (block names) also
        ``B``, ``C``, ``D`` and ``transfer_function``. Raises ValueError when
        the diagram does not compile or has no continuous states.
        """
        res = self._linearizer().linearize_at_point(
            operating_point=operating_point,
            input_blocks=inputs,
            output_blocks=outputs,
            t=t,
            eps=eps,
        )
        if res is None:
            raise ValueError("Diagram has no continuous states; nothing to linearize")
        return res

    # ------------------------------------------------------------ internals

    def _block(self, name: str):
        for qn, b in iter_blocks_qualified(root_blocks_of(self._dsim)):
            if qn == name:
                return b
        raise KeyError(f"No block named {name!r}; blocks: {self.blocks}")

    @staticmethod
    def _check_param(b, block: str, param: str) -> None:
        if not isinstance(getattr(b, "params", None), dict) or param not in b.params:
            raise KeyError(f"Block {block!r} has no parameter {param!r}")

    def _horizon(self, t, dt):
        t = self.sim_time if t is None else float(t)
        dt = self.sim_dt if dt is None else float(dt)
        if not (t > 0 and dt > 0):
            raise ValueError(f"t and dt must be positive, got t={t}, dt={dt}")
        return t, dt

    def _prepare_batch(self, t, dt):
        """Compiled engine and the diagram's own solver settings, for the runners."""
        self._dsim.use_fast_solver = True
        cli.apply_solver_settings(self._dsim, self._sim_params)
        return self._horizon(t, dt)

    def _linearizer(self):
        from lib.analysis.linearizer import Linearizer

        return Linearizer(self._dsim)

    @contextmanager
    def _overrides(self, params):
        """Apply ``{(block, param): value}`` for the duration of the block."""
        saved = []
        try:
            for (block, param), value in (params or {}).items():
                saved.append((block, param, self.get_param(block, param)))
                self.set_param(block, param, value)
            yield
        finally:
            for block, param, value in reversed(saved):
                self.set_param(block, param, value)


def load(path: str) -> Diagram:
    """Load a ``.diablos`` (or legacy ``.dat``/``.json``) diagram headlessly.

    Raises FileNotFoundError if ``path`` is missing and SimulationError if it
    will not load.
    """
    try:
        dsim, sim_params = cli.load_diagram(path)
    except FileNotFoundError:
        raise
    except RuntimeError as exc:
        raise SimulationError(str(exc)) from exc
    return Diagram(dsim, sim_params)


def from_builder(builder) -> Diagram:
    """A :class:`Diagram` from a ``lib.diagram_builder.DiagramBuilder``, no file needed."""
    # A JSON round trip gives the loader exactly what it would read from disk
    # and keeps the builder's own dicts out of the live diagram.
    data = json.loads(json.dumps(builder.to_dict()))
    dsim = cli.new_headless_dsim()
    _complete_builder_blocks(dsim, data["blocks_data"])
    sim_params = dsim.file_service.apply_loaded_data(data) or {}
    return Diagram(dsim, sim_params)


def _complete_builder_blocks(dsim, blocks_data) -> None:
    """Make builder block dicts look like what the GUI saves.

    The builder records only the parameters the caller passed, while a saved
    diagram carries every parameter, so fill the rest from the block type's
    defaults (an Integrator without ``init_conds`` fails at run time). Each
    block is also named by the name given to ``add_block``, so the same name
    addresses it in :meth:`Diagram.set_param` and in the results.
    """
    defaults = {mb.block_fn: mb.params for mb in dsim.model.menu_blocks}
    seen = set()
    for bd in blocks_data:
        name = bd["username"]
        if name in seen:
            raise ValueError(f"Duplicate block name {name!r} in the builder")
        seen.add(name)
        bd["name"] = name
        spec = defaults.get(bd["block_fn"])
        if spec is not None:
            bd["params"] = {**json.loads(json.dumps(spec, default=_jsonable)), **bd["params"]}


def _jsonable(value):
    """Copy numpy defaults through JSON (a default may be an ndarray)."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unserializable default {value!r}")
