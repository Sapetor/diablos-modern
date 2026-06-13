"""1-D / 2-D parameter sweep over a diagram on the headless re-sim path.

Re-runs a diagram across a grid of one or two block-parameter values, harvests
each run's Scope traces, and reduces them so the result window can show:

  * **1-D** -- the *response family*: one trace per swept value (overlay), plus a
    per-value scalar metric (final / peak / rms / ...) for a metric-vs-parameter
    plot.
  * **2-D** -- a *heatmap* of a per-run scalar metric over the (x, y) grid.

This is the deterministic sibling of :class:`lib.analysis.monte_carlo.MonteCarloRunner`
and shares its re-sim plumbing via :mod:`lib.analysis.resim`
(:func:`harvest_scope_signals`, :data:`OUTCOME_METRICS`). Original parameter
values are restored afterwards so the runner never mutates the user's diagram.

Result-dict contract::

    {
      "mode": "1d" | "2d",
      "n_points": int, "n_ok": int,
      "timeline": ndarray(L) | None,
      # 1-D:
      "axis":   {"block": str, "param": str, "values": ndarray(Nx)},
      # 2-D:
      "axis_x": {"block": str, "param": str, "values": ndarray(Nx)},
      "axis_y": {"block": str, "param": str, "values": ndarray(Ny)},
      "signals": {
        signal_name: {
          "traces":  ndarray(Nx, L),                 # 1-D only (response family)
          "metrics": {name: ndarray(Nx)},            # 1-D
                   | {name: ndarray(Nx, Ny)},        # 2-D
        }
      }
    }

Failed runs leave NaN rows (1-D) / NaN cells (2-D). ``signals`` is empty when no
run succeeded.
"""

import logging
import numpy as np

from lib.analysis.resim import OUTCOME_METRICS, harvest_scope_signals

logger = logging.getLogger(__name__)


class ParameterSweepRunner:
    """Run a 1-D/2-D parameter sweep of ``dsim`` and aggregate Scope statistics."""

    def __init__(self, dsim):
        self.dsim = dsim

    def run(self, axes, sim_time=None, sim_dt=None, progress_cb=None, cancel_cb=None):
        """Sweep one or two parameters across their value grids.

        Args:
            axes: list of 1 or 2 dicts ``{"block": name, "param": pname,
                "values": array-like}``. One axis -> 1-D sweep; two -> 2-D grid.
            sim_time, sim_dt: overrides (default: the diagram's current values).
            progress_cb: optional callable(done, total).
            cancel_cb: optional callable() -> bool, polled before each grid point;
                when it returns True the sweep stops early and the partial grid
                gathered so far is aggregated and returned (diagram still restored).

        Returns:
            A sweep-result dict (see module docstring).
        """
        dsim = self.dsim
        if not axes or len(axes) not in (1, 2):
            raise ValueError("axes must contain 1 or 2 entries")
        sim_time = float(sim_time if sim_time is not None else getattr(dsim, 'sim_time', 1.0))
        sim_dt = float(sim_dt if sim_dt is not None else getattr(dsim, 'sim_dt', 0.01))
        blocks_by_name = {b.name: b for b in dsim.blocks_list}

        # Resolve + validate each axis; snapshot originals so we restore exactly.
        resolved = []
        original = {}
        for ax in axes:
            bn, pn = ax.get('block'), ax.get('param')
            b = blocks_by_name.get(bn)
            if b is None:
                raise ValueError(f"No block named {bn!r}")
            if not isinstance(getattr(b, 'params', None), dict) or pn not in b.params:
                raise ValueError(f"Block {bn!r} has no parameter {pn!r}")
            values = np.asarray(ax.get('values'), dtype=float).ravel()
            if values.size == 0:
                raise ValueError(f"Axis ({bn}, {pn}) has no values")
            resolved.append((b, bn, pn, values))
            original[(bn, pn)] = b.params.get(pn)

        def _set(block, name, value):
            block.params[name] = value
            if getattr(block, 'exec_params', None):
                block.exec_params[name] = value

        mode = '1d' if len(resolved) == 1 else '2d'
        if mode == '1d':
            nx = resolved[0][3].size
            grid = [(i,) for i in range(nx)]
            shape = (nx,)
        else:
            nx, ny = resolved[0][3].size, resolved[1][3].size
            grid = [(i, j) for i in range(nx) for j in range(ny)]
            shape = (nx, ny)
        n_points = len(grid)

        harvests = [None] * n_points
        try:
            for k, idx in enumerate(grid):
                if cancel_cb is not None and cancel_cb():
                    logger.info("Parameter sweep cancelled after %d/%d points", k, n_points)
                    break
                for axi, (b, _bn, pn, values) in enumerate(resolved):
                    _set(b, pn, float(values[idx[axi]]))
                ok, err = dsim.run_tuning_simulation(sim_time, sim_dt)
                if ok:
                    harvests[k] = harvest_scope_signals(dsim)
                else:
                    logger.warning("Sweep point %s failed: %s", idx, err)
                if progress_cb:
                    progress_cb(k + 1, n_points)
        finally:
            for (bn, pn), val in original.items():
                b = blocks_by_name.get(bn)
                if b is None:
                    continue
                if val is None:
                    b.params.pop(pn, None)
                else:
                    b.params[pn] = val
                # Keep exec_params in sync with params (restore the original value
                # rather than dropping the key) so the diagram is left untouched.
                if getattr(b, 'exec_params', None) is not None:
                    if val is None:
                        b.exec_params.pop(pn, None)
                    else:
                        b.exec_params[pn] = val

        return self._aggregate(mode, resolved, shape, harvests)

    @staticmethod
    def _aggregate(mode, resolved, shape, harvests):
        """Reduce per-point harvests into the sweep-result dict (see module doc)."""
        n_points = len(harvests)
        result = {
            'mode': mode,
            'n_points': n_points,
            'n_ok': sum(1 for h in harvests if h and h.get('signals')),
            'timeline': None,
            'signals': {},
        }
        if mode == '1d':
            _b, bn, pn, xv = resolved[0]
            result['axis'] = {'block': bn, 'param': pn, 'values': xv}
        else:
            result['axis_x'] = {'block': resolved[0][1], 'param': resolved[0][2],
                                'values': resolved[0][3]}
            result['axis_y'] = {'block': resolved[1][1], 'param': resolved[1][2],
                                'values': resolved[1][3]}

        ok = [(k, h) for k, h in enumerate(harvests) if h and h.get('signals')]
        if not ok:
            return result

        # Signals present in every successful run, truncated to a common length.
        names = set(ok[0][1]['signals'])
        for _k, h in ok[1:]:
            names &= set(h['signals'])
        if not names:
            return result
        lengths = [len(h['timeline']) for _k, h in ok]
        lengths += [len(h['signals'][nm]) for _k, h in ok for nm in names]
        L = min(lengths)
        if L == 0:
            return result

        result['timeline'] = np.asarray(ok[0][1]['timeline'][:L], dtype=float)
        for nm in sorted(names):
            M = np.full((n_points, L), np.nan)
            for k, h in ok:
                M[k] = h['signals'][nm][:L]
            with np.errstate(all='ignore'):
                metrics = {m: fn(M) for m, fn in OUTCOME_METRICS.items()}
            if mode == '2d':
                result['signals'][nm] = {
                    'metrics': {m: v.reshape(shape) for m, v in metrics.items()},
                }
            else:
                result['signals'][nm] = {'traces': M, 'metrics': metrics}
        return result
