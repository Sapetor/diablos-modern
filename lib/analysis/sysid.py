"""Fit a continuous transfer function to recorded input/output data.

``identify_tf(t, u, y, num_order, den_order)`` returns ``G(s) = B(s)/A(s)``
(``A`` monic) of the requested orders, in three steps:

1. Resample onto a uniform grid (the median sample spacing, at most
   ``MAX_GRID`` points).
2. ARX least squares, ``y[k] + a1 y[k-1] + ... = b1 u[k-1] + ...``, converted
   to continuous time by inverting the zero-order-hold discretization
   (matrix logarithm of the augmented state matrix). ARX is ill-conditioned
   when sampling is fast, so it is tried at several decimations and the
   conversion that simulates closest to the data is the starting point.
3. Output-error refinement: the continuous coefficients are adjusted so the
   *simulated* model response matches ``y`` (``scipy.optimize.least_squares``
   over ``scipy.signal.lsim``). ARX alone is biased when the output is noisy;
   it is only the starting point. The refinement is kept only if it is stable
   and fits better.

The model starts at rest. A record that begins at an operating point is
handled by shifting input and output by their first samples; one that begins
at rest with the input already applied by shifting the output only. Both are
fitted and the better one is kept (``u_offset`` in the result).

The fit quality is the normalized-RMSE percentage used by MATLAB's
``compare``: ``100 * (1 - ||y - y_model|| / ||y - mean(y)||)``.
"""

from __future__ import annotations

import contextlib
import warnings

import numpy as np


MAX_GRID = 5000


class SysIdError(ValueError):
    """Data or orders that cannot be identified."""


def fit_percent(y, y_model) -> float:
    y = np.asarray(y, dtype=float)
    spread = np.linalg.norm(y - y.mean())
    if spread == 0:
        return 100.0 if np.allclose(y, y_model) else 0.0
    return float(100.0 * (1.0 - np.linalg.norm(y - np.asarray(y_model)) / spread))


@contextlib.contextmanager
def _quiet():
    """Silence scipy's BadCoefficients: leading numerator terms of ~1e-14 are
    expected here (ZOH zeros, round-off) and are trimmed by ``_shape``."""
    from scipy import signal

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", signal.BadCoefficients)
        yield


def _uniform(t, u, y):
    t = np.asarray(t, dtype=float).ravel()
    u = np.asarray(u, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if not (t.size == u.size == y.size):
        raise SysIdError("time, input and output must have the same length")
    order = np.argsort(t, kind="stable")
    t, u, y = t[order], u[order], y[order]
    keep = np.concatenate([[True], np.diff(t) > 0])
    t, u, y = t[keep], u[keep], y[keep]
    if t.size < 10:
        raise SysIdError("need at least 10 distinct samples")
    if not (np.all(np.isfinite(u)) and np.all(np.isfinite(y))):
        raise SysIdError("data contains NaN or infinite values")
    # The refinement simulates the model on this grid many times; cap it so a
    # long, finely sampled log stays fast.
    dt = max(float(np.median(np.diff(t))), (t[-1] - t[0]) / (MAX_GRID - 1))
    grid = t[0] + dt * np.arange(int(np.floor((t[-1] - t[0]) / dt)) + 1)
    return grid, np.interp(grid, t, u), np.interp(grid, t, y), dt


def _arx(u, y, nb, na):
    """Discrete B(z)/A(z) (positive powers) from ARX least squares, delay 1."""
    start = max(na, nb)
    rows = [
        np.concatenate([-y[k - na : k][::-1], u[k - nb : k][::-1]]) for k in range(start, y.size)
    ]
    theta, *_ = np.linalg.lstsq(np.array(rows), y[start:], rcond=None)
    a, b = theta[:na], theta[na:]
    den = np.concatenate([[1.0], a])
    num = np.concatenate([[0.0], b, np.zeros(na - nb)]) if na >= nb else np.concatenate([[0.0], b])
    return num, den


def _d2c(num_z, den_z, dt):
    """Invert a zero-order-hold discretization (B(z)/A(z) -> B(s)/A(s))."""
    from scipy import linalg, signal

    with _quiet():
        ad, bd, cd, dd = signal.tf2ss(num_z, den_z)
    n, m = ad.shape[0], bd.shape[1]
    aug = np.zeros((n + m, n + m))
    aug[:n, :n], aug[:n, n:], aug[n:, n:] = ad, bd, np.eye(m)
    log = linalg.logm(aug) / dt
    if np.max(np.abs(np.imag(log))) > 1e-6 * max(1.0, np.max(np.abs(log))):
        raise SysIdError("discrete model has poles on the negative real axis")
    log = np.real(log)
    with _quiet():
        num, den = signal.ss2tf(log[:n, :n], log[:n, n:], cd, dd)
    return np.atleast_1d(np.squeeze(num)), np.atleast_1d(den)


def _simulate(num, den, u, grid):
    from scipy import signal

    with _quiet():
        _, y_model, _ = signal.lsim(signal.lti(num, den), u, grid)
    return np.asarray(y_model, dtype=float)


def _stable(den) -> bool:
    return bool(np.all(np.real(np.roots(den)) < 0)) if len(den) > 1 else True


def _shape(num, den, num_order, den_order):
    """Monic denominator of exactly ``den_order`` and numerator of ``num_order``."""
    den = np.asarray(den, dtype=float)
    num = np.asarray(num, dtype=float) / den[0]
    den = den / den[0]
    num = np.trim_zeros(num, "f") if np.any(num) else np.zeros(1)
    num = np.concatenate([np.zeros(max(0, num_order + 1 - num.size)), num])[-(num_order + 1) :]
    return num, den


def _fit(grid, uz, yz, dt, num_order, den_order, refine):
    """Best (num, den, fit) for zero-start data ``uz`` -> ``yz``."""
    # ARX is ill-conditioned when the sampling is fast relative to the
    # dynamics (poles crowd z = 1), so try it at several decimations and start
    # from whichever converted model simulates closest to the data.
    candidates = []
    step = 1
    while grid.size // step >= 8 * (den_order + 1):
        try:
            # A ZOH-discretized n-th order system has n numerator terms whatever
            # the continuous numerator order, so fit nb = den_order and truncate
            # to num_order after the conversion.
            nz, dz = _arx(uz[::step], yz[::step], den_order, den_order)
            n, d = _shape(*_d2c(nz, dz, dt * step), num_order, den_order)
            if np.all(np.isfinite(d)) and np.all(np.isfinite(n)) and _stable(d):
                candidates.append((n, d, fit_percent(yz, _simulate(n, d, uz, grid))))
        except (SysIdError, np.linalg.LinAlgError, ValueError):
            pass
        step *= 2
    if not candidates:
        # Fallback start: poles spread over the record length, DC gain from
        # the final samples.
        span = grid[-1] - grid[0]
        poles = -np.linspace(1.0, 3.0, den_order) * (5.0 / span)
        d = np.real(np.poly(poles))
        gain = (yz[-1] / uz[-1]) if abs(uz[-1]) > 1e-12 else 1.0
        n = np.concatenate([np.zeros(num_order), [gain * d[-1]]])
        candidates.append((n, d, fit_percent(yz, _simulate(n, d, uz, grid))))
    best = max(candidates, key=lambda c: c[2])

    if refine:
        from scipy.optimize import least_squares

        def residual(x):
            d = np.concatenate([[1.0], x[:den_order]])
            n = x[den_order:]
            try:
                return _simulate(n, d, uz, grid) - yz
            except Exception:
                return np.full(yz.size, 1e6)

        x0 = np.concatenate([best[1][1:], best[0]])
        try:
            sol = least_squares(residual, x0, x_scale="jac", max_nfev=200 * x0.size)
            d = np.concatenate([[1.0], sol.x[:den_order]])
            n = sol.x[den_order:]
            if _stable(d):
                fit = fit_percent(yz, _simulate(n, d, uz, grid))
                if fit > best[2]:
                    best = (n, d, fit)
        except Exception:  # noqa: BLE001 -- keep the ARX estimate
            pass

    return best


def identify_tf(t, u, y, num_order: int, den_order: int, refine: bool = True) -> dict:
    """Identify ``G(s)``. Returns ``num``, ``den``, ``fit_percent``, ``grid``,
    ``y_model`` (simulated on ``grid`` from the resampled input), and the
    ``u_offset``/``y_offset`` the model is relative to."""
    num_order, den_order = int(num_order), int(den_order)
    if den_order < 1:
        raise SysIdError("denominator order must be at least 1")
    if not 0 <= num_order <= den_order:
        raise SysIdError("numerator order must be between 0 and the denominator order")
    grid, ug, yg, dt = _uniform(t, u, y)
    if grid.size < 4 * (num_order + den_order + 2):
        raise SysIdError("not enough samples for the requested orders")

    # lsim starts the model at rest. A record either starts at an operating
    # point (equilibrium under the first input: shift u and y by their first
    # samples) or at rest with the input already applied (shift y only). Fit
    # both readings and keep the better one.
    y0 = yg[0]
    best = None
    for u0 in dict.fromkeys([float(ug[0]), 0.0]):
        num, den, fit = _fit(grid, ug - u0, yg - y0, dt, num_order, den_order, refine)
        if best is None or fit > best[2]:
            best = (num, den, fit, u0)
    num, den, _, u0 = best
    y_model = _simulate(num, den, ug - u0, grid) + y0
    return {
        "num": [float(v) for v in num],
        "den": [float(v) for v in den],
        "fit_percent": fit_percent(yg, y_model),
        "grid": grid,
        "y_model": y_model,
        "u_offset": float(u0),
        "y_offset": float(y0),
    }


def format_tf(num, den, digits=4) -> str:
    """``(2 s + 1) / (s^2 + 3 s + 2)`` for logs and tooltips."""

    def poly(c):
        n = len(c) - 1
        terms = []
        for i, v in enumerate(c):
            p = n - i
            if abs(v) < 10 ** (-digits - 2):
                continue
            mag = f"{abs(v):.{digits}g}"
            var = "" if p == 0 else ("s" if p == 1 else f"s^{p}")
            body = var if (mag == "1" and var) else (f"{mag} {var}".strip())
            terms.append(("-" if v < 0 else "+", body))
        if not terms:
            return "0"
        text = ("-" if terms[0][0] == "-" else "") + terms[0][1]
        for sign, body in terms[1:]:
            text += f" {sign} {body}"
        return text

    return f"({poly(num)}) / ({poly(den)})"
