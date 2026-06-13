"""
AnalysisController - headless orchestration for the "Linearize & Analyze" feature.

Wraps :class:`lib.analysis.linearizer.Linearizer` and produces the shared
result-dict contract consumed by the results window. This module is HEADLESS:
it imports no Qt. The two BaseAnalyzer frequency-domain helpers it reuses
(``_auto_frequency_range`` and ``_compute_stability_margins``) are imported
lazily inside :meth:`analyze` so that merely importing this controller never
pulls in PyQt5 / pyqtgraph.

Result-dict contract (see also the results window which consumes EXACTLY this):

    {
      "ok": bool, "error": str,
      "n_states": int,
      "state_names": [str], "input_names": [str], "output_names": [str],
      "A": [[float]], "B": [[float]]|[], "C": [[float]]|[], "D": [[float]]|[],
      "poles": [[re,im], ...], "zeros": [[re,im], ...],
      "is_stable": bool, "time_constants": [float],
      "oscillatory_modes": [{"omega_n","zeta","period"}],
      "gain_margin_db": float|None, "phase_margin_deg": float|None,
      "gain_crossover": float|None, "phase_crossover": float|None,
      "tf_num": [float]|None, "tf_den": [float]|None,
      "bode": {"w":[float], "mag_db":[float], "phase_deg":[float]}|None,
      "controllable": bool|None, "observable": bool|None,
      "operating_point": {block_name: value},
      "summary": str,
    }
"""

import logging

import numpy as np

from lib.analysis.linearizer import Linearizer

logger = logging.getLogger(__name__)


def _empty_result(error=""):
    """A fully-populated failure result so the consuming window never KeyErrors."""
    return {
        "ok": False,
        "error": error,
        "n_states": 0,
        "state_names": [],
        "input_names": [],
        "output_names": [],
        "A": [],
        "B": [],
        "C": [],
        "D": [],
        "poles": [],
        "zeros": [],
        "is_stable": False,
        "time_constants": [],
        "oscillatory_modes": [],
        "gain_margin_db": None,
        "phase_margin_deg": None,
        "gain_crossover": None,
        "phase_crossover": None,
        "tf_num": None,
        "tf_den": None,
        "bode": None,
        "controllable": None,
        "observable": None,
        "operating_point": {},
        "summary": "",
    }


class AnalysisController:
    """Headless orchestrator: linearize a diagram and assemble the result dict."""

    def __init__(self, dsim):
        self.dsim = dsim

    # ------------------------------------------------------------------ public

    def analyze(self, input_blocks=None, output_blocks=None, find_trim=False):
        """Linearize ``self.dsim`` and return the shared result-dict contract.

        Args:
            input_blocks: optional list of source-block names treated as inputs.
            output_blocks: optional list of block names whose signal is output.
            find_trim: if True, solve for an equilibrium first and linearize there.

        Returns:
            The result dict. On any failure (uncompilable diagram, bad I/O,
            internal error) returns ``{"ok": False, "error": <msg>, ...}`` rather
            than raising.
        """
        try:
            lin = Linearizer(self.dsim)

            operating_point = None
            trim_note = None
            if find_trim:
                trim = lin.find_operating_point()
                operating_point = trim.get("operating_point")
                if trim.get("success"):
                    trim_note = "Operating point found (trim succeeded)."
                else:
                    trim_note = (
                        "Trim solve did not converge; linearizing at the best "
                        f"estimate ({trim.get('message', 'no message')})."
                    )

            lin_res = lin.linearize_at_point(
                operating_point=operating_point,
                input_blocks=input_blocks,
                output_blocks=output_blocks,
            )
            if lin_res is None:
                return _empty_result(
                    "Diagram has no continuous states; nothing to linearize."
                )

            return self._assemble(lin_res, trim_note=trim_note)

        except ValueError as exc:
            # Uncompilable diagram / unknown I/O block: expected, surfaced cleanly.
            logger.info("AnalysisController: linearization unavailable: %s", exc)
            res = _empty_result(str(exc))
            res["summary"] = f"Analysis unavailable:\n{exc}"
            return res
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("AnalysisController: unexpected failure")
            res = _empty_result(f"Unexpected error: {exc}")
            res["summary"] = f"Analysis failed:\n{exc}"
            return res

    # ----------------------------------------------------------------- helpers

    def _assemble(self, lin_res, trim_note=None):
        """Convert a Linearizer result into the shared result-dict contract."""
        A = np.atleast_2d(np.asarray(lin_res["A"], dtype=float))
        n_states = int(lin_res.get("n_states", A.shape[0]))

        eigenvalues = lin_res.get("eigenvalues")
        if eigenvalues is None:
            eigenvalues = np.linalg.eigvals(A) if A.size else np.array([])
        poles = self._complex_to_pairs(eigenvalues)

        has_io = "B" in lin_res and "C" in lin_res and "D" in lin_res

        result = _empty_result()
        result["ok"] = True
        result["error"] = ""
        result["n_states"] = n_states
        result["state_names"] = list(lin_res.get("state_names", []))
        result["A"] = A.tolist()
        result["poles"] = poles
        result["is_stable"] = bool(lin_res.get("is_stable", False))
        result["time_constants"] = [float(t) for t in lin_res.get("time_constants", [])]
        result["oscillatory_modes"] = [
            {
                "omega_n": float(m.get("omega_n", 0.0)),
                "zeta": float(m.get("zeta", 0.0)),
                "period": float(m.get("period", float("inf"))),
            }
            for m in lin_res.get("oscillatory_modes", [])
        ]
        result["operating_point"] = self._operating_point_to_dict(
            lin_res.get("operating_point"), result["state_names"]
        )

        tf_num = tf_den = None
        if has_io:
            result["B"] = np.atleast_2d(np.asarray(lin_res["B"], dtype=float)).tolist()
            result["C"] = np.atleast_2d(np.asarray(lin_res["C"], dtype=float)).tolist()
            result["D"] = np.atleast_2d(np.asarray(lin_res["D"], dtype=float)).tolist()
            result["input_names"] = list(lin_res.get("input_names", []))
            result["output_names"] = list(lin_res.get("output_names", []))
            result["controllable"] = self._opt_bool(lin_res.get("controllable"))
            result["observable"] = self._opt_bool(lin_res.get("observable"))

            tf = lin_res.get("transfer_function") or {}
            tf_num = tf.get("num")
            tf_den = tf.get("den")

        # A SISO transfer function enables zeros, Bode and margins.
        if self._is_siso_tf(tf_num, tf_den):
            num = np.atleast_1d(np.asarray(tf_num, dtype=float)).flatten()
            den = np.atleast_1d(np.asarray(tf_den, dtype=float)).flatten()
            result["tf_num"] = num.tolist()
            result["tf_den"] = den.tolist()
            result["zeros"] = self._tf_zeros(num)
            bode, margins = self._bode_and_margins(num, den)
            result["bode"] = bode
            if margins is not None:
                result["gain_margin_db"] = margins.get("gain_margin_db")
                result["phase_margin_deg"] = margins.get("phase_margin_deg")
                result["gain_crossover"] = margins.get("gain_crossover_w")
                result["phase_crossover"] = margins.get("phase_crossover_w")

        result["summary"] = self._build_summary(result, lin_res, trim_note)
        return result

    # -- conversions ----------------------------------------------------------

    @staticmethod
    def _complex_to_pairs(values):
        """[[re, im], ...] for a 1-D array of (possibly complex) numbers."""
        arr = np.atleast_1d(np.asarray(values, dtype=complex))
        return [[float(np.real(v)), float(np.imag(v))] for v in arr]

    @staticmethod
    def _opt_bool(val):
        return None if val is None else bool(val)

    @staticmethod
    def _operating_point_to_dict(op, state_names):
        """Map the operating-point vector to {state_name: value}."""
        if op is None:
            return {}
        if isinstance(op, dict):
            out = {}
            for k, v in op.items():
                arr = np.atleast_1d(np.asarray(v, dtype=float))
                out[k] = float(arr[0]) if arr.size == 1 else arr.tolist()
            return out
        arr = np.atleast_1d(np.asarray(op, dtype=float)).flatten()
        names = state_names if len(state_names) == arr.size else [
            f"x{i}" for i in range(arr.size)
        ]
        return {nm: float(v) for nm, v in zip(names, arr)}

    @staticmethod
    def _is_siso_tf(num, den):
        """True only for a usable single-input single-output TF."""
        if num is None or den is None:
            return False
        num_a = np.asarray(num)
        # ss2tf returns a 2D num for multi-output systems; only SISO is 1-D.
        if num_a.ndim > 1 and num_a.shape[0] != 1:
            return False
        den_a = np.atleast_1d(np.asarray(den, dtype=float)).flatten()
        return den_a.size >= 1 and np.any(den_a)

    @staticmethod
    def _tf_zeros(num):
        """Finite zeros (roots of the numerator) as [[re, im], ...]."""
        num = np.atleast_1d(np.asarray(num, dtype=float)).flatten()
        if num.size <= 1 or not np.any(num):
            return []
        try:
            roots = np.roots(num)
        except (np.linalg.LinAlgError, ValueError):
            return []
        roots = roots[np.isfinite(roots)]
        return [[float(np.real(r)), float(np.imag(r))] for r in roots]

    @staticmethod
    def _bode_and_margins(num, den):
        """Compute a continuous Bode response and gain/phase margins.

        Reuses BaseAnalyzer's static helpers for the frequency grid and the
        margin extraction. Imported lazily to keep this module Qt-free at import.

        Returns (bode_dict, margins_dict). On failure (bode_dict, None) or
        (None, None).
        """
        from scipy import signal
        from lib.analysis.analyzers.base_analyzer import BaseAnalyzer

        try:
            w = BaseAnalyzer._auto_frequency_range(num, den)
            tf = signal.TransferFunction(num, den)
            w, mag_db, phase_deg = tf.bode(w=w)
            # Keep only finite samples so plotting / margins stay clean.
            finite = (
                np.isfinite(w) & np.isfinite(mag_db) & np.isfinite(phase_deg)
            )
            w = np.asarray(w)[finite]
            mag_db = np.asarray(mag_db)[finite]
            phase_deg = np.asarray(phase_deg)[finite]
        except Exception:
            logger.warning("Bode computation failed", exc_info=True)
            return None, None

        bode = {
            "w": [float(v) for v in w],
            "mag_db": [float(v) for v in mag_db],
            "phase_deg": [float(v) for v in phase_deg],
        }

        try:
            margins = BaseAnalyzer._compute_stability_margins(w, mag_db, phase_deg)
        except Exception:
            logger.warning("Stability-margin computation failed", exc_info=True)
            margins = None

        return bode, margins

    # -- summary --------------------------------------------------------------

    @staticmethod
    def _fmt_complex(v):
        re = float(np.real(v))
        im = float(np.imag(v))
        if abs(im) < 1e-9:
            return f"{re:.4g}"
        sign = "+" if im >= 0 else "-"
        return f"{re:.4g} {sign} {abs(im):.4g}j"

    def _build_summary(self, result, lin_res, trim_note):
        """Human-readable multi-line summary."""
        lines = []

        if trim_note:
            lines.append(trim_note)

        lines.append(f"States: {result['n_states']}")

        # Stability verdict.
        if result["is_stable"]:
            verdict = "STABLE (all poles in the open left-half plane)"
        elif lin_res.get("is_marginally_stable"):
            verdict = "MARGINALLY STABLE (poles on the imaginary axis)"
        else:
            verdict = "UNSTABLE (at least one pole in the right-half plane)"
        lines.append(f"Stability: {verdict}")

        dom = lin_res.get("dominant_pole")
        if dom is not None:
            lines.append(f"Dominant pole: {self._fmt_complex(dom)}")

        if result["poles"]:
            ev_str = ", ".join(
                self._fmt_complex(complex(re, im)) for re, im in result["poles"]
            )
            lines.append(f"Eigenvalues: {ev_str}")

        if result["time_constants"]:
            tcs = ", ".join(f"{t:.4g}s" for t in sorted(result["time_constants"]))
            lines.append(f"Time constants: {tcs}")

        if result["oscillatory_modes"]:
            for m in result["oscillatory_modes"]:
                lines.append(
                    f"Oscillatory mode: omega_n={m['omega_n']:.4g} rad/s, "
                    f"zeta={m['zeta']:.4g}, period={m['period']:.4g}s"
                )

        if result["controllable"] is not None:
            lines.append(
                f"Controllable: {'yes' if result['controllable'] else 'no'}"
            )
        if result["observable"] is not None:
            lines.append(
                f"Observable: {'yes' if result['observable'] else 'no'}"
            )

        if result["tf_num"] is not None and result["tf_den"] is not None:
            lines.append(
                f"Transfer function: num={self._fmt_poly(result['tf_num'])}, "
                f"den={self._fmt_poly(result['tf_den'])}"
            )

        gm = result["gain_margin_db"]
        pm = result["phase_margin_deg"]
        if gm is not None or pm is not None:
            gm_s = "inf" if gm == float("inf") else (
                "n/a" if gm is None else f"{gm:.4g} dB"
            )
            pm_s = "inf" if pm == float("inf") else (
                "n/a" if pm is None else f"{pm:.4g} deg"
            )
            lines.append(f"Gain margin: {gm_s}, Phase margin: {pm_s}")

        return "\n".join(lines)

    @staticmethod
    def _fmt_poly(coeffs):
        return "[" + ", ".join(f"{c:.4g}" for c in coeffs) + "]"
