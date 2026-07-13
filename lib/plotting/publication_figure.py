"""
Publication-quality figure building for scope data.

Qt-free by design: uses :class:`matplotlib.figure.Figure` directly (no
pyplot), so no backend window is opened, no global pyplot state is touched,
and the module is safe under headless backends (MPLBACKEND=Agg). It must
never import Qt or modern_ui.
"""

import numpy as np
from matplotlib.figure import Figure

_FONT_FAMILY = "serif"


def _serif_ticks(ax):
    """Render tick labels in the serif family."""
    try:
        ax.tick_params(labelfontfamily=_FONT_FAMILY)
    except (TypeError, ValueError):  # matplotlib < 3.9
        for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            label.set_fontfamily(_FONT_FAMILY)


def build_publication_figure(t, traces, options=None):
    """Build a publication-style matplotlib Figure from scope traces.

    :param t: 1D time array (seconds).
    :param traces: list of dicts ``{"name": str, "y": 1D array, "step": bool}``.
        ``(N, 1)`` arrays are flattened; length mismatches between ``t`` and
        ``y`` are resolved by truncating to the shorter of the two.
    :param options: optional dict of figure options: ``"title"`` (str),
        ``"figsize"`` (tuple of inches), ``"dpi"`` (int). This dict is the
        seam for future layout options (e.g. per-trace subplots).
    :returns: :class:`matplotlib.figure.Figure` with a single axes holding
        all traces (step traces use ``where='post'``), serif fonts, a legend,
        and a light grid. The caller saves it via ``fig.savefig(path)``.
    """
    options = options or {}
    t = np.asarray(t, dtype=float)

    fig = Figure(figsize=options.get("figsize", (8.0, 4.5)), dpi=options.get("dpi", 100))
    ax = fig.add_subplot(111)

    for trace in traces:
        y = np.asarray(trace.get("y"), dtype=float)
        if y.ndim > 1:
            y = y.ravel()
        n = min(len(t), len(y))
        name = str(trace.get("name", ""))
        if trace.get("step"):
            ax.step(t[:n], y[:n], where="post", label=name, linewidth=1.5)
        else:
            ax.plot(t[:n], y[:n], label=name, linewidth=1.5)

    ax.set_xlabel("Time (s)", fontfamily=_FONT_FAMILY)
    title = options.get("title")
    if title:
        ax.set_title(title, fontfamily=_FONT_FAMILY)
    ax.grid(True, alpha=0.3)
    if traces:
        ax.legend(prop={"family": _FONT_FAMILY})
    _serif_ticks(ax)
    fig.tight_layout()
    return fig
