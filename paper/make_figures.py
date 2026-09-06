"""Regenerate the figures used by ``paper/paper.md``.

Both figures come out of DiaBloS itself, so they cannot drift from what the
tool actually produces:

* ``figures/diagram.svg`` and ``figures/diagram.png`` -- the
  ``examples/library_block_demo.diablos`` diagram, rendered by the
  application's own diagram exporter (``modern_ui.tools.diagram_image_exporter``),
  i.e. the same paint pass the canvas uses on screen, in the light theme so it
  prints. The paper embeds the PNG because JOSS's Pandoc pipeline is only
  documented to accept raster and PDF figures; the SVG is the vector original.
* ``figures/response.png`` -- the closed-loop speed response, from a headless
  compiled-solver run of the same diagram via ``lib.cli.run_diagram`` and
  ``lib.analysis.resim.harvest_scope_signals``, plotted with the matplotlib
  Agg backend.

Usage (from the repository root)::

    python paper/make_figures.py
"""

import os
import sys

# Headless before anything imports Qt or matplotlib.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(REPO_ROOT, "paper", "figures")
DIAGRAM = os.path.join(REPO_ROOT, "examples", "library_block_demo.diablos")

sys.path.insert(0, REPO_ROOT)

# Process-lifetime reference to the QApplication. Without it PyQt tears the C++
# objects down while live QWidgets still exist (see the same guard in lib/cli.py).
_QAPP = None


def make_diagram_figures(svg_path, png_path):
    """Render the example diagram to SVG and PNG with DiaBloS's own exporter."""
    from PyQt5.QtWidgets import QApplication

    from lib.lib import DSim
    from lib.theming.theme_manager import ThemeType, theme_manager
    from modern_ui.tools.diagram_image_exporter import (
        render_diagram_image,
        render_diagram_svg,
    )
    from modern_ui.widgets.modern_canvas import ModernCanvas

    global _QAPP
    _QAPP = QApplication.instance() or QApplication(["diablos-figures"])
    theme_manager.set_theme(ThemeType.LIGHT)

    dsim = DSim()
    data = dsim.file_service.load(filepath=DIAGRAM)
    if data is None:
        raise RuntimeError("could not load %s" % DIAGRAM)
    dsim.file_service.apply_loaded_data(data)

    canvas = ModernCanvas(dsim)
    canvas.resize(1200, 700)
    # Build every wire's painter path, as the canvas does on first paint.
    canvas.connection_manager.update_line_positions()

    if not render_diagram_svg(canvas, svg_path):
        raise RuntimeError("empty diagram -- nothing rendered")

    image = render_diagram_image(canvas, scale=3)
    if image is None or not image.save(png_path):
        raise RuntimeError("could not write %s" % png_path)
    return [svg_path, png_path]


def _step_reference(dsim):
    """The Step block's commanded value, so the plot can show the setpoint."""
    for block in dsim.blocks_list:
        if getattr(block, "block_fn", None) == "Step":
            try:
                return float(block.params["value"])
            except (KeyError, TypeError, ValueError):
                return None
    return None


def make_response_png(out_path):
    """Simulate the example headlessly and plot its Scope traces."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from lib.analysis.resim import harvest_scope_signals
    from lib.cli import run_diagram

    dsim = run_diagram(DIAGRAM)
    result = harvest_scope_signals(dsim)
    timeline = result["timeline"]
    signals = result["signals"]

    fig, ax = plt.subplots(figsize=(6.0, 3.0), dpi=160)
    reference = _step_reference(dsim)
    if reference is not None:
        ax.axhline(reference, color="0.45", ls="--", lw=1.2, label="reference")
    for label, values in signals.items():
        ax.plot(timeline[: len(values)], values[: len(timeline)], lw=1.8, label=label)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("speed [m/s]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    written = make_diagram_figures(
        os.path.join(FIG_DIR, "diagram.svg"),
        os.path.join(FIG_DIR, "diagram.png"),
    )
    written.append(make_response_png(os.path.join(FIG_DIR, "response.png")))
    for path in written:
        print("%s (%.1f KB)" % (path, os.path.getsize(path) / 1024.0))


if __name__ == "__main__":
    main()
