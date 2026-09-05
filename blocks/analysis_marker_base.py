"""Shared base for the analysis *marker* blocks.

BodeMag, BodePhase, Nyquist, RootLocus and LQR carry no simulation semantics.
They are dropped on the canvas and wired to a dynamic block so the right-click
Analysis menu knows what to linearize; the plot (or the Riccati solve) happens
entirely outside ``execute()``. Every one of them therefore declares the same
thing: a purple, single-input, zero-output block whose ``execute()`` is inert,
plus a ``draw_icon`` that is nothing but the same three lines of Qt boilerplate
around a different set of path segments.

Subclasses supply ``block_name``, ``doc`` and ``_trace_icon``; anything else
(``category`` for LQR, ``params`` for LQR and RootLocus, the port list and the
validation flags for LQR) is overridden only where it genuinely differs.
"""

from blocks.base_block import BaseBlock


class AnalysisMarkerBlock(BaseBlock):
    """Abstract base for right-click-driven analysis/design marker blocks."""

    @property
    def category(self):
        return "Analysis"

    @property
    def color(self):
        return "purple"

    @property
    def params(self):
        return {
            "_init_start_": {"default": True, "type": "bool"},
        }

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        # Declaring no output ports (rather than overriding requires_outputs)
        # is what keeps DiagramValidator from warning about a dangling output.
        return []

    def draw_icon(self, block_rect):
        """Build the marker's icon path in 0-1 normalized coordinates."""
        from PyQt5.QtGui import QPainterPath

        path = QPainterPath()
        self._trace_icon(path)
        return path

    def _trace_icon(self, path):
        """Append this marker's segments to ``path``. Implemented per subclass."""
        raise NotImplementedError

    def execute(self, time, inputs, params, **kwargs):
        """Inert during simulation: no outputs, no error, no state in params."""
        return {}
