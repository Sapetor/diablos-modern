from blocks.analysis_marker_base import AnalysisMarkerBlock


class BodePhaseBlock(AnalysisMarkerBlock):
    @property
    def block_name(self):
        return "BodePhase"

    @property
    def doc(self):
        return "Right-click to generate a Bode Phase plot from a connected dynamic block."

    def _trace_icon(self, path):
        """Draw an S-curve phase roll-off beside an L-shaped axis pair."""
        # Axes (L-shape for plot)
        path.moveTo(0.15, 0.15)
        path.lineTo(0.15, 0.85)
        path.lineTo(0.85, 0.85)

        # Phase curve (high to low transition)
        path.moveTo(0.25, 0.25)
        path.cubicTo(0.45, 0.25, 0.55, 0.75, 0.75, 0.75)
