from blocks.analysis_marker_base import AnalysisMarkerBlock


class BodeMagBlock(AnalysisMarkerBlock):
    @property
    def block_name(self):
        return "BodeMag"

    @property
    def doc(self):
        return "Right-click to generate a Bode magnitude plot from a connected Transfer Function block."

    def _trace_icon(self, path):
        """Draw a magnitude asymptote rolling off over log axes."""
        # Axes
        path.moveTo(0.1, 0.9)
        path.lineTo(0.9, 0.9)
        path.moveTo(0.1, 0.9)
        path.lineTo(0.1, 0.1)
        # Magnitude asymptote with a single break
        path.moveTo(0.1, 0.4)
        path.lineTo(0.4, 0.4)
        path.lineTo(0.6, 0.7)
        path.lineTo(0.9, 0.7)
