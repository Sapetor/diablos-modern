from blocks.analysis_marker_base import AnalysisMarkerBlock


class NyquistBlock(AnalysisMarkerBlock):
    @property
    def block_name(self):
        return "Nyquist"

    @property
    def doc(self):
        return "Right-click to generate a Nyquist plot from a connected dynamic block."

    def _trace_icon(self, path):
        """Draw a contour spiralling in around the real/imaginary cross."""
        # Axes (small cross)
        path.moveTo(0.2, 0.5)
        path.lineTo(0.8, 0.5)  # Real axis
        path.moveTo(0.5, 0.2)
        path.lineTo(0.5, 0.8)  # Imaginary axis

        # Contour: start near +infinity (right) and spiral in
        path.moveTo(0.8, 0.4)
        path.cubicTo(0.8, 0.9, 0.3, 0.9, 0.3, 0.5)  # Bottom loop
        path.cubicTo(0.3, 0.2, 0.6, 0.2, 0.6, 0.4)  # Top loop, spiralling in
