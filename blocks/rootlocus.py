from blocks.analysis_marker_base import AnalysisMarkerBlock


class RootLocusBlock(AnalysisMarkerBlock):
    @property
    def block_name(self):
        return "RootLocus"

    @property
    def params(self):
        # Stateless analysis block: execute() is a no-op, so no persistent
        # per-step state (and hence no _init_start_ flag) is required.
        return {}

    @property
    def doc(self):
        return (
            "Root Locus Plotter."
            "\n\nAnalyzes the closed-loop poles of a system as a parameter varies (typically gain K)."
            "\n\nFeatures:"
            "\n- Connect to a Transfer Function or State Space block to define the system."
            "\n- Right-click the block and select 'Analysis > Root Locus' to generate the plot."
            "\n- Shows pole trajectories and stability boundaries."
        )

    def _trace_icon(self, path):
        """Draw two locus branches leaving a pole marked with an 'x'."""
        # Axes centered
        path.moveTo(0.5, 0.1)
        path.lineTo(0.5, 0.9)  # Imaginary axis
        path.moveTo(0.1, 0.5)
        path.lineTo(0.9, 0.5)  # Real axis

        # Branches
        path.moveTo(0.3, 0.5)  # Pole on left (stable)
        path.quadTo(0.3, 0.3, 0.5, 0.2)  # Branch going to zero/asymptote

        path.moveTo(0.3, 0.5)
        path.quadTo(0.3, 0.7, 0.5, 0.8)  # Mirror branch

        # 'x' for the pole at (0.3, 0.5)
        path.moveTo(0.28, 0.48)
        path.lineTo(0.32, 0.52)
        path.moveTo(0.32, 0.48)
        path.lineTo(0.28, 0.52)
