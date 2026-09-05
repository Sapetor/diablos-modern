from blocks.analysis_marker_base import AnalysisMarkerBlock


class LQRBlock(AnalysisMarkerBlock):
    """LQR design block — computes optimal state-feedback gain K.

    This is an analysis/design tool, not a simulation block.
    Right-click → "Compute LQR Gain" to solve the continuous algebraic
    Riccati equation and display K, closed-loop eigenvalues, and cost.

    Parameters A, B, Q, R can be entered as matrices or workspace variable names.
    """

    @property
    def block_name(self):
        return "LQR"

    @property
    def category(self):
        return "Control"

    @property
    def doc(self):
        return (
            "LQR optimal state-feedback gain designer."
            "\n\nRight-click → Compute LQR Gain to solve:"
            "\n  min ∫(x'Qx + u'Ru) dt"
            "\n  subject to dx/dt = Ax + Bu"
            "\n\nConnect the input to a StateSpace block to read A, B automatically."
            "\nOtherwise, set A, B manually (matrices or workspace variables)."
            "\n\nOutputs:"
            "\n- K: optimal gain matrix (u = -Kx)"
            "\n- Closed-loop eigenvalues of (A - BK)"
            "\n- Optimal cost matrix P"
        )

    @property
    def params(self):
        return {
            "A": {
                "default": "[[0, 1], [0, 0]]",
                "type": "string",
                "doc": "State matrix (n×n). Matrix or workspace variable.",
            },
            "B": {
                "default": "[[0], [1]]",
                "type": "string",
                "doc": "Input matrix (n×m). Matrix or workspace variable.",
            },
            "Q": {
                "default": "[[1, 0], [0, 1]]",
                "type": "string",
                "doc": "State cost matrix (n×n, positive semidefinite). Matrix or workspace variable.",
            },
            "R": {
                "default": "[[1]]",
                "type": "string",
                "doc": "Input cost matrix (m×m, positive definite). Matrix or workspace variable.",
            },
        }

    @property
    def inputs(self):
        return [{"name": "plant", "type": "any"}]

    @property
    def optional_inputs(self):
        return [0]

    @property
    def requires_inputs(self):
        # A/B can be typed in by hand instead of read off a connected plant.
        return False

    @property
    def requires_outputs(self):
        return False

    def _trace_icon(self, path):
        """Draw a state-feedback loop: plant box closed through a K gain."""
        # Plant block
        path.moveTo(0.38, 0.10)
        path.lineTo(0.74, 0.10)
        path.lineTo(0.74, 0.44)
        path.lineTo(0.38, 0.44)
        path.lineTo(0.38, 0.10)
        # Forward path
        path.moveTo(0.10, 0.27)
        path.lineTo(0.38, 0.27)
        path.moveTo(0.74, 0.27)
        path.lineTo(0.94, 0.27)
        # Feedback tap down to the gain
        path.moveTo(0.88, 0.27)
        path.lineTo(0.88, 0.80)
        path.lineTo(0.62, 0.80)
        # Gain triangle K (pointing left)
        path.moveTo(0.62, 0.62)
        path.lineTo(0.62, 0.98)
        path.lineTo(0.34, 0.80)
        path.lineTo(0.62, 0.62)
        # Return to the summing point
        path.moveTo(0.34, 0.80)
        path.lineTo(0.10, 0.80)
        path.lineTo(0.10, 0.27)
