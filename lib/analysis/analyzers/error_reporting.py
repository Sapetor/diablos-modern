"""User-facing error reporting for the control-system analyzers.

``lib/`` is the engine layer and must not import Qt widgets: an analyzer that
pops its own ``QMessageBox`` cannot be driven headlessly (tests, the CLI runner,
Monte-Carlo/sweep workers) and inverts the GUI dependency.

Instead, the GUI injects a callback -- ``ControlSystemAnalyzer(..., error_cb=...)``,
supplied by ``ModernCanvas`` -- with the signature ``(title, message) -> None``.
When no callback is injected the failure is only logged, which is exactly the
right behaviour for a headless run.
"""

import logging

logger = logging.getLogger(__name__)


def report_error(error_cb, title, message):
    """Send ``message`` to the injected GUI sink, or log it when there is none.

    Returns True when the callback handled it, False when it was only logged.
    """
    if callable(error_cb):
        try:
            error_cb(title, message)
            return True
        except Exception:  # noqa: BLE001 - a broken GUI sink must not hide the error
            logger.exception("Analysis error callback failed; logging instead")
    logger.error("%s: %s", title, message)
    return False


class ErrorReportingMixin:
    """Adds an injectable ``error_cb`` and a ``_show_error`` helper.

    Mixed in ahead of :class:`~lib.analysis.analyzers.base_analyzer.BaseAnalyzer`
    so subclasses keep the ``(parent)`` constructor they always had while gaining
    the optional ``error_cb``.
    """

    #: Window/dialog title used when a caller does not pass one.
    error_title = "Analysis Error"

    def __init__(self, parent=None, error_cb=None):
        super().__init__(parent)
        self.error_cb = error_cb

    def _show_error(self, message, title=None):
        """Report a user-facing failure through the injected callback."""
        return report_error(self.error_cb, title or self.error_title, message)

    def _report_no_model(self, what):
        """Report "no LTI model" using :attr:`last_error` when it is known.

        ``BaseAnalyzer`` records *why* extraction was refused (a branching
        subsystem, internal feedback, mixed rates, a non-LTI block) in
        ``last_error``; surfacing it beats the generic "no valid linear system
        found" the analyzers used to log.
        """
        reason = getattr(self, "last_error", None)
        if reason:
            message = "Cannot generate {}:\n{}".format(what, reason)
        else:
            message = (
                "Cannot generate {}:\nno linear (transfer-function) model could be "
                "extracted from the connected blocks.".format(what)
            )
        logger.warning("No valid linear system found for %s", what)
        return self._show_error(message)
