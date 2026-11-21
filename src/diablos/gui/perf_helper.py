"""Wall-clock timers for the GUI's per-frame and per-step work.

``ModernMainWindow.safe_update`` times each tick and simulation step (warning
on slow steps, logging the totals on close) and ``ModernCanvas.paintEvent``
times each repaint.
"""

import logging
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class PerformanceHelper:
    """Named start/stop timers with per-operation statistics."""

    def __init__(self):
        self.start_times: Dict[str, float] = {}
        self.durations: Dict[str, List[float]] = {}

    def start_timer(self, operation: str) -> None:
        self.start_times[operation] = time.time()

    def end_timer(self, operation: str) -> Optional[float]:
        """Stop ``operation`` and return its duration (None if it was never started)."""
        if operation not in self.start_times:
            return None
        duration = time.time() - self.start_times.pop(operation)
        self.durations.setdefault(operation, []).append(duration)
        return duration

    def get_stats(self, operation: str) -> Optional[Dict[str, float]]:
        durations = self.durations.get(operation)
        if not durations:
            return None
        return {
            "count": len(durations),
            "total": sum(durations),
            "average": sum(durations) / len(durations),
            "min": min(durations),
            "max": max(durations),
        }

    def log_stats(self) -> None:
        for operation in self.durations:
            stats = self.get_stats(operation)
            if stats:
                logger.info(
                    f"Performance - {operation}: "
                    f"count={stats['count']}, "
                    f"avg={stats['average']:.4f}s, "
                    f"total={stats['total']:.4f}s"
                )
