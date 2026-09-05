"""Shared history-buffer machinery for the transport-delay blocks.

``TransportDelay`` (fixed tau) and ``VariableTransportDelay`` (tau read from a
port) both answer the same question every step: *what was the input at
``target_time``?*  Both keep a ``(time, value)`` deque pair in ``params`` and
read it back with the same linear interpolation, so that read lives here.

The one genuine difference is which branch wins when the requested time
coincides with **both** ends of the buffer -- possible only when every recorded
timestamp is equal, e.g. during the interpreter's four RK4 sub-steps at a fixed
``time``.  ``TransportDelay`` answers ``initial_value`` there and
``VariableTransportDelay`` answers the newest sample (its buffer is seeded at
t0, so ``buffer[0] == buffer[-1]`` on the very first step and tau = 0 has to
stay a passthrough).  ``_HOLD_LAST_ON_TIE`` selects between the two rather than
silently changing either block's behaviour.
"""

import bisect

import numpy as np

from blocks.base_block import BaseBlock


class DelayBufferBlock(BaseBlock):
    """Abstract base for blocks that delay a signal via a ``(time, value)`` history."""

    #: When the request lands at/after the newest sample *and* at/before the
    #: oldest one, return the newest sample (True) or ``initial_value`` (False).
    _HOLD_LAST_ON_TIE = False

    def _interpolate(self, time_buffer, value_buffer, target_time, initial_value):
        """Linear interpolation of the recorded history at ``target_time``.

        Args:
            time_buffer: Sequence of recorded timestamps, ascending.
            value_buffer: Recorded values, parallel to ``time_buffer``.
            target_time: Instant to read the signal at.
            initial_value: Value to report for instants before the history starts.

        Returns:
            np.ndarray: The (interpolated) value at ``target_time``.
        """
        if len(time_buffer) == 0:
            return np.atleast_1d(initial_value)

        if self._HOLD_LAST_ON_TIE and target_time >= time_buffer[-1]:
            return value_buffer[-1].copy()

        # Request precedes the start of recorded history.
        if target_time <= time_buffer[0]:
            return np.atleast_1d(initial_value)

        # Request at/after the most recent sample (tau == 0 -> passthrough).
        if target_time >= time_buffer[-1]:
            return value_buffer[-1].copy()

        # Find bracketing indices via binary search (O(log n)).
        i = bisect.bisect_right(time_buffer, target_time) - 1
        i = max(0, min(i, len(time_buffer) - 2))

        t0 = time_buffer[i]
        t1 = time_buffer[i + 1]

        if t1 - t0 == 0:
            return value_buffer[i].copy()

        alpha = (target_time - t0) / (t1 - t0)
        v0 = value_buffer[i]
        v1 = value_buffer[i + 1]
        return (1.0 - alpha) * v0 + alpha * v1
