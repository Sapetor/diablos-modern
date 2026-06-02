"""NumPy 1.x / 2.x compatibility helpers for the PDE blocks.

NumPy 2.0 tightened two behaviours these blocks relied on:

* ``np.trapz`` was renamed to ``np.trapezoid`` (the old name was removed).
* Assigning a 1-element array into a scalar slot (e.g. ``T[0] = bc_left``)
  and ``float()`` on a >0-d array are now hard errors instead of silently
  taking the single element.

Block inputs arrive as arrays (signals are vectors), so boundary-condition
values feeding scalar slots must be coerced explicitly.
"""

import numpy as np

# np.trapz -> np.trapezoid in NumPy 2.0 (old name removed).
trapezoid = getattr(np, "trapezoid", None) or np.trapz


def as_scalar(value, default=0.0):
    """Return ``value`` as a Python float, taking the first element if it is
    an array. Works identically on NumPy 1.x and 2.x.
    """
    arr = np.asarray(value, dtype=float).reshape(-1)
    return float(arr[0]) if arr.size else float(default)
