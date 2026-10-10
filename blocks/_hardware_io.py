"""Shared plumbing for the UDP and serial hardware-I/O blocks.

Open sockets and serial ports are not stored in ``params``: params are saved
to ``.diablos`` files, deep-copied by the experiment runners and reset by the
engine. A block keeps an integer handle there instead (``_io_handle_``), and
the live object sits in this module's registry. Re-initializing a run
(``_init_start_``) closes the previous handle; everything left open is
closed at interpreter exit.

Wire formats:

* ``text``: ASCII numbers separated by commas, semicolons or whitespace, one
  message per datagram / per newline-terminated serial line (what an Arduino
  ``Serial.println(a); `` or ``print(f"{a},{b}")`` produces).
* ``float64``: little-endian IEEE doubles packed back to back (UDP only).
"""

from __future__ import annotations

import atexit
import itertools
import re
import struct
from typing import Dict, List, Optional

import numpy as np

_HANDLES: Dict[int, object] = {}
_COUNTER = itertools.count(1)
_SPLIT = re.compile(r"[,;\s]+")


def register(obj) -> int:
    handle = next(_COUNTER)
    _HANDLES[handle] = obj
    return handle


def get(handle) -> Optional[object]:
    return _HANDLES.get(handle)


def release(handle) -> None:
    obj = _HANDLES.pop(handle, None)
    if obj is not None:
        try:
            obj.close()
        except Exception:  # noqa: BLE001 -- closing must never raise into a run
            pass


@atexit.register
def release_all() -> None:
    for handle in list(_HANDLES):
        release(handle)


def reopen(params, opener) -> object:
    """Close the block's previous handle and open a new one with ``opener()``."""
    release(params.pop("_io_handle_", None))
    obj = opener()
    params["_io_handle_"] = register(obj)
    return obj


def current(params) -> Optional[object]:
    return get(params.get("_io_handle_"))


def parse(payload: bytes, fmt: str) -> Optional[np.ndarray]:
    """Decode one message; None if it holds no numbers."""
    if fmt == "float64":
        n = len(payload) // 8
        if n == 0:
            return None
        return np.array(struct.unpack(f"<{n}d", payload[: 8 * n]), dtype=float)
    text = payload.decode("ascii", errors="ignore").strip()
    if not text:
        return None
    values: List[float] = []
    for token in _SPLIT.split(text):
        if not token:
            continue
        try:
            values.append(float(token))
        except ValueError:
            return None  # a malformed message is dropped, not half-used
    return np.array(values, dtype=float) if values else None


def encode(values, fmt: str) -> bytes:
    flat = np.atleast_1d(np.asarray(values, dtype=float)).ravel()
    if fmt == "float64":
        return struct.pack(f"<{flat.size}d", *flat)
    return (",".join(repr(float(v)) for v in flat) + "\n").encode("ascii")


def fit_width(values: Optional[np.ndarray], width: int, held: np.ndarray) -> np.ndarray:
    """``values`` padded/truncated to ``width``; missing entries keep ``held``."""
    out = held.copy()
    if values is not None:
        n = min(width, values.size)
        out[:n] = values[:n]
    return out
