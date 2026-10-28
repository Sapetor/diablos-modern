"""Crash-safe file writes (Qt-free, so both ``lib/services`` and the
headless loaders can use it)."""

import json
import os
import stat
import tempfile
from typing import Any, Optional


def _current_umask() -> int:
    mask = os.umask(0)
    os.umask(mask)
    return mask


def atomic_write_json(filename: str, data: Any, indent: Optional[int] = 4) -> None:
    """Write ``data`` as JSON to ``filename`` atomically.

    The JSON goes to a temporary file in the same folder, is flushed to disk,
    and then replaces ``filename`` in a single ``os.replace``. A failure at any
    point (unserializable data, full disk, crash mid-write) leaves the previous
    file intact, and the temporary file is removed. The replacement keeps the
    old file's permission bits (or the umask default for a new file), since
    ``mkstemp`` creates it owner-only.
    """
    dirname = os.path.dirname(os.path.abspath(filename))
    try:
        mode = stat.S_IMODE(os.stat(filename).st_mode)
    except FileNotFoundError:
        mode = 0o666 & ~_current_umask()

    fd, tmp_path = tempfile.mkstemp(
        prefix="." + os.path.basename(filename) + ".", suffix=".tmp", dir=dirname
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fp:
            json.dump(data, fp, indent=indent)
            fp.flush()
            os.fsync(fp.fileno())
        os.chmod(tmp_path, mode)
        os.replace(tmp_path, filename)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
