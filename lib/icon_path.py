"""Toolkit-neutral block icon paths.

A block's ``draw_icon(block_rect)`` describes its icon in normalized
coordinates (0..1, origin top-left). It used to build a ``QPainterPath``, so
every block module needed PyQt6 to draw itself and a block author had to know
Qt. It now returns an :class:`IconPath`, which only records the drawing
operations; the GUI renderer replays them onto a real ``QPainterPath``
(:func:`to_qpainterpath`), so the result is element-for-element identical.

The methods and argument orders are those of ``QPainterPath`` (numbers only,
no point or rect objects), so existing icon code ports by changing the import:

    path = IconPath()
    path.moveTo(0.1, 0.5)
    path.cubicTo(0.3, 0.1, 0.7, 0.9, 0.9, 0.5)
    path.addEllipse(0.4, 0.4, 0.2, 0.2)

A ``QPainterPath`` returned by an older user block is still accepted by
:func:`to_qpainterpath`.
"""

from typing import List, Tuple

__all__ = ["IconPath", "to_qpainterpath"]

_Op = Tuple[str, Tuple[float, ...]]

# Operation name -> number of numeric arguments it takes (Qt's float overloads).
_ARITY = {
    "moveTo": (2,),
    "lineTo": (2,),
    "quadTo": (4,),
    "cubicTo": (6,),
    "arcTo": (6,),
    "arcMoveTo": (5,),
    "addRect": (4,),
    "addEllipse": (4,),
    "addRoundedRect": (6,),
    "closeSubpath": (0,),
}


class IconPath:
    """Records ``QPainterPath``-style drawing operations (see module doc)."""

    __slots__ = ("_ops",)

    def __init__(self):
        self._ops: List[_Op] = []

    def _record(self, name, args):
        if len(args) not in _ARITY[name]:
            raise TypeError(
                f"IconPath.{name}() takes {_ARITY[name][0]} numbers, got {len(args)}: {args!r}"
            )
        self._ops.append((name, tuple(float(a) for a in args)))

    def moveTo(self, x, y):
        self._record("moveTo", (x, y))

    def lineTo(self, x, y):
        self._record("lineTo", (x, y))

    def quadTo(self, cx, cy, ex, ey):
        self._record("quadTo", (cx, cy, ex, ey))

    def cubicTo(self, c1x, c1y, c2x, c2y, ex, ey):
        self._record("cubicTo", (c1x, c1y, c2x, c2y, ex, ey))

    def arcTo(self, x, y, w, h, start_angle, sweep_length):
        self._record("arcTo", (x, y, w, h, start_angle, sweep_length))

    def arcMoveTo(self, x, y, w, h, angle):
        self._record("arcMoveTo", (x, y, w, h, angle))

    def addRect(self, x, y, w, h):
        self._record("addRect", (x, y, w, h))

    def addEllipse(self, x, y, w, h):
        self._record("addEllipse", (x, y, w, h))

    def addRoundedRect(self, x, y, w, h, x_radius, y_radius):
        self._record("addRoundedRect", (x, y, w, h, x_radius, y_radius))

    def closeSubpath(self):
        self._record("closeSubpath", ())

    @property
    def ops(self) -> Tuple[_Op, ...]:
        """The recorded ``(method name, numeric args)`` operations, in order."""
        return tuple(self._ops)

    def isEmpty(self) -> bool:
        """True when nothing would be drawn: no operations, or only moves
        (Qt's ``QPainterPath.isEmpty`` gives the same answer on the replay)."""
        return all(name in ("moveTo", "arcMoveTo") for name, _ in self._ops)

    def replay(self, target):
        """Call each recorded operation on ``target`` (e.g. a QPainterPath)."""
        for name, args in self._ops:
            getattr(target, name)(*args)
        return target

    def __eq__(self, other):
        if not isinstance(other, IconPath):
            return NotImplemented
        return self._ops == other._ops

    __hash__ = None  # mutable: equal paths may stop being equal

    def __repr__(self):
        return f"IconPath({len(self._ops)} ops)"


def to_qpainterpath(path):
    """A ``QPainterPath`` for ``path`` (an IconPath, a QPainterPath, or None).

    Needs PyQt6; only the GUI (and tests) call it.
    """
    from PyQt6.QtGui import QPainterPath

    if path is None:
        return QPainterPath()
    if isinstance(path, QPainterPath):
        return path
    if isinstance(path, IconPath):
        return path.replay(QPainterPath())
    raise TypeError(f"draw_icon must return an IconPath or None, not {type(path).__name__}")
