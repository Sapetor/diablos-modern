"""Geometry and colour value types for ``lib/``, with or without PyQt6.

The diagram model (``DBlock``, ``DLine``, the wire router, file loading) keeps
its coordinates and colours as ``QPoint`` / ``QRect`` / ``QColor`` objects,
which the GUI hands straight to ``QPainter``. Importing them from PyQt6 at
module level made the whole engine require PyQt6, so a headless run (CLI,
notebook, cluster Monte Carlo) could not even ``import lib.lib`` without it.

This module re-exports the real Qt classes when PyQt6 is importable, so the
GUI behaves exactly as before, and otherwise provides small pure-Python
stand-ins implementing the subset of the Qt API the engine uses. The
semantics follow Qt, including its integer-rectangle quirk
(``QRect.right() == left + width - 1``).

``QPainterPath`` stand-in only records the elements: hit testing and painting
happen in the GUI, which always has the real class.

Set ``DIABLOS_PURE_TYPES=1`` to force the stand-ins even when PyQt6 is
installed (used by the tests).
"""

import math
import os

__all__ = ["HAS_QT", "QColor", "QPainterPath", "QPoint", "QPointF", "QRect", "QRectF"]

_FORCE_PURE = os.environ.get("DIABLOS_PURE_TYPES") == "1"

try:
    if _FORCE_PURE:
        raise ImportError("DIABLOS_PURE_TYPES=1")
    from PyQt6.QtCore import QPoint, QPointF, QRect, QRectF
    from PyQt6.QtGui import QColor, QPainterPath

    HAS_QT = True
except ImportError:
    HAS_QT = False


if not HAS_QT:

    def _half(n):
        return n // 2 if n >= 0 else -(-n // 2)

    class QPoint:  # noqa: D101 - mirrors Qt
        __slots__ = ("_x", "_y")

        def __init__(self, *args):
            if not args:
                x, y = 0, 0
            elif len(args) == 1:
                x, y = args[0].x(), args[0].y()
            else:
                x, y = args
            self._x, self._y = int(round(x)), int(round(y))

        def x(self):
            return self._x

        def y(self):
            return self._y

        def setX(self, x):
            self._x = int(round(x))

        def setY(self, y):
            self._y = int(round(y))

        def isNull(self):
            return self._x == 0 and self._y == 0

        def manhattanLength(self):
            return abs(self._x) + abs(self._y)

        def toPointF(self):
            return QPointF(self._x, self._y)

        def __add__(self, o):
            return type(self)(self.x() + o.x(), self.y() + o.y())

        def __sub__(self, o):
            return type(self)(self.x() - o.x(), self.y() - o.y())

        def __mul__(self, k):
            return type(self)(self.x() * k, self.y() * k)

        __rmul__ = __mul__

        def __truediv__(self, k):
            return type(self)(self.x() / k, self.y() / k)

        def __neg__(self):
            return type(self)(-self.x(), -self.y())

        def __eq__(self, o):
            try:
                return self.x() == o.x() and self.y() == o.y()
            except AttributeError:
                return NotImplemented

        def __hash__(self):
            return hash((self.x(), self.y()))

        def __repr__(self):
            return f"{type(self).__name__}({self.x()}, {self.y()})"

        def __reduce__(self):
            return (type(self), (self.x(), self.y()))

    class QPointF(QPoint):  # noqa: D101
        __slots__ = ()

        def __init__(self, *args):
            if not args:
                x, y = 0.0, 0.0
            elif len(args) == 1:
                x, y = args[0].x(), args[0].y()
            else:
                x, y = args
            self._x, self._y = float(x), float(y)

        def setX(self, x):
            self._x = float(x)

        def setY(self, y):
            self._y = float(y)

        def toPoint(self):
            return QPoint(self._x, self._y)

    class QRect:  # noqa: D101
        """Integer rectangle; right()/bottom() are inclusive like Qt's."""

        __slots__ = ("_l", "_t", "_w", "_h")

        def __init__(self, *args):
            if not args:
                self._l = self._t = self._w = self._h = 0
            elif len(args) == 4:
                self._l, self._t, self._w, self._h = (int(round(a)) for a in args)
            elif len(args) == 2 and hasattr(args[1], "x"):  # two corners
                a, b = args
                self._l, self._t = int(a.x()), int(a.y())
                self._w, self._h = int(b.x()) - self._l + 1, int(b.y()) - self._t + 1
            elif len(args) == 2:  # top-left + size (w, h) tuple-like
                p, s = args
                self._l, self._t = int(p.x()), int(p.y())
                self._w, self._h = int(s.width()), int(s.height())
            elif len(args) == 1:
                r = args[0]
                self._l, self._t, self._w, self._h = r.left(), r.top(), r.width(), r.height()
            else:
                raise TypeError(f"QRect{args!r}")

        def left(self):
            return self._l

        def top(self):
            return self._t

        def width(self):
            return self._w

        def height(self):
            return self._h

        def right(self):
            return self._l + self._w - 1

        def bottom(self):
            return self._t + self._h - 1

        def x(self):
            return self._l

        def y(self):
            return self._t

        def topLeft(self):
            return QPoint(self._l, self._t)

        def topRight(self):
            return QPoint(self.right(), self._t)

        def bottomLeft(self):
            return QPoint(self._l, self.bottom())

        def bottomRight(self):
            return QPoint(self.right(), self.bottom())

        def center(self):
            # C++ integer division truncates toward zero.
            return QPoint(_half(self._l + self.right()), _half(self._t + self.bottom()))

        def isNull(self):
            return self._w == 0 and self._h == 0

        def isEmpty(self):
            return self._w <= 0 or self._h <= 0

        def isValid(self):
            return self._w > 0 and self._h > 0

        def setLeft(self, v):
            r = self.right()
            self._l = int(v)
            self._w = r - self._l + 1

        def setTop(self, v):
            b = self.bottom()
            self._t = int(v)
            self._h = b - self._t + 1

        def setRight(self, v):
            self._w = int(v) - self._l + 1

        def setBottom(self, v):
            self._h = int(v) - self._t + 1

        def setWidth(self, w):
            self._w = int(w)

        def setHeight(self, h):
            self._h = int(h)

        def setX(self, v):
            self.setLeft(v)

        def setY(self, v):
            self.setTop(v)

        def setRect(self, left, top, w, h):
            self._l, self._t, self._w, self._h = int(left), int(top), int(w), int(h)

        def moveTo(self, *args):
            p = args[0] if len(args) == 1 else QPoint(*args)
            self._l, self._t = int(p.x()), int(p.y())

        moveTopLeft = moveTo

        def moveLeft(self, v):
            self._l = int(v)

        def moveTop(self, v):
            self._t = int(v)

        def moveCenter(self, p):
            c = self.center()
            self.translate(p.x() - c.x(), p.y() - c.y())

        def translate(self, *args):
            dx, dy = (args[0].x(), args[0].y()) if len(args) == 1 else args
            self._l += int(dx)
            self._t += int(dy)

        def translated(self, *args):
            r = type(self)(self)
            r.translate(*args)
            return r

        def adjusted(self, dx1, dy1, dx2, dy2):
            return type(self)(
                self._l + dx1, self._t + dy1, self._w + dx2 - dx1, self._h + dy2 - dy1
            )

        def adjust(self, dx1, dy1, dx2, dy2):
            r = self.adjusted(dx1, dy1, dx2, dy2)
            self._l, self._t, self._w, self._h = r._l, r._t, r._w, r._h

        def normalized(self):
            left, top, w, h = self._l, self._t, self._w, self._h
            if w < 0:
                left, w = left + w, -w
            if h < 0:
                top, h = top + h, -h
            return type(self)(left, top, w, h)

        def contains(self, *args, proper=False):
            if len(args) == 2 and not hasattr(args[0], "x"):
                args = (QPoint(*args),)
            o = args[0]
            n = self.normalized()
            if isinstance(o, QRect):
                o = o.normalized()
                if o.isNull() or n.isNull():
                    return False
                return (
                    n.left() <= o.left()
                    and o.right() <= n.right()
                    and n.top() <= o.top()
                    and o.bottom() <= n.bottom()
                )
            if n.isNull():
                return False
            x, y = o.x(), o.y()
            return n.left() <= x <= n.right() and n.top() <= y <= n.bottom()

        def intersects(self, o):
            a, b = self.normalized(), o.normalized()
            if a.isNull() or b.isNull():
                return False
            return not (
                a.left() > b.right()
                or b.left() > a.right()
                or a.top() > b.bottom()
                or b.top() > a.bottom()
            )

        def united(self, o):
            if self.isNull():
                return type(self)(o)
            if o.isNull():
                return type(self)(self)
            a, b = self.normalized(), o.normalized()
            left, top = min(a.left(), b.left()), min(a.top(), b.top())
            right, bottom = max(a.right(), b.right()), max(a.bottom(), b.bottom())
            return type(self)(left, top, right - left + 1, bottom - top + 1)

        def intersected(self, o):
            a, b = self.normalized(), o.normalized()
            left, top = max(a.left(), b.left()), max(a.top(), b.top())
            right, bottom = min(a.right(), b.right()), min(a.bottom(), b.bottom())
            if left > right or top > bottom:
                return type(self)()
            return type(self)(left, top, right - left + 1, bottom - top + 1)

        def getRect(self):
            return (self._l, self._t, self._w, self._h)

        def __eq__(self, o):
            try:
                return (self.left(), self.top(), self.width(), self.height()) == (
                    o.left(),
                    o.top(),
                    o.width(),
                    o.height(),
                )
            except AttributeError:
                return NotImplemented

        def __hash__(self):
            return hash(self.getRect())

        def __repr__(self):
            return f"QRect({self._l}, {self._t}, {self._w}, {self._h})"

        def __reduce__(self):
            return (type(self), self.getRect())

    class QRectF(QRect):  # noqa: D101
        """Float rectangle; right()/bottom() are exclusive like Qt's."""

        __slots__ = ()

        def __init__(self, *args):
            if len(args) == 4:
                self._l, self._t, self._w, self._h = (float(a) for a in args)
            elif len(args) == 1:
                r = args[0]
                self._l, self._t = float(r.left()), float(r.top())
                self._w, self._h = float(r.width()), float(r.height())
            elif len(args) == 2 and hasattr(args[1], "x"):
                a, b = args
                self._l, self._t = float(a.x()), float(a.y())
                self._w, self._h = float(b.x()) - self._l, float(b.y()) - self._t
            elif not args:
                self._l = self._t = self._w = self._h = 0.0
            else:
                raise TypeError(f"QRectF{args!r}")

        def right(self):
            return self._l + self._w

        def bottom(self):
            return self._t + self._h

        def center(self):
            return QPointF(self._l + self._w / 2, self._t + self._h / 2)

        def topLeft(self):
            return QPointF(self._l, self._t)

        def moveTo(self, *args):
            p = args[0] if len(args) == 1 else QPointF(*args)
            self._l, self._t = float(p.x()), float(p.y())

        moveTopLeft = moveTo

        def toRect(self):
            return QRect(self._l, self._t, self._w, self._h)

        def __repr__(self):
            return f"QRectF({self._l}, {self._t}, {self._w}, {self._h})"

    _NAMED = {
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "red": (255, 0, 0),
        "green": (0, 128, 0),
        "lime": (0, 255, 0),
        "blue": (0, 0, 255),
        "yellow": (255, 255, 0),
        "magenta": (255, 0, 255),
        "fuchsia": (255, 0, 255),
        "cyan": (0, 255, 255),
        "aqua": (0, 255, 255),
        "gray": (128, 128, 128),
        "grey": (128, 128, 128),
        "darkgray": (169, 169, 169),
        "lightgray": (211, 211, 211),
        "orange": (255, 165, 0),
        "purple": (128, 0, 128),
        "pink": (255, 192, 203),
        "brown": (165, 42, 42),
        "navy": (0, 0, 128),
        "transparent": (0, 0, 0, 0),
    }

    class QColor:  # noqa: D101
        __slots__ = ("_rgba", "_valid")

        def __init__(self, *args):
            self._valid = True
            if not args:
                self._rgba, self._valid = (0, 0, 0, 255), False
            elif len(args) == 1 and isinstance(args[0], QColor):
                self._rgba, self._valid = args[0]._rgba, args[0]._valid
            elif len(args) == 1 and isinstance(args[0], str):
                self._rgba = self._parse(args[0])
                if self._rgba is None:
                    self._rgba, self._valid = (0, 0, 0, 255), False
            elif len(args) in (3, 4):
                r, g, b = (int(v) for v in args[:3])
                a = int(args[3]) if len(args) == 4 else 255
                self._rgba = (r, g, b, a)
            else:
                raise TypeError(f"QColor{args!r}")

        @staticmethod
        def _parse(s):
            s = s.strip().lower()
            if s in _NAMED:
                v = _NAMED[s]
                return v if len(v) == 4 else (*v, 255)
            if s.startswith("#"):
                h = s[1:]
                try:
                    if len(h) == 3:
                        return (*(int(c * 2, 16) for c in h), 255)
                    if len(h) == 6:
                        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), 255)
                    if len(h) == 8:  # Qt's #AARRGGBB
                        a, r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4, 6))
                        return (r, g, b, a)
                except ValueError:
                    return None
            return None

        def isValid(self):
            return self._valid

        def red(self):
            return self._rgba[0]

        def green(self):
            return self._rgba[1]

        def blue(self):
            return self._rgba[2]

        def alpha(self):
            return self._rgba[3]

        def setAlpha(self, a):
            self._rgba = (*self._rgba[:3], int(a))

        def getRgb(self):
            return self._rgba

        def name(self, *_fmt):
            if not self._valid:
                return "#000000"
            return "#{:02x}{:02x}{:02x}".format(*self._rgba[:3])

        def __eq__(self, o):
            if not isinstance(o, QColor):
                return NotImplemented
            return self._valid == o._valid and self._rgba == o._rgba

        def __hash__(self):
            return hash((self._valid, self._rgba))

        def __repr__(self):
            return f"QColor({self.name()})" if self._valid else "QColor(invalid)"

        def __reduce__(self):
            return (QColor, (*self._rgba,)) if self._valid else (QColor, ())

    class _Element:
        __slots__ = ("x", "y", "_curve")

        def __init__(self, x, y, curve):
            self.x, self.y, self._curve = x, y, curve

        def isCurveTo(self):
            return self._curve

        def isLineTo(self):
            return not self._curve

    class QPainterPath:  # noqa: D101
        """Records path elements; enough for the model, not for painting."""

        def __init__(self, start=None):
            self._el = []
            if start is not None:
                self.moveTo(start)

        def moveTo(self, p):
            self._el.append(_Element(float(p.x()), float(p.y()), False))

        def lineTo(self, p):
            self._el.append(_Element(float(p.x()), float(p.y()), False))

        def quadTo(self, c, p):
            self._el.append(_Element(float(c.x()), float(c.y()), True))
            self._el.append(_Element(float(p.x()), float(p.y()), True))

        def cubicTo(self, c1, c2, p):
            for q in (c1, c2, p):
                self._el.append(_Element(float(q.x()), float(q.y()), True))

        def isEmpty(self):
            return len(self._el) <= 1

        def elementCount(self):
            return len(self._el)

        def elementAt(self, i):
            return self._el[i]

        def currentPosition(self):
            e = self._el[-1] if self._el else _Element(0.0, 0.0, False)
            return QPointF(e.x, e.y)

        def pointAtPercent(self, t):
            """Point at fraction ``t`` of the control polygon (an approximation)."""
            pts = [(e.x, e.y) for e in self._el]
            if not pts:
                return QPointF()
            lengths = [math.dist(a, b) for a, b in zip(pts, pts[1:])]
            total = sum(lengths)
            if total == 0:
                return QPointF(*pts[0])
            target = max(0.0, min(1.0, t)) * total
            for (a, b), seg in zip(zip(pts, pts[1:]), lengths):
                if target <= seg and seg > 0:
                    f = target / seg
                    return QPointF(a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)
                target -= seg
            return QPointF(*pts[-1])
