"""Geometry and colour value types for ``lib/``, with or without PyQt6.

The diagram model (``DBlock``, ``DLine``, the wire router, file loading) keeps
its coordinates and colours as ``QPoint`` / ``QRect`` / ``QColor`` objects,
which the GUI hands straight to ``QPainter``. Importing them from PyQt6 at
module level made the whole engine require PyQt6, so a headless run (CLI,
notebook, cluster Monte Carlo) could not even ``import diablos.core.lib`` without it.

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
import operator
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

    def _int(v):
        """An int argument, refusing floats as PyQt6 does (even ``1.0``)."""
        try:
            return operator.index(v)
        except TypeError:
            raise TypeError(f"expected an integer, got {type(v).__name__} {v!r}") from None

    def _round(v):
        """Qt's qRound: halves away from zero."""
        return int(math.floor(v + 0.5)) if v >= 0 else -int(math.floor(-v + 0.5))

    class QPoint:  # noqa: D101 - mirrors Qt
        __slots__ = ("_x", "_y")

        def __init__(self, *args):
            if not args:
                x, y = 0, 0
            elif len(args) == 1:
                x, y = args[0].x(), args[0].y()
            else:
                x, y = args
            self._x, self._y = _int(x), _int(y)

        @classmethod
        def _rounded(cls, x, y):
            return cls(_round(x), _round(y))

        def x(self):
            return self._x

        def y(self):
            return self._y

        def setX(self, x):
            self._x = _int(x)

        def setY(self, y):
            self._y = _int(y)

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
            return self._rounded(self.x() * k, self.y() * k)

        __rmul__ = __mul__

        def __truediv__(self, k):
            return self._rounded(self.x() / k, self.y() / k)

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

        @classmethod
        def _rounded(cls, x, y):  # float points do not round
            return cls(x, y)

        def setX(self, x):
            self._x = float(x)

        def setY(self, y):
            self._y = float(y)

        def toPoint(self):
            return QPoint._rounded(self._x, self._y)

    class QRect:  # noqa: D101
        """Integer rectangle; right()/bottom() are inclusive like Qt's."""

        __slots__ = ("_l", "_t", "_w", "_h")
        _c = staticmethod(_int)  # coordinate coercion: ints, as PyQt6 enforces

        def __init__(self, *args):
            if not args:
                self._l = self._t = self._w = self._h = 0
            elif len(args) == 4:
                self._l, self._t, self._w, self._h = (_int(a) for a in args)
            elif len(args) == 2 and hasattr(args[1], "x"):  # two corners
                a, b = args
                self._l, self._t = _int(a.x()), _int(a.y())
                self._w, self._h = _int(b.x()) - self._l + 1, _int(b.y()) - self._t + 1
            elif len(args) == 2:  # top-left + size
                p, size = args
                self._l, self._t = _int(p.x()), _int(p.y())
                self._w, self._h = _int(size.width()), _int(size.height())
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
            self._l = self._c(v)
            self._w = r - self._l + 1

        def setTop(self, v):
            b = self.bottom()
            self._t = self._c(v)
            self._h = b - self._t + 1

        def setRight(self, v):
            self._w = self._c(v) - self._l + 1

        def setBottom(self, v):
            self._h = self._c(v) - self._t + 1

        def setWidth(self, w):
            self._w = self._c(w)

        def setHeight(self, h):
            self._h = self._c(h)

        def setX(self, v):
            self.setLeft(v)

        def setY(self, v):
            self.setTop(v)

        def setRect(self, left, top, w, h):
            self._l, self._t, self._w, self._h = self._c(left), self._c(top), self._c(w), self._c(h)

        def moveTo(self, *args):
            p = args[0] if len(args) == 1 else QPoint(*args)
            self._l, self._t = self._c(p.x()), self._c(p.y())

        moveTopLeft = moveTo

        def moveLeft(self, v):
            self._l = self._c(v)

        def moveTop(self, v):
            self._t = self._c(v)

        def moveCenter(self, p):
            c = self.center()
            self.translate(p.x() - c.x(), p.y() - c.y())

        def translate(self, *args):
            dx, dy = (args[0].x(), args[0].y()) if len(args) == 1 else args
            self._l += self._c(dx)
            self._t += self._c(dy)

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
        _c = staticmethod(float)

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
            left, top = _round(self._l), _round(self._t)
            return QRect(
                QPoint(left, top),
                QPoint(_round(self._l + self._w) - 1, _round(self._t + self._h) - 1),
            )

        def __repr__(self):
            return f"QRectF({self._l}, {self._t}, {self._w}, {self._h})"

    # Qt's named colours (QColor.colorNames(): the SVG set plus transparent).
    _NAMED = {
        "aliceblue": (240, 248, 255, 255),
        "antiquewhite": (250, 235, 215, 255),
        "aqua": (0, 255, 255, 255),
        "aquamarine": (127, 255, 212, 255),
        "azure": (240, 255, 255, 255),
        "beige": (245, 245, 220, 255),
        "bisque": (255, 228, 196, 255),
        "black": (0, 0, 0, 255),
        "blanchedalmond": (255, 235, 205, 255),
        "blue": (0, 0, 255, 255),
        "blueviolet": (138, 43, 226, 255),
        "brown": (165, 42, 42, 255),
        "burlywood": (222, 184, 135, 255),
        "cadetblue": (95, 158, 160, 255),
        "chartreuse": (127, 255, 0, 255),
        "chocolate": (210, 105, 30, 255),
        "coral": (255, 127, 80, 255),
        "cornflowerblue": (100, 149, 237, 255),
        "cornsilk": (255, 248, 220, 255),
        "crimson": (220, 20, 60, 255),
        "cyan": (0, 255, 255, 255),
        "darkblue": (0, 0, 139, 255),
        "darkcyan": (0, 139, 139, 255),
        "darkgoldenrod": (184, 134, 11, 255),
        "darkgray": (169, 169, 169, 255),
        "darkgreen": (0, 100, 0, 255),
        "darkgrey": (169, 169, 169, 255),
        "darkkhaki": (189, 183, 107, 255),
        "darkmagenta": (139, 0, 139, 255),
        "darkolivegreen": (85, 107, 47, 255),
        "darkorange": (255, 140, 0, 255),
        "darkorchid": (153, 50, 204, 255),
        "darkred": (139, 0, 0, 255),
        "darksalmon": (233, 150, 122, 255),
        "darkseagreen": (143, 188, 143, 255),
        "darkslateblue": (72, 61, 139, 255),
        "darkslategray": (47, 79, 79, 255),
        "darkslategrey": (47, 79, 79, 255),
        "darkturquoise": (0, 206, 209, 255),
        "darkviolet": (148, 0, 211, 255),
        "deeppink": (255, 20, 147, 255),
        "deepskyblue": (0, 191, 255, 255),
        "dimgray": (105, 105, 105, 255),
        "dimgrey": (105, 105, 105, 255),
        "dodgerblue": (30, 144, 255, 255),
        "firebrick": (178, 34, 34, 255),
        "floralwhite": (255, 250, 240, 255),
        "forestgreen": (34, 139, 34, 255),
        "fuchsia": (255, 0, 255, 255),
        "gainsboro": (220, 220, 220, 255),
        "ghostwhite": (248, 248, 255, 255),
        "gold": (255, 215, 0, 255),
        "goldenrod": (218, 165, 32, 255),
        "gray": (128, 128, 128, 255),
        "green": (0, 128, 0, 255),
        "greenyellow": (173, 255, 47, 255),
        "grey": (128, 128, 128, 255),
        "honeydew": (240, 255, 240, 255),
        "hotpink": (255, 105, 180, 255),
        "indianred": (205, 92, 92, 255),
        "indigo": (75, 0, 130, 255),
        "ivory": (255, 255, 240, 255),
        "khaki": (240, 230, 140, 255),
        "lavender": (230, 230, 250, 255),
        "lavenderblush": (255, 240, 245, 255),
        "lawngreen": (124, 252, 0, 255),
        "lemonchiffon": (255, 250, 205, 255),
        "lightblue": (173, 216, 230, 255),
        "lightcoral": (240, 128, 128, 255),
        "lightcyan": (224, 255, 255, 255),
        "lightgoldenrodyellow": (250, 250, 210, 255),
        "lightgray": (211, 211, 211, 255),
        "lightgreen": (144, 238, 144, 255),
        "lightgrey": (211, 211, 211, 255),
        "lightpink": (255, 182, 193, 255),
        "lightsalmon": (255, 160, 122, 255),
        "lightseagreen": (32, 178, 170, 255),
        "lightskyblue": (135, 206, 250, 255),
        "lightslategray": (119, 136, 153, 255),
        "lightslategrey": (119, 136, 153, 255),
        "lightsteelblue": (176, 196, 222, 255),
        "lightyellow": (255, 255, 224, 255),
        "lime": (0, 255, 0, 255),
        "limegreen": (50, 205, 50, 255),
        "linen": (250, 240, 230, 255),
        "magenta": (255, 0, 255, 255),
        "maroon": (128, 0, 0, 255),
        "mediumaquamarine": (102, 205, 170, 255),
        "mediumblue": (0, 0, 205, 255),
        "mediumorchid": (186, 85, 211, 255),
        "mediumpurple": (147, 112, 219, 255),
        "mediumseagreen": (60, 179, 113, 255),
        "mediumslateblue": (123, 104, 238, 255),
        "mediumspringgreen": (0, 250, 154, 255),
        "mediumturquoise": (72, 209, 204, 255),
        "mediumvioletred": (199, 21, 133, 255),
        "midnightblue": (25, 25, 112, 255),
        "mintcream": (245, 255, 250, 255),
        "mistyrose": (255, 228, 225, 255),
        "moccasin": (255, 228, 181, 255),
        "navajowhite": (255, 222, 173, 255),
        "navy": (0, 0, 128, 255),
        "oldlace": (253, 245, 230, 255),
        "olive": (128, 128, 0, 255),
        "olivedrab": (107, 142, 35, 255),
        "orange": (255, 165, 0, 255),
        "orangered": (255, 69, 0, 255),
        "orchid": (218, 112, 214, 255),
        "palegoldenrod": (238, 232, 170, 255),
        "palegreen": (152, 251, 152, 255),
        "paleturquoise": (175, 238, 238, 255),
        "palevioletred": (219, 112, 147, 255),
        "papayawhip": (255, 239, 213, 255),
        "peachpuff": (255, 218, 185, 255),
        "peru": (205, 133, 63, 255),
        "pink": (255, 192, 203, 255),
        "plum": (221, 160, 221, 255),
        "powderblue": (176, 224, 230, 255),
        "purple": (128, 0, 128, 255),
        "red": (255, 0, 0, 255),
        "rosybrown": (188, 143, 143, 255),
        "royalblue": (65, 105, 225, 255),
        "saddlebrown": (139, 69, 19, 255),
        "salmon": (250, 128, 114, 255),
        "sandybrown": (244, 164, 96, 255),
        "seagreen": (46, 139, 87, 255),
        "seashell": (255, 245, 238, 255),
        "sienna": (160, 82, 45, 255),
        "silver": (192, 192, 192, 255),
        "skyblue": (135, 206, 235, 255),
        "slateblue": (106, 90, 205, 255),
        "slategray": (112, 128, 144, 255),
        "slategrey": (112, 128, 144, 255),
        "snow": (255, 250, 250, 255),
        "springgreen": (0, 255, 127, 255),
        "steelblue": (70, 130, 180, 255),
        "tan": (210, 180, 140, 255),
        "teal": (0, 128, 128, 255),
        "thistle": (216, 191, 216, 255),
        "tomato": (255, 99, 71, 255),
        "transparent": (0, 0, 0, 0),
        "turquoise": (64, 224, 208, 255),
        "violet": (238, 130, 238, 255),
        "wheat": (245, 222, 179, 255),
        "white": (255, 255, 255, 255),
        "whitesmoke": (245, 245, 245, 255),
        "yellow": (255, 255, 0, 255),
        "yellowgreen": (154, 205, 50, 255),
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
                return _NAMED[s]
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
        """A path element, typed like Qt's (0 MoveTo, 1 LineTo, 2 CurveTo,
        3 CurveToData: a cubic is one CurveTo followed by two data points)."""

        __slots__ = ("x", "y", "type")

        def __init__(self, x, y, kind):
            self.x, self.y, self.type = x, y, kind

        def isMoveTo(self):
            return self.type == 0

        def isLineTo(self):
            return self.type == 1

        def isCurveTo(self):
            return self.type == 2

    _FLATTEN = 64  # line segments per cubic when measuring arc length

    def _bezier_at(ctrl, t):
        (x0, y0), (x1, y1), (x2, y2), (x3, y3) = ctrl
        u = 1.0 - t
        a, b, c, d = u * u * u, 3 * u * u * t, 3 * u * t * t, t * t * t
        return (a * x0 + b * x1 + c * x2 + d * x3, a * y0 + b * y1 + c * y2 + d * y3)

    class QPainterPath:  # noqa: D101
        """Qt's element model (see _Element); enough for the engine and wire
        hit-testing, not for painting (the GUI always has the real class)."""

        def __init__(self, start=None):
            self._el = []
            if start is not None:
                self.moveTo(start)

        @staticmethod
        def _xy(args):
            if len(args) == 1:
                return float(args[0].x()), float(args[0].y())
            return float(args[0]), float(args[1])

        def _cur(self):
            if not self._el:
                self._el.append(_Element(0.0, 0.0, 0))
            e = self._el[-1]
            return e.x, e.y

        def moveTo(self, *args):
            x, y = self._xy(args)
            if self._el and self._el[-1].type == 0:
                self._el[-1] = _Element(x, y, 0)  # Qt replaces a trailing moveTo
            else:
                self._el.append(_Element(x, y, 0))

        def lineTo(self, *args):
            x, y = self._xy(args)
            if self._cur() == (x, y):
                return  # Qt drops a zero-length line
            self._el.append(_Element(x, y, 1))

        def cubicTo(self, *args):
            if len(args) == 3:
                pts = [self._xy((a,)) for a in args]
            else:
                pts = [(float(args[i]), float(args[i + 1])) for i in (0, 2, 4)]
            if self._cur() == pts[0] == pts[1] == pts[2]:
                return  # Qt drops a degenerate curve
            for kind, (x, y) in zip((2, 3, 3), pts):
                self._el.append(_Element(x, y, kind))

        def quadTo(self, *args):
            if len(args) == 2:
                (cx, cy), (ex, ey) = self._xy((args[0],)), self._xy((args[1],))
            else:
                cx, cy, ex, ey = (float(a) for a in args)
            sx, sy = self._cur()
            if (sx, sy) == (cx, cy) == (ex, ey):
                return
            # Qt stores a quadratic as the equivalent cubic.
            c1 = (sx + 2.0 / 3.0 * (cx - sx), sy + 2.0 / 3.0 * (cy - sy))
            c2 = (ex + 2.0 / 3.0 * (cx - ex), ey + 2.0 / 3.0 * (cy - ey))
            self.cubicTo(*c1, *c2, ex, ey)

        def isEmpty(self):
            return not self._el or (len(self._el) == 1 and self._el[0].type == 0)

        def elementCount(self):
            return len(self._el)

        def elementAt(self, i):
            return self._el[i]

        def currentPosition(self):
            return QPointF(*self._cur()) if self._el else QPointF()

        def _beziers(self):
            """Each drawn segment as a cubic (a line is a straight cubic) with
            its arc length. A moveTo starts a new subpath and adds no length."""
            out, cur, i, el = [], None, 0, self._el
            while i < len(el):
                e = el[i]
                if e.type == 0:
                    cur = (e.x, e.y)
                    i += 1
                    continue
                if e.type == 1:
                    end = (e.x, e.y)
                    ctrl = (cur, cur, end, end)
                    i += 1
                else:
                    end = (el[i + 2].x, el[i + 2].y)
                    ctrl = (cur, (el[i].x, el[i].y), (el[i + 1].x, el[i + 1].y), end)
                    i += 3
                pts = [_bezier_at(ctrl, k / _FLATTEN) for k in range(_FLATTEN + 1)]
                out.append((ctrl, sum(math.dist(p, q) for p, q in zip(pts, pts[1:]))))
                cur = end
            return out

        def length(self):
            return sum(n for _c, n in self._beziers())

        def pointAtPercent(self, t):
            """Qt's rule: find the segment holding fraction ``t`` of the total
            length, then evaluate it at the *linear* fraction of its length."""
            beziers = self._beziers()
            if not beziers:
                e = self._el[0] if self._el else None
                return QPointF(e.x, e.y) if e else QPointF()
            total = sum(n for _c, n in beziers)
            target = max(0.0, min(1.0, t)) * total
            cur = 0.0
            for ctrl, n in beziers:
                if cur + n >= target:
                    real_t = (target - cur) / n if n > 0 else 0.0
                    return QPointF(*_bezier_at(ctrl, max(0.0, min(1.0, real_t))))
                cur += n
            return QPointF(*beziers[-1][0][3])
