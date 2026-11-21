"""The pure-Python stand-ins in lib/qt_types.py against the real Qt classes.

lib/ uses QPoint/QRect/QColor/QPainterPath for the diagram model. Without
PyQt6 it falls back to stand-ins; a mismatch with Qt would make a headless
run route wires or hit-test differently from the GUI.
"""

import importlib.util
import os
import pickle
import math
import random
import sys

import pytest
from PyQt6.QtCore import QPoint, QPointF, QRect, QRectF
from PyQt6.QtGui import QColor

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def pure():
    path = os.path.join(os.path.dirname(__file__), "..", "..", "lib", "qt_types.py")
    os.environ["DIABLOS_PURE_TYPES"] = "1"
    try:
        spec = importlib.util.spec_from_file_location("_pure_qt_types", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod  # so pickle can find the classes
        spec.loader.exec_module(mod)
    finally:
        del os.environ["DIABLOS_PURE_TYPES"]
    assert not mod.HAS_QT
    return mod


def _r(rect):
    return (rect.left(), rect.top(), rect.width(), rect.height())


def _p(pt):
    return (pt.x(), pt.y())


def test_real_qt_is_used_when_available():
    from lib import qt_types

    assert qt_types.HAS_QT and qt_types.QRect is QRect


def test_rect_operations_match_qt(pure):
    rng = random.Random(7)
    for _ in range(3000):
        a = [rng.randint(-20, 20) for _ in range(4)]
        b = [rng.randint(-20, 20) for _ in range(4)]
        q1, q2, p1, p2 = QRect(*a), QRect(*b), pure.QRect(*a), pure.QRect(*b)
        pt = (rng.randint(-25, 25), rng.randint(-25, 25))
        assert _r(q1) == _r(p1)
        assert (q1.right(), q1.bottom()) == (p1.right(), p1.bottom())
        assert _r(q1.normalized()) == _r(p1.normalized()), a
        assert _p(q1.center()) == _p(p1.center()), a
        assert q1.contains(QPoint(*pt)) == p1.contains(pure.QPoint(*pt)), (a, pt)
        assert q1.contains(q2) == p1.contains(p2), (a, b)
        assert q1.intersects(q2) == p1.intersects(p2), (a, b)
        assert _r(q1.adjusted(*b)) == _r(p1.adjusted(*b))
        assert _r(q1.translated(*pt)) == _r(p1.translated(*pt))
        assert (q1.isNull(), q1.isEmpty(), q1.isValid()) == (
            p1.isNull(),
            p1.isEmpty(),
            p1.isValid(),
        )
        if q1.isValid() and q2.isValid():
            assert _r(q1.united(q2)) == _r(p1.united(p2)), (a, b)
            if q1.intersects(q2):
                assert _r(q1.intersected(q2)) == _r(p1.intersected(p2)), (a, b)
        c1, c2 = (
            (rng.randint(-20, 20), rng.randint(-20, 20)),
            (rng.randint(-20, 20), rng.randint(-20, 20)),
        )
        qc = QRect(QPoint(*c1), QPoint(*c2))
        pc = pure.QRect(pure.QPoint(*c1), pure.QPoint(*c2))
        assert _r(qc) == _r(pc) and _r(qc.normalized()) == _r(pc.normalized()), (c1, c2)


def test_rect_setters_match_qt(pure):
    for name, arg in [("setLeft", 3), ("setTop", -2), ("setRight", 30), ("setBottom", 1)]:
        q, p = QRect(5, 6, 10, 12), pure.QRect(5, 6, 10, 12)
        getattr(q, name)(arg)
        getattr(p, name)(arg)
        assert _r(q) == _r(p), name
    q, p = QRect(5, 6, 10, 12), pure.QRect(5, 6, 10, 12)
    q.moveTopLeft(QPoint(1, 2))
    p.moveTopLeft(pure.QPoint(1, 2))
    assert _r(q) == _r(p)
    q.moveCenter(QPoint(40, 40))
    p.moveCenter(pure.QPoint(40, 40))
    assert _r(q) == _r(p)


def test_rectf_matches_qt(pure):
    for args in [(1.5, 2.0, 10.25, 4.0), (-2.5, 0.5, 3.5, 0.4), (0.49, 1.51, 2.0, 2.0)]:
        assert _r(QRectF(*args).toRect()) == _r(pure.QRectF(*args).toRect()), args
    q, p = QRectF(1.5, 2.0, 10.25, 4.0), pure.QRectF(1.5, 2.0, 10.25, 4.0)
    assert (q.right(), q.bottom()) == (p.right(), p.bottom())
    assert _p(q.center()) == _p(p.center())
    assert _r(q.toRect()) == _r(p.toRect())


def test_points_match_qt(pure):
    rng = random.Random(3)
    for _ in range(500):
        a = (rng.randint(-50, 50), rng.randint(-50, 50))
        b = (rng.randint(-50, 50), rng.randint(-50, 50))
        qa, qb, pa, pb = QPoint(*a), QPoint(*b), pure.QPoint(*a), pure.QPoint(*b)
        assert _p(qa + qb) == _p(pa + pb)
        assert _p(qa - qb) == _p(pa - pb)
        assert (qa - qb).manhattanLength() == (pa - pb).manhattanLength()
        assert (qa == qb) == (pa == pb)
    f = (1.25, -3.75)
    assert _p(QPointF(*f).toPoint()) == _p(pure.QPointF(*f).toPoint())
    assert _p(QPointF(QPoint(2, 3))) == _p(pure.QPointF(pure.QPoint(2, 3)))


@pytest.mark.parametrize(
    "args",
    [
        (),
        ("#1e90ff",),
        ("#abc",),
        ("red",),
        ("white",),
        ("not-a-colour",),
        (10, 20, 30),
        (1, 2, 3, 4),
    ],
)
def test_colours_match_qt(pure, args):
    q, p = QColor(*args), pure.QColor(*args)
    assert q.isValid() == p.isValid()
    if q.isValid():
        assert q.name() == p.name()
        assert (q.red(), q.green(), q.blue(), q.alpha()) == (
            p.red(),
            p.green(),
            p.blue(),
            p.alpha(),
        )
    assert pure.QColor(p) == p


def test_stand_ins_pickle(pure):
    for obj in (
        pure.QPoint(1, 2),
        pure.QPointF(1.5, 2),
        pure.QRect(1, 2, 3, 4),
        pure.QColor(1, 2, 3),
    ):
        assert pickle.loads(pickle.dumps(obj)) == obj


def _path_ops(lib, P):
    """The same drawing calls against either backend's QPainterPath."""
    path = lib.QPainterPath(P(0, 0))
    path.lineTo(P(0, 0))  # zero-length: Qt drops it
    path.lineTo(P(10, 0))
    path.quadTo(P(20, 0), P(20, 10))  # Qt stores a cubic
    path.cubicTo(P(20, 10), P(20, 10), P(20, 10))  # degenerate: dropped
    path.cubicTo(P(30, 20), P(0, 40), P(40, 40))
    path.moveTo(P(50, 50))
    path.moveTo(P(60, 60))  # a trailing moveTo is replaced
    path.lineTo(P(60, 80))
    return path


def test_painter_path_matches_qt(pure):
    from PyQt6 import QtGui

    real = _path_ops(QtGui, QPointF)
    stand_in = _path_ops(pure, pure.QPointF)
    assert stand_in.elementCount() == real.elementCount()
    for i in range(real.elementCount()):
        a, b = real.elementAt(i), stand_in.elementAt(i)
        assert (a.type.value, a.isCurveTo()) == (b.type, b.isCurveTo()), i
        assert (a.x, a.y) == pytest.approx((b.x, b.y)), i
    for t in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
        qa, qb = real.pointAtPercent(t), stand_in.pointAtPercent(t)
        assert math.dist((qa.x(), qa.y()), (qb.x(), qb.y())) < 0.5, t
    assert pure.QPainterPath(pure.QPointF(1, 1)).isEmpty()
    degenerate = pure.QPainterPath(pure.QPointF(5, 5))
    degenerate.cubicTo(pure.QPointF(5, 5), pure.QPointF(5, 5), pure.QPointF(5, 5))
    assert degenerate.isEmpty() and degenerate.elementCount() == 1


def test_integer_types_refuse_floats_like_pyqt6(pure):
    for bad in ((1.5, 2), (1.0, 2)):
        with pytest.raises(TypeError):
            QPoint(*bad)
        with pytest.raises(TypeError):
            pure.QPoint(*bad)
        with pytest.raises(TypeError):
            pure.QRect(*bad, 3, 4)
    # ...but scaling a point rounds, as Qt's operator* does.
    assert _p(QPoint(3, 5) * 1.5) == _p(pure.QPoint(3, 5) * 1.5)
    assert _p(QPoint(-3, 5) / 2) == _p(pure.QPoint(-3, 5) / 2)


def test_every_qt_colour_name_is_known(pure):
    for name in QColor.colorNames():
        assert pure.QColor(name).name() == QColor(name).name(), name
    assert pure.QColor("SteelBlue").name() == "#4682b4"
