"""Regression tests for ZoomPanManager zoom clamping.

Reported bug: on a Mac trackpad, a pinch/zoom gesture fires a rapid burst of
zoom events. With no bounds on ``zoom_factor`` the value ran toward 0, after
which ``screen_to_world`` divided by ~0 and produced world coordinates large
enough to overflow QPoint's 32-bit int, crashing the app.

These tests pin the clamping behavior so the runaway can't return.
"""

import pytest
from PyQt5.QtCore import QPoint

from modern_ui.managers.zoom_pan_manager import ZoomPanManager


class _FakeCanvas:
    """Minimal stand-in exposing only what ZoomPanManager touches."""

    def __init__(self, w=800, h=600):
        self._w = w
        self._h = h
        self.dsim = type("S", (), {"blocks_list": []})()
        self.zoom_factor = 1.0
        self.pan_offset = QPoint(0, 0)
        self.update_calls = 0

    def width(self):
        return self._w

    def height(self):
        return self._h

    def update(self):
        self.update_calls += 1


class _WheelEvent:
    def __init__(self, delta_y, modifiers):
        self._delta_y = delta_y
        self._modifiers = modifiers

    def modifiers(self):
        return self._modifiers

    def pos(self):
        return QPoint(400, 300)

    class _Angle:
        def __init__(self, y):
            self._y = y

        def y(self):
            return self._y

        def x(self):
            return 0

    def angleDelta(self):
        return self._Angle(self._delta_y)


@pytest.mark.unit
class TestZoomClamping:
    def test_repeated_zoom_out_hits_floor(self):
        """Hundreds of zoom-out steps must not drive the factor below MIN_ZOOM."""
        mgr = ZoomPanManager(_FakeCanvas())
        for _ in range(500):
            mgr.zoom_out()
        assert mgr.zoom_factor == pytest.approx(ZoomPanManager.MIN_ZOOM)
        # Canvas mirror stays in sync.
        assert mgr.canvas.zoom_factor == pytest.approx(ZoomPanManager.MIN_ZOOM)

    def test_repeated_zoom_in_hits_ceiling(self):
        mgr = ZoomPanManager(_FakeCanvas())
        for _ in range(500):
            mgr.zoom_in()
        assert mgr.zoom_factor == pytest.approx(ZoomPanManager.MAX_ZOOM)

    def test_set_zoom_clamps_both_ends(self):
        mgr = ZoomPanManager(_FakeCanvas())
        mgr.set_zoom(1e-9)
        assert mgr.zoom_factor == pytest.approx(ZoomPanManager.MIN_ZOOM)
        mgr.set_zoom(1e9)
        assert mgr.zoom_factor == pytest.approx(ZoomPanManager.MAX_ZOOM)

    def test_pan_offset_stays_in_int32_range_after_runaway(self):
        """The crash itself: tiny zoom -> huge world coords -> QPoint int overflow.

        With clamping, pan_offset must remain representable as a 32-bit int so
        QPoint construction never raises OverflowError.
        """
        mgr = ZoomPanManager(_FakeCanvas())
        for _ in range(1000):
            mgr.zoom_out()
        # QPoint stores 32-bit ints; assert well within range.
        assert abs(mgr.pan_offset.x()) < 2**31
        assert abs(mgr.pan_offset.y()) < 2**31

    def test_screen_to_world_safe_when_zoom_zero(self):
        """Defensive guard: a zero zoom factor must not divide-by-zero."""
        mgr = ZoomPanManager(_FakeCanvas())
        mgr.zoom_factor = 0.0
        # Should fall back to 1.0 internally rather than raise.
        world = mgr.screen_to_world(QPoint(100, 100))
        assert world == QPoint(100, 100)

    def test_wheel_zoom_out_burst_is_bounded(self):
        from PyQt5.QtCore import Qt

        mgr = ZoomPanManager(_FakeCanvas())
        for _ in range(500):
            mgr.handle_wheel_event(_WheelEvent(-120, Qt.ControlModifier))
        assert mgr.zoom_factor >= ZoomPanManager.MIN_ZOOM
        assert abs(mgr.pan_offset.x()) < 2**31


@pytest.mark.unit
class TestNativeGesture:
    def test_pinch_zoom_bounded(self):
        from PyQt5.QtCore import Qt

        class _Gesture:
            def __init__(self, value):
                self._value = value

            def gestureType(self):
                return Qt.ZoomNativeGesture

            def value(self):
                return self._value

            def pos(self):
                return QPoint(400, 300)

        mgr = ZoomPanManager(_FakeCanvas())
        # A long pinch-in burst.
        for _ in range(1000):
            handled = mgr.handle_native_gesture(_Gesture(-0.05))
            assert handled is True
        assert mgr.zoom_factor >= ZoomPanManager.MIN_ZOOM
        assert abs(mgr.pan_offset.x()) < 2**31

    def test_non_zoom_gesture_ignored(self):
        from PyQt5.QtCore import Qt

        class _Gesture:
            def gestureType(self):
                return Qt.SmartZoomNativeGesture

            def value(self):
                return 0.0

            def pos(self):
                return QPoint(0, 0)

        mgr = ZoomPanManager(_FakeCanvas())
        assert mgr.handle_native_gesture(_Gesture()) is False
