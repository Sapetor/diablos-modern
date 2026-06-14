"""Tests for the idle-OFF glow animation timer and its renderer hooks.

A single ~30fps ``QTimer`` on ``ModernCanvas`` drives a subtle sine-modulated
glow on the hovered port / active wire / running simulation. The whole point is
that it stays STOPPED while the canvas is idle so a large diagram never repaints
continuously. These tests drive the gated state directly and assert:

* the timer is inactive on a freshly-built, idle canvas;
* setting a hovered port, a connection-drag state, or a running simulation each
  starts the timer, and clearing every gated state stops it again;
* the phase accessor / ``glow_pulse_alpha`` modulation behaves (stable alpha at
  rest, clamped, sane swing), and the renderers accept the pulse hook cleanly;
* the toolbar's running status-dot pulse timer is gated strictly to 'running'.

The canvas is built against a REAL ``DSim`` (same approach as
tests/modern_ui/test_empty_canvas_hint.py) so the real wiring runs.

Run with:
    QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python -m pytest \
        tests/modern_ui/test_canvas_glow_animation.py -p no:cacheprovider -o addopts=""
"""

import math

import pytest
from PyQt5.QtCore import QPoint
from PyQt5.QtGui import QPixmap, QPainter, QColor

from lib.lib import DSim
from modern_ui.widgets.modern_canvas import ModernCanvas
from modern_ui.renderers.canvas_renderer import CanvasRenderer
from modern_ui.renderers.connection_renderer import ConnectionRenderer
from modern_ui.widgets.modern_toolbar import _StateDot


@pytest.fixture(autouse=True)
def _qt(qapp):
    """Bind the shared session QApplication (from conftest) for every test."""
    return qapp


@pytest.fixture
def canvas(qapp):
    """Build a real ModernCanvas wrapping a real DSim."""
    dsim = DSim()
    c = ModernCanvas(dsim)
    c.resize(800, 600)
    return c


class _StubPortBlock:
    """Minimal block exposing only the coords draw_hover_effects reads."""

    def __init__(self):
        self.in_coords = [QPoint(40, 40)]
        self.out_coords = [QPoint(120, 40)]
        self.selected = False


# ---------------------------------------------------------------------------
# Timer gating — the load-bearing behaviour
# ---------------------------------------------------------------------------

class TestAnimationTimerGating:
    def test_idle_canvas_timer_inactive(self, canvas):
        # Freshly built, nothing hovered / dragging / running: must be OFF so an
        # idle (possibly huge) diagram never repaints continuously.
        assert canvas.hovered_port is None
        assert not canvas.line_creation_state
        assert not canvas.is_simulation_running()
        assert not canvas._animation_timer.isActive()

    def test_hovered_port_starts_and_clearing_stops(self, canvas):
        canvas.hovered_port = (_StubPortBlock(), 0, True)
        assert canvas._animation_timer.isActive()

        canvas.hovered_port = None
        assert not canvas._animation_timer.isActive()

    def test_connection_drag_starts_and_clearing_stops(self, canvas):
        canvas.line_creation_state = 'start'
        assert canvas._animation_timer.isActive()

        canvas.line_creation_state = None
        assert not canvas._animation_timer.isActive()

    def test_running_simulation_starts_and_stopping_stops(self, canvas, monkeypatch):
        # Drive the gate the way the simulation_status_changed slot does.
        running = {'v': True}
        monkeypatch.setattr(canvas, 'is_simulation_running', lambda: running['v'])

        canvas._evaluate_animation_state()
        assert canvas._animation_timer.isActive()

        running['v'] = False
        canvas._evaluate_animation_state()
        assert not canvas._animation_timer.isActive()

    def test_timer_stays_on_until_all_states_clear(self, canvas, monkeypatch):
        # Overlapping gated states: clearing one must not stop the pulse while
        # another still holds.
        running = {'v': True}
        monkeypatch.setattr(canvas, 'is_simulation_running', lambda: running['v'])
        canvas._evaluate_animation_state()
        canvas.hovered_port = (_StubPortBlock(), 0, True)
        assert canvas._animation_timer.isActive()

        canvas.hovered_port = None  # sim still running
        assert canvas._animation_timer.isActive()

        running['v'] = False
        canvas._evaluate_animation_state()  # now everything clear
        assert not canvas._animation_timer.isActive()

    def test_timer_parented_to_canvas(self, canvas):
        # Parenting is what prevents a leak on widget destruction.
        assert canvas._animation_timer.parent() is canvas

    def test_interval_is_about_30fps(self, canvas):
        assert canvas._animation_timer.interval() == ModernCanvas._ANIM_INTERVAL_MS
        assert 20 <= ModernCanvas._ANIM_INTERVAL_MS <= 50


# ---------------------------------------------------------------------------
# Phase + alpha modulation
# ---------------------------------------------------------------------------

class TestPhaseAndPulseAlpha:
    def test_phase_starts_at_zero(self, canvas):
        assert canvas.animation_phase == 0.0

    def test_tick_advances_phase_and_wraps(self, canvas):
        canvas._on_animation_tick()
        assert math.isclose(canvas.animation_phase, ModernCanvas._ANIM_PHASE_STEP)
        # Many ticks keep the phase bounded to [0, 2π).
        for _ in range(200):
            canvas._on_animation_tick()
        assert 0.0 <= canvas.animation_phase < 2 * math.pi

    def test_pulse_alpha_at_rest_equals_base(self, canvas):
        # phase 0 -> sin(0)=0 -> exactly the base alpha (a stable, non-pulsing glow).
        assert canvas.animation_phase == 0.0
        assert canvas.glow_pulse_alpha(100) == 100

    def test_pulse_alpha_swings_with_phase(self, canvas):
        canvas._animation_phase = math.pi / 2  # sin = +1 -> brightest
        assert canvas.glow_pulse_alpha(100, depth=0.4) == 140
        canvas._animation_phase = 3 * math.pi / 2  # sin = -1 -> dimmest
        assert canvas.glow_pulse_alpha(100, depth=0.4) == 60

    def test_pulse_alpha_is_clamped_to_byte_range(self, canvas):
        canvas._animation_phase = math.pi / 2
        # A huge base + depth would overflow 255 without clamping.
        assert canvas.glow_pulse_alpha(250, depth=1.0) == 255
        canvas._animation_phase = 3 * math.pi / 2
        assert canvas.glow_pulse_alpha(10, depth=5.0) == 0


# ---------------------------------------------------------------------------
# Renderer hooks accept the pulse callable
# ---------------------------------------------------------------------------

class TestRendererPulseHooks:
    def _painter(self):
        pixmap = QPixmap(200, 200)
        pixmap.fill(QColor(0, 0, 0))
        return QPainter(pixmap), pixmap

    def test_canvas_renderer_accepts_pulse_alpha(self, canvas):
        renderer = CanvasRenderer()
        painter, _pm = self._painter()
        try:
            # Should run cleanly with a pulse callable supplied for a hovered port.
            renderer.draw_hover_effects(
                painter,
                hovered_port=(_StubPortBlock(), 0, True),
                pulse_alpha=canvas.glow_pulse_alpha,
            )
        finally:
            painter.end()

    def test_canvas_renderer_pulse_alpha_optional(self):
        # Still usable standalone with no pulse callable (flat base alpha).
        renderer = CanvasRenderer()
        painter, _pm = self._painter()
        try:
            renderer.draw_hover_effects(painter, hovered_port=(_StubPortBlock(), 0, False))
        finally:
            painter.end()

    def test_connection_renderer_glow_alpha_uses_callable(self):
        renderer = ConnectionRenderer()
        # Default: no pulse -> flat base alpha.
        assert renderer._glow_alpha(40) == 40
        # With a callable set (as the canvas does), the value is routed through it.
        renderer.pulse_alpha = lambda base: base + 7
        assert renderer._glow_alpha(40) == 47

    def test_canvas_wires_connection_renderer_pulse(self, canvas):
        # The canvas hands its glow_pulse_alpha to its connection renderer so the
        # active-wire glow can breathe with the shared phase.
        assert canvas.connection_renderer.pulse_alpha == canvas.glow_pulse_alpha


# ---------------------------------------------------------------------------
# Toolbar status-dot pulse — gated strictly to 'running'
# ---------------------------------------------------------------------------

class TestStatusDotPulseGating:
    def test_idle_dot_pulse_inactive(self, qapp):
        dot = _StateDot()
        assert not dot._pulse_timer.isActive()

    def test_running_starts_pulse(self, qapp):
        dot = _StateDot()
        dot.set_state('running')
        assert dot._pulse_timer.isActive()

    def test_non_running_states_stop_pulse(self, qapp):
        dot = _StateDot()
        dot.set_state('running')
        assert dot._pulse_timer.isActive()
        for s in ('paused', 'error', 'idle'):
            dot.set_state(s)
            assert not dot._pulse_timer.isActive(), s

    def test_glow_alpha_flat_when_idle_pulses_when_running(self, qapp):
        dot = _StateDot()
        # Not running -> flat base alpha regardless of stored phase.
        dot._pulse_phase = math.pi / 2
        assert dot._glow_alpha() == _StateDot._GLOW_BASE_ALPHA
        # Running -> phase-modulated, clamped to byte range.
        dot.set_state('running')
        dot._pulse_phase = math.pi / 2
        assert 0 <= dot._glow_alpha() <= 255
        assert dot._glow_alpha() >= _StateDot._GLOW_BASE_ALPHA

    def test_pulse_timer_parented(self, qapp):
        dot = _StateDot()
        assert dot._pulse_timer.parent() is dot
