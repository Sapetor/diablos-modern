# Memory

## Property Editor Upgrade (Feb 2026)
- Rewrote `modern_ui/widgets/property_editor.py` with 7 UI improvements
- Features: generic choices, param tooltips, block header, collapsible sections, slider+spinbox, reset buttons, inline validation
- Tests expanded from 4 to 25 in `tests/test_property_editor.py`
- Key lesson: Qt `isVisible()` checks parent chain; use `isHidden()` for unit tests on non-shown widgets

## Qt Testing Gotchas
- `widget.isVisible()` returns False if any ancestor is not shown (even if widget itself is set visible)
- `widget.isHidden()` checks only the widget's own visibility flag - use this in unit tests
- `findChild`/`findChildren` search recursively through all descendants
