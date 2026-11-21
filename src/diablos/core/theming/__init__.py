"""Theming primitives for DiaBloS Modern.

Houses the design-token / theme system (``theme_manager``) in the core ``lib``
package so that ``lib`` modules never need to import from ``modern_ui``. The
old import path ``diablos.gui.themes.theme_manager`` remains valid via a
backward-compat shim that re-exports from here.
"""
