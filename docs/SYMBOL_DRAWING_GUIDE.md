# Symbol Drawing Guide

This guide outlines the standards for drawing block symbols in the `modern_ui` framework to ensure consistency and visual quality.

## Coordinate System

- **Normalized Coordinates**: All symbol paths should be defined using a normalized coordinate system from **0.0 to 1.0**.
    - `0.0, 0.0`: Top-Left corner of the block.
    - `1.0, 1.0`: Bottom-Right corner of the block.
    - `0.5, 0.5`: Center of the block.
- **Scaling**: The `draw_Block` method automatically scales these normalized coordinates to the actual block size (width, height) using a `QTransform`.
- **Margins**: Avoid drawing too close to the edges. A safe internal margin is generally **0.1** to **0.9**.

## drawing Implementation

Symbols are drawn using `QPainterPath` in `lib/simulation/block.py`.

### Code Structure

```python
elif self.block_fn == "BlockName":
    # 1. Define the path using normalized coordinates
    path.moveTo(0.1, 0.5)
    path.lineTo(0.9, 0.5)
    
    # 2. Add shapes or details
    # Arrows, Curves, etc.
```

### Best Practices

1.  **Use `moveTo` and `lineTo`**: Primary method for defining schematic lines.
2.  **Use `quadTo`**: For curved lines (e.g., sine waves, root locus branches).
3.  **Closed Loops**: For filled shapes (like the Hysteresis loop), ensure the path is closed or visually complete.
4.  **Avoid Absolute Pixel Values**: Never use hardcoded pixel values (e.g., `10`, `50`) for the path geometry. Always use relative float values or `self.width * factor`.

## Standard Symbols

### Arrows
small arrows should be drawn manually with 3 line segments to ensure they scale correctly with the path.
```python
# Right-pointing arrow
path.moveTo(0.45, 0.75); path.lineTo(0.45, 0.72); path.lineTo(0.51, 0.75); path.lineTo(0.45, 0.78); path.lineTo(0.45, 0.75)
```

### Text Labels
- Use `painter.drawText` for labels (like "PID", "1/s", "Ax+Bu").
- Use `QRect` with relative dimensions for positioning.
- Use `Qt.AlignCenter` for text alignment.

## Styling (handled by `draw_Block`)

- **Stroke Width**: Standard stroke width is **2px**. Selected blocks use **3px**.
- **Colors**:
    - **Stroke**: Dark Grey (`#1F2937`) or Theme Border Color.
    - **Fill**: Block Category Color (defined in `lib/simulation/block.py`).
- **Shadows**: Handled automatically by the base drawing logic.

## Examples

**Step Input**:
```python
path.moveTo(0.1, 0.7)
path.lineTo(0.5, 0.7)
path.lineTo(0.5, 0.3)
path.lineTo(0.9, 0.3)
```

**Gain (Triangle)**:
Handled as a special polygon case, not a path.

**Hysteresis (Loop)**:
```python
path.moveTo(0.15, 0.75); path.lineTo(0.75, 0.75) # Low -> High
path.lineTo(0.75, 0.25); path.lineTo(0.85, 0.25) # Jump
path.moveTo(0.85, 0.25); path.lineTo(0.25, 0.25) # High -> Low
path.lineTo(0.25, 0.75); path.lineTo(0.15, 0.75) # Jump
```
