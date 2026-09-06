# macOS Layout Fix for 13" Retina MacBooks

## Problem

On a 13" M1 MacBook Air (Retina display, `devicePixelRatio = 2.0`, ~1440×900
logical pixels), the application window opened cramped: panels did not fit well
by default and blocks clipped in the palette.

**Root cause:** the original layout code treated every display with
`devicePixelRatio > 1.25` as generic "high DPI" and applied aggressive scaling.
On smaller Retina screens the combined minimum-width requirements of the panels
exceeded the available logical width.

## Solution — centralized `PlatformConfig`

Platform detection and all platform-specific UI sizing live in a single module,
`modern_ui/platform_config.py`. A `PlatformConfig` object detects the display
once and exposes sizing as `@property` values; UI components read those values
instead of re-deriving platform logic.

```python
from modern_ui.platform_config import get_platform_config

config = get_platform_config()          # singleton, created on first call
min_width = config.left_panel_min_width  # platform-specific value
```

**`platform_config.py` is the source of truth for the numeric values.** Do not
hard-code panel/window/palette sizes elsewhere — read them from the config so a
change propagates to every component. The properties are grouped into window
sizing, left panel (block palette), canvas, right panel (properties), splitter
sizing, and palette-block metrics.

### Detection

```python
is_macos        = platform.system() == 'Darwin'
is_retina_small = is_macos and device_ratio >= 1.9 and logical_width < 1500
is_high_dpi     = device_ratio > 1.25
```

`is_retina_small` is the branch that fixes the 13" MacBook case; larger Retina
displays fall through to the generic high-DPI path.

### Consumers

- `modern_ui/main_window.py` — window sizing and the left / canvas / property
  panel minimums and splitter ratios all read from `PlatformConfig`, replacing
  the per-method detection blocks that previously duplicated the logic.
- `modern_ui/widgets/modern_palette.py` — block size and palette width come from
  `config.palette_block_size` / `config.calculate_palette_width()`, so block
  sizing stays consistent with panel sizing.

## Platform impact

| Platform | Example resolution | devicePixelRatio | `is_retina_small` | Effect |
|----------|--------------------|------------------|-------------------|--------|
| macOS 13" MBA | 1440×900 logical | 2.0 | Yes | Compact layout applied |
| macOS 27" iMac | 2560×1440 logical | 2.0 | No | Generic high-DPI path |
| Windows 1080p | 1920×1080 | 1.0 | No | Standard-DPI path |
| Ubuntu 1080p | 1920×1080 | 1.0 | No | Standard-DPI path |

Only the `is_retina_small` branch changes behavior; all other displays use the
same paths as before.

## Verification

```bash
python -m py_compile modern_ui/platform_config.py

python -c "from PyQt5.QtWidgets import QApplication; import sys; \
  app = QApplication(sys.argv); \
  from modern_ui.platform_config import get_platform_config; \
  print('is_retina_small =', get_platform_config().is_retina_small)"

python diablos_modern.py
```

On a 13" M1 MacBook Air the panels fit, blocks no longer clip in the palette,
and the window opens at a usable default size; Windows and Ubuntu 1080p layouts
are unchanged.

## Extending

To support a new display class (for example 4K), add the detection flag in
`PlatformConfig._detect_platform()` and branch on it inside the relevant sizing
`@property`. Because every component reads the config, only that one file needs
to change. Tunable knobs for the existing fix are the `is_retina_small`
thresholds (`device_ratio >= 1.9`, `logical_width < 1500`) and the per-property
return values.
