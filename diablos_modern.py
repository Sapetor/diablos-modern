print("Running diablos_modern.py")
"""
Modern DiaBloS Application - Phase 1
Enhanced UI with modern styling, improved layout, and better user experience.

This is the main entry point for the modernized DiaBloS application.
"""

import sys
import os
import logging
import json
import warnings
import threading
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont


def _preload_heavy_modules():
    """Preload heavy modules in background thread to avoid first-simulation delay."""
    import time as _t
    _start = _t.time()
    try:
        # scipy.signal takes ~3s to import on first use
        from scipy import signal
        from scipy.integrate import solve_ivp  # noqa: F401
        # numpy should already be loaded, but ensure it's ready
        import numpy as np

        # CRITICAL: signal.cont2discrete has lazy init that takes ~3s on first call
        # Trigger it now with a dummy system to avoid delay during first simulation
        _dummy_A = np.array([[0, 1], [-1, -1]])
        _dummy_B = np.array([[0], [1]])
        _dummy_C = np.array([[1, 0]])
        _dummy_D = np.array([[0]])
        signal.cont2discrete((_dummy_A, _dummy_B, _dummy_C, _dummy_D), 0.01)

        print(f"[PRELOAD] Heavy modules loaded in {_t.time() - _start:.2f}s")
    except ImportError as e:
        print(f"[PRELOAD] Failed: {e}")
    except Exception as e:
        print(f"[PRELOAD] Warning: {e}")


# Start preloading immediately in background thread
print("[PRELOAD] Starting background import of scipy...")
_preload_thread = threading.Thread(target=_preload_heavy_modules, daemon=True)
_preload_thread.start()

# Import modern UI components
from modern_ui.main_window import ModernDiaBloSWindow
from modern_ui.styles.qss_styles import apply_modern_theme
from modern_ui.themes.theme_manager import theme_manager, ThemeType

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('diablos_modern.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Set specific logger levels - reduce verbosity for simulation
logging.getLogger('lib.lib').setLevel(logging.WARNING)
logging.getLogger('lib.improvements').setLevel(logging.WARNING)
logging.getLogger('lib.engine').setLevel(logging.WARNING)
logging.getLogger('lib.engine.simulation_engine').setLevel(logging.WARNING)
logging.getLogger('lib.plotting').setLevel(logging.WARNING)
logging.getLogger('modern_ui.widgets.modern_canvas').setLevel(logging.WARNING)
logging.getLogger('modern_ui.renderers').setLevel(logging.WARNING)


def setup_application():
    """Setup application-wide settings and styling."""
    # Silence PyQtGraph Qt version warning on older Qt (harmless noise)
    warnings.filterwarnings(
        "ignore",
        message="PyQtGraph supports Qt version >= 5.15",
        category=RuntimeWarning,
    )
    # Enable high DPI scaling
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("DiaBloS Modern")
    app.setApplicationVersion("2.0.0")
    app.setOrganizationName("DiaBloS Project")
    
    # Load config and set font
    try:
        with open('config/default_config.json', 'r') as f:
            config = json.load(f)
        scaling_factor = config.get('display', {}).get('scaling_factor', 1.0)
    except (FileNotFoundError, json.JSONDecodeError):
        scaling_factor = 1.0

    font = QFont("Segoe UI", int(10 * scaling_factor))
    font.setHintingPreference(QFont.PreferDefaultHinting)
    app.setFont(font)
    
    # Apply modern theme
    apply_modern_theme(app)
    
    return app


def main():
    """Main application entry point."""
    try:
        # Enable automatic scaling for high-DPI displays
        os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

        logger.info("Starting Modern DiaBloS Application - Phase 1")

        # Set the initial theme to light
        theme_manager.set_theme(ThemeType.LIGHT)

        # Setup application
        app = setup_application()

        # Get available screen geometry (excludes taskbar)
        screen_geometry = app.primaryScreen().availableGeometry()

        # Create main window with screen-aware sizing
        window = ModernDiaBloSWindow(screen_geometry)
        window.show()

        # Check for file argument (open diagram on startup)
        if len(sys.argv) > 1:
            file_path = sys.argv[1]
            if os.path.isfile(file_path) and file_path.endswith('.diablos'):
                logger.info(f"Opening diagram from command line: {file_path}")
                # Use QTimer to load after event loop starts
                from PyQt5.QtCore import QTimer
                def load_file():
                    try:
                        if hasattr(window.dsim, 'file_service'):
                            block_data = window.dsim.file_service.load(filepath=file_path)
                            window.dsim.deserialize(block_data)
                        window.canvas.update()
                        logger.info(f"Diagram loaded: {file_path}")
                    except Exception as e:
                        logger.error(f"Failed to load diagram: {e}")
                QTimer.singleShot(200, load_file)

        # Center window on screen
        window_size = window.geometry()
        x = screen_geometry.x() + (screen_geometry.width() - window_size.width()) // 2
        y = screen_geometry.y() + (screen_geometry.height() - window_size.height()) // 2
        window.move(x, y)
        
        logger.info("Modern DiaBloS started successfully")
        logger.info(f"Theme: {theme_manager.current_theme.value}")
        logger.info("Phase 1 Features:")
        logger.info("- Modern dark/light theming")
        logger.info("- Enhanced toolbar with zoom controls")
        logger.info("- Improved layout with splitter panels")
        logger.info("- Modern styling and typography")
        
        # Start application event loop
        exit_code = app.exec_()
        
        logger.info(f"Modern DiaBloS exiting with code: {exit_code}")
        return exit_code
        
    except Exception as e:
        logger.critical(f"Critical error in main: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
