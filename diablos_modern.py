print("GEMINI: Running diablos_modern.py")
"""
Modern DiaBloS Application - Phase 1
Enhanced UI with modern styling, improved layout, and better user experience.

This is the main entry point for the modernized DiaBloS application.
"""

import sys
import logging
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

# Import modern UI components
from modern_ui.main_window import ModernDiaBloSWindow
from modern_ui.styles.qss_styles import apply_modern_theme
from modern_ui.themes.theme_manager import theme_manager, ThemeType

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('diablos_modern.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Set specific logger levels for debugging
logging.getLogger('lib.lib').setLevel(logging.INFO)
logging.getLogger('lib.improvements').setLevel(logging.INFO)


def setup_application():
    """Setup application-wide settings and styling."""
    # Enable high DPI scaling
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("DiaBloS Modern")
    app.setApplicationVersion("2.0.0")
    app.setOrganizationName("DiaBloS Project")
    
    # Set application font
    font = QFont("Segoe UI", 10)
    font.setHintingPreference(QFont.PreferDefaultHinting)
    app.setFont(font)
    
    # Apply modern theme
    apply_modern_theme(app)
    
    return app


def main():
    """Main application entry point."""
    try:
        logger.info("Starting Modern DiaBloS Application - Phase 1")
        
        # Set the initial theme to light
        theme_manager.set_theme(ThemeType.LIGHT)

        # Setup application
        app = setup_application()
        
        # Create and show main window
        window = ModernDiaBloSWindow()
        window.show()
        
        # Center window on screen
        screen = app.desktop().screenGeometry()
        window_size = window.geometry()
        x = (screen.width() - window_size.width()) // 2
        y = (screen.height() - window_size.height()) // 2
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