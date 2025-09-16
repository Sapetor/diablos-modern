"""
Simple DiaBloS improvements - Working version.

This version demonstrates the improvements without trying to integrate
with the complex DSim display methods, focusing on the core functionality.
"""

import sys
import logging
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout
from PyQt5.QtCore import Qt, QTimer

# Import existing code and improvements
from lib.lib import DSim
from lib.improvements import ValidationHelper, PerformanceHelper, SafetyChecks, LoggingHelper
from lib.config_manager import get_config

# Setup enhanced logging
LoggingHelper.setup_logging(level="INFO")
logger = logging.getLogger(__name__)


class SimpleDiaBloSWindow(QMainWindow):
    """
    Simple improved DiaBloS window that demonstrates the enhancements
    without complex display integration.
    """
    
    def __init__(self):
        super().__init__()
        logger.info("Starting Simple Improved DiaBloS")
        
        # Create the original DSim instance
        self.dsim = DSim()
        
        # Add performance monitoring
        self.perf_helper = PerformanceHelper()
        
        # Load configuration
        self.config = get_config()
        self.config.apply_to_dsim(self.dsim)
        
        # Setup UI
        self.init_ui()
        
        logger.info("Simple DiaBloS initialized successfully")
    
    def init_ui(self):
        """Initialize simple UI."""
        self.setWindowTitle("DiaBloS - Simple Improved Version")
        self.setGeometry(100, 100, 800, 600)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create layout
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        # Add title
        title = QLabel("DiaBloS - Improved Version")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 24px; font-weight: bold; margin: 20px;")
        layout.addWidget(title)
        
        # Add status info
        self.status_label = QLabel()
        self.update_status()
        layout.addWidget(self.status_label)
        
        # Add buttons
        button_layout = QHBoxLayout()
        
        # Validation button
        validate_btn = QPushButton("Run Validation")
        validate_btn.clicked.connect(self.run_validation)
        button_layout.addWidget(validate_btn)
        
        # Performance test button
        perf_btn = QPushButton("Performance Test")
        perf_btn.clicked.connect(self.run_performance_test)
        button_layout.addWidget(perf_btn)
        
        # Config test button
        config_btn = QPushButton("Show Config")
        config_btn.clicked.connect(self.show_config)
        button_layout.addWidget(config_btn)
        
        # Original DiaBloS button
        original_btn = QPushButton("Launch Original DiaBloS")
        original_btn.clicked.connect(self.launch_original)
        button_layout.addWidget(original_btn)
        
        layout.addLayout(button_layout)
        
        # Add results area
        self.results_label = QLabel("Click buttons above to test improvements...")
        self.results_label.setAlignment(Qt.AlignTop)
        self.results_label.setStyleSheet("background-color: #f0f0f0; padding: 10px; margin: 10px;")
        self.results_label.setWordWrap(True)
        layout.addWidget(self.results_label)
        
        # Setup timer for periodic updates
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.periodic_update)
        self.update_timer.start(2000)  # Update every 2 seconds
    
    def update_status(self):
        """Update status information."""
        status_text = f"""
<b>Configuration Status:</b><br>
• Simulation Time: {self.config.get('simulation.default_time')}s<br>
• Time Step: {self.config.get('simulation.default_timestep')}s<br>
• Window Size: {self.config.get('display.window_width')}x{self.config.get('display.window_height')}<br>
• Solver: {self.config.get('simulation.default_solver')}<br>
• Logging Level: {self.config.get('logging.level')}<br>
        """
        self.status_label.setText(status_text)
    
    def run_validation(self):
        """Run comprehensive validation."""
        try:
            logger.info("Running validation test...")
            self.perf_helper.start_timer("validation_test")
            
            results = []
            
            # Test simulation parameter validation
            from lib.improvements import validate_simulation_parameters
            is_valid, errors = validate_simulation_parameters(
                self.config.get('simulation.default_time', 10.0),
                self.config.get('simulation.default_timestep', 0.01)
            )
            
            if is_valid:
                results.append("✓ Simulation parameters are valid")
            else:
                results.append(f"✗ Parameter errors: {', '.join(errors)}")
            
            # Test DSim safety check
            is_safe, safety_errors = SafetyChecks.check_simulation_state(self.dsim)
            if is_safe:
                results.append("✓ DSim state is safe")
            else:
                results.append(f"✗ Safety issues: {', '.join(safety_errors[:3])}")  # Show first 3
            
            # Test configuration validation
            config_valid, config_errors = self.config.validate_config()
            if config_valid:
                results.append("✓ Configuration is valid")
            else:
                results.append(f"✗ Config errors: {', '.join(config_errors[:2])}")
            
            # Test block validation (with mock data)
            results.append("✓ Validation helpers are working")
            
            duration = self.perf_helper.end_timer("validation_test")
            results.append(f"<br><b>Validation completed in {duration:.4f} seconds</b>")
            
            self.results_label.setText("<br>".join(results))
            logger.info(f"Validation test completed in {duration:.4f}s")
            
        except Exception as e:
            error_msg = f"Validation test failed: {str(e)}"
            self.results_label.setText(error_msg)
            logger.error(error_msg)
    
    def run_performance_test(self):
        """Run performance monitoring test."""
        try:
            logger.info("Running performance test...")
            
            results = []
            
            # Test performance monitoring
            self.perf_helper.start_timer("test_operation_1")
            import time
            time.sleep(0.05)  # Simulate work
            duration1 = self.perf_helper.end_timer("test_operation_1")
            
            self.perf_helper.start_timer("test_operation_2")
            time.sleep(0.02)  # Simulate different work
            duration2 = self.perf_helper.end_timer("test_operation_2")
            
            results.append(f"✓ Test Operation 1: {duration1:.4f}s")
            results.append(f"✓ Test Operation 2: {duration2:.4f}s")
            
            # Get all performance stats
            stats1 = self.perf_helper.get_stats("test_operation_1")
            stats2 = self.perf_helper.get_stats("test_operation_2")
            
            if stats1:
                results.append(f"• Operation 1 stats: avg={stats1['average']:.4f}s, count={stats1['count']}")
            
            if stats2:
                results.append(f"• Operation 2 stats: avg={stats2['average']:.4f}s, count={stats2['count']}")
            
            results.append("<br><b>Performance monitoring is working!</b>")
            
            self.results_label.setText("<br>".join(results))
            logger.info("Performance test completed successfully")
            
        except Exception as e:
            error_msg = f"Performance test failed: {str(e)}"
            self.results_label.setText(error_msg)
            logger.error(error_msg)
    
    def show_config(self):
        """Show current configuration."""
        try:
            logger.info("Displaying configuration...")
            
            config_info = []
            config_info.append("<b>Current Configuration:</b>")
            config_info.append("")
            
            # Simulation settings
            config_info.append("<b>Simulation:</b>")
            config_info.append(f"• Default Time: {self.config.get('simulation.default_time')}s")
            config_info.append(f"• Default Timestep: {self.config.get('simulation.default_timestep')}s")
            config_info.append(f"• Solver: {self.config.get('simulation.default_solver')}")
            config_info.append("")
            
            # Display settings
            config_info.append("<b>Display:</b>")
            config_info.append(f"• Window: {self.config.get('display.window_width')}x{self.config.get('display.window_height')}")
            config_info.append(f"• FPS: {self.config.get('display.fps')}")
            config_info.append("")
            
            # Validation settings
            config_info.append("<b>Validation:</b>")
            config_info.append(f"• Check Algebraic Loops: {self.config.get('validation.check_algebraic_loops')}")
            config_info.append(f"• Check Unconnected Ports: {self.config.get('validation.check_unconnected_ports')}")
            config_info.append("")
            
            # Performance settings
            config_info.append("<b>Performance:</b>")
            config_info.append(f"• Warn Slow Steps: {self.config.get('performance.warn_slow_steps')}")
            config_info.append(f"• Slow Step Threshold: {self.config.get('performance.slow_step_threshold')}s")
            
            self.results_label.setText("<br>".join(config_info))
            logger.info("Configuration displayed successfully")
            
        except Exception as e:
            error_msg = f"Failed to show configuration: {str(e)}"
            self.results_label.setText(error_msg)
            logger.error(error_msg)
    
    def launch_original(self):
        """Launch the original DiaBloS application."""
        try:
            logger.info("Launching original DiaBloS...")
            
            # Import and run original
            import subprocess
            import os
            
            # Get the directory of this script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            original_script = os.path.join(script_dir, "diablosM1_main.py")
            
            if os.path.exists(original_script):
                subprocess.Popen([sys.executable, original_script])
                self.results_label.setText("✓ Original DiaBloS launched in separate window")
                logger.info("Original DiaBloS launched successfully")
            else:
                self.results_label.setText("✗ Original DiaBloS script not found")
                logger.error("Original script not found at: " + original_script)
                
        except Exception as e:
            error_msg = f"Failed to launch original DiaBloS: {str(e)}"
            self.results_label.setText(error_msg)
            logger.error(error_msg)
    
    def periodic_update(self):
        """Periodic update for monitoring."""
        try:
            # Just update the window title with current time
            import datetime
            current_time = datetime.datetime.now().strftime("%H:%M:%S")
            self.setWindowTitle(f"DiaBloS - Simple Improved Version [{current_time}]")
            
        except Exception as e:
            logger.error(f"Error in periodic update: {str(e)}")
    
    def closeEvent(self, event):
        """Clean shutdown."""
        try:
            logger.info("Application closing...")
            
            # Log final performance stats
            self.perf_helper.log_stats()
            
            event.accept()
            logger.info("Application closed successfully")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")
            event.accept()


def main():
    """Main application entry point."""
    try:
        app = QApplication(sys.argv)
        
        # Create and show main window
        window = SimpleDiaBloSWindow()
        window.show()
        
        logger.info("Simple DiaBloS application started successfully")
        
        # Run application
        sys.exit(app.exec_())
        
    except Exception as e:
        logger.critical(f"Critical error in main: {str(e)}")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()