"""
Centralized logging configuration for DiaBloS.
Loads logging settings from config/logging.json or falls back to defaults.
"""

import logging
import logging.config
import json
import os
import sys
from typing import Optional


def setup_logging(config_path: Optional[str] = None) -> None:
    """
    Configure logging from a JSON config file or use defaults.

    Args:
        config_path: Path to logging config JSON file.
                    Defaults to 'config/logging.json' relative to project root.
    """
    if config_path is None:
        # Find config relative to this file's location
        lib_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(lib_dir)
        config_path = os.path.join(project_root, 'config', 'logging.json')

    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logging.config.dictConfig(config)
            return
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"Warning: Failed to load logging config from {config_path}: {e}")
            print("Falling back to default logging configuration.")

    # Fallback to basic configuration
    _setup_default_logging()


def _setup_default_logging() -> None:
    """Setup default logging configuration if config file is unavailable."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('diablos_modern.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Set specific logger levels - reduce verbosity for simulation
    quiet_loggers = [
        'lib.lib',
        'lib.improvements',
        'lib.engine',
        'lib.engine.simulation_engine',
        'lib.plotting',
        'modern_ui.widgets.modern_canvas',
        'modern_ui.renderers',
    ]
    for logger_name in quiet_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name: Logger name, typically __name__ from the calling module.

    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name)
