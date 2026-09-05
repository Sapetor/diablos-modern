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

#: Loggers whose INFO chatter is per-simulation-step or per-frame and would
#: drown the log; they are held at WARNING.
#:
#: THIS IS THE SINGLE SOURCE OF TRUTH for that list. It used to be duplicated
#: as seven verbose entries in ``config/logging.json``'s ``loggers`` section,
#: which drifted from this one. The JSON now ships an empty ``loggers`` map and
#: this list is applied on top of whatever ``dictConfig`` set up -- an explicit
#: entry in the JSON still wins, so per-logger overrides remain possible.
QUIET_LOGGERS = (
    "lib.lib",
    "lib.improvements",
    "lib.engine",
    "lib.engine.simulation_engine",
    "lib.plotting",
    "modern_ui.widgets.modern_canvas",
    "modern_ui.renderers",
)


def _apply_quiet_loggers(configured: Optional[dict] = None) -> None:
    """Hold every ``QUIET_LOGGERS`` entry at WARNING.

    Args:
        configured: the ``loggers`` mapping from a loaded logging config, if
            any. Names present there were configured explicitly and are left
            alone, so the JSON can still override a default.
    """
    configured = configured or {}
    for logger_name in QUIET_LOGGERS:
        if logger_name in configured:
            continue
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def setup_logging(config_path: Optional[str] = None) -> None:
    """
    Configure logging from a JSON config file or use defaults.

    Args:
        config_path: Path to logging config JSON file.
                    Defaults to 'config/logging.json' relative to project root.
    """
    if config_path is None:
        from lib.app_paths import resource_path

        config_path = resource_path("config/logging.json")

    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            # Fix relative log file paths for frozen (PyInstaller) mode
            for handler in config.get("handlers", {}).values():
                fn = handler.get("filename")
                if fn:
                    handler["filename"] = _get_log_file_path(fn)
            logging.config.dictConfig(config)
            _apply_quiet_loggers(config.get("loggers"))
            return
        except (json.JSONDecodeError, ValueError, KeyError, OSError) as e:
            print(f"Warning: Failed to load logging config from {config_path}: {e}")
            print("Falling back to default logging configuration.")
            # Fallback to basic configuration, then re-log the original error.
            _setup_default_logging()
            logging.getLogger(__name__).warning(
                "Failed to load logging config from %s: %s; using default logging configuration.",
                config_path,
                e,
            )
            return

    # Fallback to basic configuration
    _setup_default_logging()


def _get_log_file_path(filename: str) -> str:
    """Resolve log file path, using a writable location in frozen mode."""
    if getattr(sys, "frozen", False) and not os.path.isabs(filename):
        if sys.platform == "darwin":
            log_dir = os.path.expanduser("~/Library/Logs/DiaBloS")
        else:
            log_dir = os.path.dirname(sys.executable)
        os.makedirs(log_dir, exist_ok=True)
        return os.path.join(log_dir, filename)
    return filename


def _setup_default_logging() -> None:
    """Setup default logging configuration if config file is unavailable."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(_get_log_file_path("diablos_modern.log")),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Reduce verbosity for the per-step / per-frame loggers (see QUIET_LOGGERS).
    _apply_quiet_loggers()


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name: Logger name, typically __name__ from the calling module.

    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name)
