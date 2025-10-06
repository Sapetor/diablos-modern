"""
Simple configuration management for DiaBloS.

This module provides easy configuration management that can be integrated
with the existing codebase without breaking changes.
"""

import json
import os
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigManager:
    """Simple configuration manager for DiaBloS settings."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "config/default_config.json"
        self._config: Dict[str, Any] = {}
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from file."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    self._config = json.load(f)
                logger.info(f"Configuration loaded from {self.config_file}")
            else:
                logger.warning(f"Configuration file {self.config_file} not found, using defaults")
                self._config = self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self._config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "simulation": {
                "default_time": 10.0,
                "default_timestep": 0.01,
                "max_simulation_time": 1000.0,
                "min_timestep": 1e-6,
                "max_timestep": 1.0,
                "default_solver": "SOLVE_IVP",
                "available_solvers": ["FWD_RECT", "BWD_RECT", "TUSTIN", "RK45", "SOLVE_IVP"]
            },
            "display": {
                "window_width": 1280,
                "window_height": 770,
                "fps": 60,
                "canvas_top_limit": 60,
                "canvas_left_limit": 200,
                "default_block_width": 120,
                "default_block_height": 60
            },
            "validation": {
                "check_algebraic_loops": True,
                "check_unconnected_ports": True,
                "warn_orphaned_blocks": True,
                "max_simulation_steps": 1000000
            },
            "logging": {
                "level": "INFO",
                "log_to_file": True,
                "log_file": "diablos.log",
                "log_performance": True,
                "log_simulation_steps": False
            },
            "performance": {
                "warn_slow_steps": True,
                "slow_step_threshold": 0.1,
                "warn_slow_paint": True,
                "slow_paint_threshold": 0.05,
                "enable_profiling": False
            },
            "external_functions": {
                "directory": "external",
                "auto_reload": False,
                "validate_on_load": True
            }
        }
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Path to the configuration key (e.g., "simulation.default_time")
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        try:
            keys = key_path.split('.')
            value = self._config
            
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return default
            
            return value
            
        except Exception as e:
            logger.error(f"Error getting config key {key_path}: {e}")
            return default
    
    def set(self, key_path: str, value: Any) -> bool:
        """
        Set configuration value using dot notation.
        
        Args:
            key_path: Path to the configuration key
            value: Value to set
            
        Returns:
            True if successful, False otherwise
        """
        try:
            keys = key_path.split('.')
            config_ref = self._config
            
            # Navigate to the parent of the target key
            for key in keys[:-1]:
                if key not in config_ref:
                    config_ref[key] = {}
                config_ref = config_ref[key]
            
            # Set the final key
            config_ref[keys[-1]] = value
            return True
            
        except Exception as e:
            logger.error(f"Error setting config key {key_path}: {e}")
            return False
    
    def save(self) -> bool:
        """Save current configuration to file."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            
            with open(self.config_file, 'w') as f:
                json.dump(self._config, f, indent=2)
            
            logger.info(f"Configuration saved to {self.config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
    
    def apply_to_dsim(self, dsim_instance: Any) -> None:
        """Apply configuration settings to existing DSim instance."""
        try:
            # Apply display settings
            if hasattr(dsim_instance, 'SCREEN_WIDTH'):
                dsim_instance.SCREEN_WIDTH = self.get("display.window_width", 1280)
            
            if hasattr(dsim_instance, 'SCREEN_HEIGHT'):
                dsim_instance.SCREEN_HEIGHT = self.get("display.window_height", 770)
            
            if hasattr(dsim_instance, 'FPS'):
                dsim_instance.FPS = self.get("display.fps", 60)
            
            if hasattr(dsim_instance, 'canvas_top_limit'):
                dsim_instance.canvas_top_limit = self.get("display.canvas_top_limit", 60)
            
            if hasattr(dsim_instance, 'canvas_left_limit'):
                dsim_instance.canvas_left_limit = self.get("display.canvas_left_limit", 200)
            
            # Apply simulation settings
            if hasattr(dsim_instance, 'sim_time'):
                dsim_instance.sim_time = self.get("simulation.default_time", 10.0)
            
            if hasattr(dsim_instance, 'sim_dt'):
                dsim_instance.sim_dt = self.get("simulation.default_timestep", 0.01)
            
            logger.info("Configuration applied to DSim instance")
            
        except Exception as e:
            logger.error(f"Error applying configuration to DSim: {e}")
    
    def validate_config(self) -> tuple[bool, list[str]]:
        """Validate current configuration."""
        errors = []
        
        try:
            # Validate simulation settings
            sim_time = self.get("simulation.default_time")
            if sim_time is not None and sim_time <= 0:
                errors.append("Simulation default_time must be positive")
            
            sim_dt = self.get("simulation.default_timestep")
            if sim_dt is not None and sim_dt <= 0:
                errors.append("Simulation default_timestep must be positive")
            
            if sim_time and sim_dt and sim_dt >= sim_time:
                errors.append("Simulation timestep cannot be larger than simulation time")
            
            # Validate display settings
            width = self.get("display.window_width")
            if width is not None and width < 800:
                errors.append("Window width should be at least 800 pixels")
            
            height = self.get("display.window_height")
            if height is not None and height < 600:
                errors.append("Window height should be at least 600 pixels")
            
            fps = self.get("display.fps")
            if fps is not None and (fps < 1 or fps > 120):
                errors.append("FPS should be between 1 and 120")
            
            # Validate solver
            solver = self.get("simulation.default_solver")
            available_solvers = self.get("simulation.available_solvers", [])
            if solver and available_solvers and solver not in available_solvers:
                errors.append(f"Invalid solver '{solver}', must be one of {available_solvers}")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            logger.error(f"Error validating configuration: {e}")
            return False, [f"Configuration validation error: {str(e)}"]
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration as dictionary."""
        return self._config.copy()
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to defaults."""
        self._config = self._get_default_config()
        logger.info("Configuration reset to defaults")


# Global configuration instance for easy access
_global_config: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    """Get global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = ConfigManager()
    return _global_config


def reload_config(config_file: Optional[str] = None) -> ConfigManager:
    """Reload configuration from file."""
    global _global_config
    _global_config = ConfigManager(config_file)
    return _global_config