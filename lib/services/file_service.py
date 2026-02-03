"""
FileService - File I/O operations for DiaBloS.
Handles saving and loading diagram files.
"""

import json
import os
import logging
from typing import Dict, Optional, Any
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QColor
from PyQt5.QtCore import QRect

logger = logging.getLogger(__name__)


class FileService:
    """
    Service for saving and loading simulation diagrams.
    Handles JSON serialization/deserialization of blocks, lines, and metadata.

    Attributes:
        model: SimulationModel instance containing diagram data
        filename: Current filename for save/load operations
        SCREEN_WIDTH: Default window width for saved diagrams
        SCREEN_HEIGHT: Default window height for saved diagrams
    """

    def __init__(self, model: Any) -> None:
        """
        Initialize file service.

        Args:
            model: SimulationModel instance containing blocks and lines
        """
        self.model = model
        self.filename: str = 'data.dat'
        self.SCREEN_WIDTH: int = 1280
        self.SCREEN_HEIGHT: int = 770

    def serialize(self, modern_ui_data: Optional[Dict[str, Any]] = None, sim_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Serialize the current diagram state into a dictionary.
        
        Args:
            modern_ui_data: Additional UI state data
            sim_params: Simulation parameters
            
        Returns:
            Dictionary containing the diagram data.
        """
        # Prepare simulation data
        sim_params = sim_params or {}
        init_dict = {
            "wind_width": self.SCREEN_WIDTH,
            "wind_height": self.SCREEN_HEIGHT,
            "fps": 60,
            "sim_time": sim_params.get('sim_time', 1.0),
            "sim_dt": sim_params.get('sim_dt', 0.01),
            "sim_trange": sim_params.get('plot_trange', 100)
        }

        # Serialize blocks
        blocks_dict = []
        for block in self.model.blocks_list:
            block_dict = {
                "block_fn": block.block_fn,
                "sid": block.sid,
                "username": block.username,
                "coords_left": block.left,
                "coords_top": block.top,
                "coords_width": block.width,
                "coords_height": block.height,
                "coords_height_base": block.height_base,
                "in_ports": block.in_ports,
                "out_ports": block.out_ports,
                "dragging": block.dragging,
                "selected": block.selected,
                "b_color": block.b_color.name(),
                "b_type": block.b_type,
                "io_edit": block.io_edit,
                "fn_name": block.fn_name,
                "params": block.saving_params(),
                "external": block.external,
                "flipped": block.flipped
            }
            blocks_dict.append(block_dict)

        # Serialize lines
        lines_dict = []
        for line in self.model.line_list:
            line_dict = {
                "name": line.name,
                "sid": line.sid,
                "srcblock": line.srcblock,
                "srcport": line.srcport,
                "dstblock": line.dstblock,
                "dstport": line.dstport,
                "points": [(p.x(), p.y()) for p in line.points],
                "cptr": getattr(line, 'cptr', 0),
                "selected": line.selected
            }
            lines_dict.append(line_dict)

        # Assemble final data structure
        main_dict = {
            "sim_data": init_dict,
            "blocks_data": blocks_dict,
            "lines_data": lines_dict,
            "version": "2.0"
        }

        if modern_ui_data:
            main_dict["modern_ui_data"] = modern_ui_data
            
        return main_dict

    def save_to_file(self, data: Dict[str, Any], filename: str) -> bool:
        """
        Write serialized data to a file.
        
        Args:
            data: Diagram data dict
            filename: Path to save
        """
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            with open(filename, 'w') as fp:
                json.dump(data, fp, indent=4)
                
            self.filename = os.path.basename(filename)
            self.model.dirty = False
            logger.info(f"SAVED AS {filename}")
            return True
        except Exception as e:
            logger.error(f"Error saving file {filename}: {e}")
            return False

    def save(self, autosave: bool = False, modern_ui_data: Optional[Dict[str, Any]] = None,
             sim_params: Optional[Dict[str, Any]] = None, filepath: Optional[str] = None) -> int:
        """
        Legacy save method. Wraps serialize and save_to_file.
        """
        if not autosave:
            if filepath:
                file = filepath
            else:
                options = QFileDialog.Options()
            initial_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'saves')
            file, _ = QFileDialog.getSaveFileName(
                None,
                "Save File",
                os.path.join(initial_dir, self.filename),
                "Data Files (*.dat);;All Files (*)",
                options=options
            )

            if not file:
                return 1
            if not file.lower().endswith('.dat'):
                file += '.dat'
        else:
            # Autosave to saves/ directory
            if filepath:
                 file = filepath
            elif '_AUTOSAVE' not in self.filename:
                file = f'saves/{self.filename[:-4]}_AUTOSAVE.dat'
            else:
                file = f'saves/{self.filename}'
        
        # Use new methods
        data = self.serialize(modern_ui_data, sim_params)
        success = self.save_to_file(data, file)
        
        return 0 if success else 1

    def load(self, filepath: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Load diagram from a JSON file.

        Args:
            filepath: Path to file to load. If None, shows file dialog

        Returns:
            Dictionary containing:
                - sim_data: Simulation parameters (sim_time, sim_dt, etc.)
                - blocks_data: List of serialized block dictionaries
                - lines_data: List of serialized connection dictionaries
                - version: File format version string
                - modern_ui_data: Optional UI state data
            Returns None if user cancelled or error occurred
        """
        if filepath is None:
            options = QFileDialog.Options()
            initial_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'saves')
            filepath, _ = QFileDialog.getOpenFileName(
                None,
                "Open File",
                initial_dir,
                "Data Files (*.dat);;All Files (*)",
                options=options
            )

            if not filepath:
                return None

        try:
            with open(filepath, 'r') as fp:
                data = json.load(fp)

            self.filename = os.path.basename(filepath)
            logger.info(f"LOADED FROM {filepath}")
            return data

        except Exception as e:
            logger.error(f"Error loading file {filepath}: {e}")
            return None

    def apply_loaded_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply loaded data to the model, reconstructing the diagram.

        Args:
            data: Dictionary containing sim_data, blocks_data, lines_data from loaded file

        Returns:
            Dictionary with extracted simulation parameters:
                - sim_time: Total simulation duration
                - sim_dt: Simulation time step
                - plot_trange: Plot time range for scope displays
        """
        # Clear existing data
        self.model.clear_all()

        # Extract simulation parameters
        sim_data = data.get('sim_data', {})
        sim_params = {
            'sim_time': sim_data.get('sim_time', 1.0),
            'sim_dt': sim_data.get('sim_dt', 0.01),
            'plot_trange': sim_data.get('sim_trange', 100)
        }

        # Recreate blocks
        blocks_data = data.get('blocks_data', [])
        for block_data in blocks_data:
            # Find matching menu block
            menu_block = None
            for mb in self.model.menu_blocks:
                if mb.block_fn == block_data['block_fn']:
                    menu_block = mb
                    break

            if menu_block is None:
                logger.warning(f"Block type {block_data['block_fn']} not found in menu_blocks")
                continue

            # Create block rect
            block_rect = QRect(
                block_data['coords_left'],
                block_data['coords_top'],
                block_data['coords_width'],
                block_data['coords_height']
            )

            # Import DBlock here to avoid circular import
            from lib.simulation.block import DBlock

            # Create block instance
            # Load params - use class method if available, otherwise pass directly
            params = block_data.get('params', {})

            # Use fn_name from menu_block to ensure we have the correct function name,
            # not the potentially outdated one from saved file
            block = DBlock(
                block_data['block_fn'],
                block_data['sid'],
                block_rect,
                QColor(block_data['b_color']),
                block_data['in_ports'],
                block_data['out_ports'],
                block_data.get('b_type', 2),
                block_data.get('io_edit', 'none'),
                menu_block.fn_name,  # Use correct fn_name from menu_block
                params,
                block_data.get('external', False),
                username=block_data.get('username', ''),
                block_class=menu_block.block_class,
                colors=self.model.colors
            )

            block.flipped = block_data.get('flipped', False)
            block.height_base = block_data.get('coords_height_base', block.height)

            self.model.blocks_list.append(block)

        # Build username→name and name→name maps for flexible line references
        block_name_map = {}
        for block in self.model.blocks_list:
            # Map block.name to itself
            block_name_map[block.name] = block.name
            # Map username to block.name (if username exists)
            if block.username:
                block_name_map[block.username] = block.name

        # Recreate lines
        from lib.simulation.connection import DLine
        lines_data = data.get('lines_data', [])
        for line_data in lines_data:
            points = [tuple(p) if isinstance(p, list) else p for p in line_data['points']]

            # Resolve srcblock and dstblock - allow username OR block.name
            srcblock = line_data['srcblock']
            dstblock = line_data['dstblock']
            srcblock = block_name_map.get(srcblock, srcblock)
            dstblock = block_name_map.get(dstblock, dstblock)

            line = DLine(
                line_data['sid'],
                srcblock,
                line_data['srcport'],
                dstblock,
                line_data['dstport'],
                points
            )
            self.model.line_list.append(line)

        # Update line positions
        self.model.update_lines()
        self.model.dirty = False

        return sim_params
