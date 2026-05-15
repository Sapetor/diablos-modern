"""
FileService - File I/O operations for DiaBloS.
Handles saving and loading diagram files.
"""

import json
import os
import sys
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

        # Serialize blocks (recurses into Subsystems via _serialize_block)
        blocks_dict = [self._serialize_block(block) for block in self.model.blocks_list]

        # Serialize lines
        lines_dict = [self._serialize_line(line) for line in self.model.line_list]

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

    def _serialize_block(self, block: Any) -> Dict[str, Any]:
        """
        Serialize a single block to a dict. Recurses into Subsystems
        so their internal blocks, lines, ports, and ports_map are preserved.
        """
        block_dict = {
            "block_fn": block.block_fn,
            "sid": block.sid,
            "name": block.name,
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
            "flipped": block.flipped,
        }

        # Subsystem-specific: persist nested structure so reload can rebuild it.
        # Without this, reopened diagrams would have empty subsystems and the
        # flattener would silently produce no primitives from them.
        if block.block_fn == 'Subsystem' or hasattr(block, 'sub_blocks'):
            block_dict['sub_blocks'] = [
                self._serialize_block(child) for child in getattr(block, 'sub_blocks', [])
            ]
            block_dict['sub_lines'] = [
                self._serialize_line(child_line) for child_line in getattr(block, 'sub_lines', [])
            ]
            block_dict['ports'] = getattr(block, 'ports', {}) or {}
            # JSON only allows string keys; ports_map uses int port indices,
            # so coerce here and convert back on load.
            ports_map = getattr(block, 'ports_map', {}) or {}
            block_dict['ports_map'] = {
                kind: {str(idx): name for idx, name in mapping.items()}
                for kind, mapping in ports_map.items()
            }

        return block_dict

    def _serialize_line(self, line: Any) -> Dict[str, Any]:
        """Serialize a single connection line to a dict."""
        return {
            "name": line.name,
            "sid": line.sid,
            "srcblock": line.srcblock,
            "srcport": line.srcport,
            "dstblock": line.dstblock,
            "dstport": line.dstport,
            "points": [(p.x(), p.y()) for p in line.points],
            "cptr": getattr(line, 'cptr', 0),
            "selected": line.selected,
        }

    def save_to_file(self, data: Dict[str, Any], filename: str) -> bool:
        """
        Write serialized data to a file.
        
        Args:
            data: Diagram data dict
            filename: Path to save
        """
        try:
            # Ensure the directory exists
            dirname = os.path.dirname(filename)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            
            with open(filename, 'w', encoding='utf-8') as fp:
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
            # In frozen mode, redirect saves/ to a writable location
            if getattr(sys, 'frozen', False) and not os.path.isabs(file):
                from lib.app_paths import get_user_data_dir
                file = os.path.join(get_user_data_dir(), file)
        
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
            with open(filepath, 'r', encoding='utf-8') as fp:
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

        # Recreate top-level blocks (recurses into Subsystems via _construct_block)
        blocks_data = data.get('blocks_data', [])
        for block_data in blocks_data:
            block = self._construct_block(block_data)
            if block is not None:
                self.model.blocks_list.append(block)

        # Build username→name and name→name maps for flexible line references
        block_name_map = self._build_name_map(self.model.blocks_list)

        # Recreate top-level lines
        lines_data = data.get('lines_data', [])
        for line_data in lines_data:
            line = self._construct_line(line_data, block_name_map)
            if line is not None:
                self.model.line_list.append(line)

        # Update line positions
        self.model.update_lines()
        self.model.dirty = False

        return sim_params

    @staticmethod
    def _build_name_map(blocks: list) -> Dict[str, str]:
        """Build {block.name: block.name, block.username: block.name} for line resolution."""
        name_map: Dict[str, str] = {}
        for block in blocks:
            name_map[block.name] = block.name
            if block.username:
                name_map[block.username] = block.name
        return name_map

    def _construct_block(self, block_data: Dict[str, Any]) -> Optional[Any]:
        """
        Reconstruct a single block from its serialized dict. Subsystem, Inport,
        and Outport are special — they inherit from DBlock (not BaseBlock) and
        therefore are not registered in menu_blocks, so they must be
        instantiated directly. For Subsystems, sub_blocks, sub_lines, ports,
        and ports_map are rebuilt recursively so the reloaded diagram matches
        what was saved.
        """
        from lib.simulation.block import DBlock

        block_fn = block_data['block_fn']
        block_rect = QRect(
            block_data['coords_left'],
            block_data['coords_top'],
            block_data['coords_width'],
            block_data['coords_height']
        )
        params = block_data.get('params', {})

        if block_fn == 'Subsystem':
            block = self._construct_subsystem(block_data, block_rect, params)
        elif block_fn in ('Inport', 'Outport'):
            block = self._construct_port_block(block_fn, block_data, block_rect, params)
        else:
            menu_block = next(
                (mb for mb in self.model.menu_blocks if mb.block_fn == block_fn),
                None
            )
            if menu_block is None:
                logger.warning(f"Block type {block_fn} not found in menu_blocks")
                return None

            category = getattr(menu_block, 'category', 'Other')
            block_color = self.model._get_category_color(category)

            if menu_block.block_class:
                try:
                    block_instance = menu_block.block_class()
                    b_type = getattr(block_instance, 'b_type', block_data.get('b_type', 2))
                except Exception:
                    b_type = block_data.get('b_type', 2)
            else:
                b_type = block_data.get('b_type', 2)

            block = DBlock(
                block_fn,
                block_data['sid'],
                block_rect,
                block_color,
                block_data['in_ports'],
                block_data['out_ports'],
                b_type,
                menu_block.io_edit,
                menu_block.fn_name,
                params,
                block_data.get('external', False),
                username=block_data.get('username', ''),
                block_class=menu_block.block_class,
                colors=self.model.colors,
                category=category,
            )

        # Restore the exact saved name. Subsystem boundary blocks (Inport/Outport)
        # are renamed by subsystem_manager based on port index rather than sid,
        # so re-deriving from block_fn+sid would produce the wrong key for
        # internal lines. Older save files without a "name" field keep the
        # constructor-generated default.
        saved_name = block_data.get('name')
        if saved_name:
            block.name = saved_name
            if block.params is not None:
                block.params['_name_'] = saved_name

        block.flipped = block_data.get('flipped', False)
        block.height_base = block_data.get('coords_height_base', block.height)

        # b_color is re-derived from the current palette via category, not
        # restored from the file — old files have stale palette hex baked in.

        return block

    def _construct_subsystem(self, block_data, block_rect, params):
        """Build a real Subsystem instance and recursively restore its contents."""
        from blocks.subsystem import Subsystem

        sid = block_data['sid']
        username = block_data.get('username', '') or f"Subsystem{sid}"
        block = Subsystem(
            block_name=username,
            sid=sid,
            coords=block_rect,
            color=block_data.get('b_color') or 'lightgray',
        )
        block.username = username
        if params:
            block.params.update(params)
        block.params['_name_'] = block.name
        block.external = block_data.get('external', False)

        # Restore external port layout and the index→port-name map.
        block.ports = block_data.get('ports', {}) or {}
        ports_map_raw = block_data.get('ports_map', {}) or {}
        block.ports_map = {
            kind: {int(idx): name for idx, name in mapping.items()}
            for kind, mapping in ports_map_raw.items()
        }

        # Recursively rebuild internal blocks and lines.
        for child_data in block_data.get('sub_blocks', []) or []:
            child = self._construct_block(child_data)
            if child is not None:
                block.sub_blocks.append(child)

        sub_name_map = self._build_name_map(block.sub_blocks)
        for child_line_data in block_data.get('sub_lines', []) or []:
            child_line = self._construct_line(child_line_data, sub_name_map)
            if child_line is not None:
                block.sub_lines.append(child_line)

        # Recompute external port positions now that ports dict is populated.
        try:
            block.update_Block()
        except Exception as e:
            logger.warning(f"Subsystem update_Block failed after load: {e}")
        return block

    def _construct_port_block(self, block_fn, block_data, block_rect, params):
        """Build an Inport or Outport block (subsystem boundary markers)."""
        if block_fn == 'Inport':
            from blocks.inport import Inport
            cls = Inport
        else:
            from blocks.outport import Outport
            cls = Outport

        sid = block_data['sid']
        username = block_data.get('username', '') or f"{block_fn[:2]}{sid}"
        block = cls(
            block_name=username,
            sid=sid,
            coords=block_rect,
            color=block_data.get('b_color') or ('green' if block_fn == 'Inport' else 'red'),
        )
        if params:
            block.params.update(params)
        block.external = block_data.get('external', False)
        return block

    def _construct_line(self, line_data: Dict[str, Any],
                        block_name_map: Dict[str, str]) -> Optional[Any]:
        """Reconstruct a single connection line from its serialized dict."""
        from lib.simulation.connection import DLine

        points = [tuple(p) if isinstance(p, list) else p for p in line_data['points']]

        srcblock = block_name_map.get(line_data['srcblock'], line_data['srcblock'])
        dstblock = block_name_map.get(line_data['dstblock'], line_data['dstblock'])

        return DLine(
            line_data['sid'],
            srcblock,
            line_data['srcport'],
            dstblock,
            line_data['dstport'],
            points,
        )
