"""
Canvas State Management for DiaBloS

Consolidates all canvas state into organized dataclasses for cleaner
state management and easier debugging.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Set, List, Tuple, Any
from PyQt5.QtCore import QPoint, QRect


@dataclass
class ZoomPanState:
    """State for zoom and pan operations."""
    zoom_factor: float = 1.0
    pan_offset: QPoint = field(default_factory=lambda: QPoint(0, 0))
    is_panning: bool = False
    last_pan_pos: QPoint = field(default_factory=lambda: QPoint(0, 0))

    def reset(self):
        """Reset zoom and pan to defaults."""
        self.zoom_factor = 1.0
        self.pan_offset = QPoint(0, 0)
        self.is_panning = False
        self.last_pan_pos = QPoint(0, 0)


@dataclass
class GridState:
    """State for grid display and snapping."""
    visible: bool = True
    size: int = 20
    snap_enabled: bool = True

    def toggle_visibility(self) -> bool:
        """Toggle grid visibility and return new state."""
        self.visible = not self.visible
        return self.visible


@dataclass
class SelectionState:
    """State for rectangle selection operations."""
    rect_start: Optional[QPoint] = None
    rect_end: Optional[QPoint] = None
    is_selecting: bool = False

    def start_selection(self, pos: QPoint):
        """Begin a rectangle selection at pos."""
        self.rect_start = pos
        self.rect_end = pos
        self.is_selecting = True

    def update_selection(self, pos: QPoint):
        """Update the selection rectangle end point."""
        self.rect_end = pos

    def end_selection(self) -> Optional[QRect]:
        """End selection and return the selection rectangle."""
        if self.rect_start and self.rect_end:
            x1 = min(self.rect_start.x(), self.rect_end.x())
            y1 = min(self.rect_start.y(), self.rect_end.y())
            x2 = max(self.rect_start.x(), self.rect_end.x())
            y2 = max(self.rect_start.y(), self.rect_end.y())
            rect = QRect(x1, y1, x2 - x1, y2 - y1)
        else:
            rect = None
        self.clear()
        return rect

    def clear(self):
        """Clear the selection state."""
        self.rect_start = None
        self.rect_end = None
        self.is_selecting = False


@dataclass
class HoverState:
    """State for hover tracking and visual feedback."""
    block: Any = None  # The currently hovered block
    port: Optional[Tuple[Any, int, bool]] = None  # (block, port_index, is_output)
    line: Any = None  # The currently hovered connection line

    def clear(self):
        """Clear all hover state."""
        self.block = None
        self.port = None
        self.line = None

    def set_port(self, block: Any, port_index: int, is_output: bool):
        """Set the hovered port."""
        self.port = (block, port_index, is_output)

    def clear_port(self):
        """Clear just the port hover."""
        self.port = None


@dataclass
class DragState:
    """State for block dragging operations."""
    offset: Optional[QPoint] = None  # Offset from mouse to block origin
    offsets: Dict[Any, QPoint] = field(default_factory=dict)  # For multi-block dragging
    start_positions: Dict[Any, Tuple[int, int]] = field(default_factory=dict)  # For undo

    def start_drag(self, primary_block: Any, mouse_pos: QPoint, selected_blocks: List[Any]):
        """Initialize drag state for one or more blocks."""
        self.offset = QPoint(mouse_pos.x() - primary_block.left,
                            mouse_pos.y() - primary_block.top)
        self.offsets = {}
        self.start_positions = {}

        for block in selected_blocks:
            self.offsets[block] = QPoint(block.left - primary_block.left,
                                         block.top - primary_block.top)
            self.start_positions[block] = (block.left, block.top)

    def end_drag(self):
        """Clear drag state."""
        self.offset = None
        self.offsets = {}
        self.start_positions = {}

    @property
    def is_dragging(self) -> bool:
        """Check if a drag operation is active."""
        return self.offset is not None


@dataclass
class ResizeState:
    """State for block resizing operations."""
    block: Any = None  # The block being resized
    handle: Optional[str] = None  # Which handle is being dragged
    start_rect: Optional[QRect] = None  # Original block rect before resize
    start_pos: Optional[QPoint] = None  # Mouse position at start of resize
    at_limit: bool = False  # True when resize hits minimum size

    def start_resize(self, block: Any, handle: str, mouse_pos: QPoint):
        """Initialize resize state."""
        self.block = block
        self.handle = handle
        self.start_pos = mouse_pos
        self.start_rect = QRect(block.left, block.top, block.width, block.height)
        self.at_limit = False

    def end_resize(self):
        """Clear resize state."""
        self.block = None
        self.handle = None
        self.start_rect = None
        self.start_pos = None
        self.at_limit = False

    @property
    def is_resizing(self) -> bool:
        """Check if a resize operation is active."""
        return self.block is not None and self.handle is not None


@dataclass
class ConnectionState:
    """State for connection creation operations."""
    creation_state: Optional[str] = None  # Current state of line creation
    start_block: Any = None  # Block where connection starts
    start_port: Optional[int] = None  # Port index where connection starts
    temp_line: Any = None  # Temporary line being drawn
    source_block: Any = None  # Source block for the connection
    default_routing_mode: str = "bezier"  # Default routing mode for new connections

    def start_connection(self, block: Any, port: int, state: str = "creating"):
        """Begin creating a new connection."""
        self.creation_state = state
        self.start_block = block
        self.start_port = port
        self.source_block = block

    def end_connection(self):
        """Clear connection creation state."""
        self.creation_state = None
        self.start_block = None
        self.start_port = None
        self.temp_line = None
        self.source_block = None

    @property
    def is_creating(self) -> bool:
        """Check if a connection is being created."""
        return self.creation_state is not None


@dataclass
class ValidationState:
    """State for diagram validation."""
    errors: List[Any] = field(default_factory=list)
    blocks_with_errors: Set[Any] = field(default_factory=set)
    blocks_with_warnings: Set[Any] = field(default_factory=set)
    show_errors: bool = False

    def clear(self):
        """Clear all validation state."""
        self.errors = []
        self.blocks_with_errors = set()
        self.blocks_with_warnings = set()

    def add_error(self, error: Any, block: Any = None):
        """Add a validation error."""
        self.errors.append(error)
        if block:
            self.blocks_with_errors.add(block)

    def add_warning(self, block: Any):
        """Add a block with warnings."""
        self.blocks_with_warnings.add(block)


@dataclass
class CanvasState:
    """
    Unified state container for the ModernCanvas.

    Groups all canvas state into logical sub-states for cleaner
    management and easier debugging/serialization.
    """
    zoom_pan: ZoomPanState = field(default_factory=ZoomPanState)
    grid: GridState = field(default_factory=GridState)
    selection: SelectionState = field(default_factory=SelectionState)
    hover: HoverState = field(default_factory=HoverState)
    drag: DragState = field(default_factory=DragState)
    resize: ResizeState = field(default_factory=ResizeState)
    connection: ConnectionState = field(default_factory=ConnectionState)
    validation: ValidationState = field(default_factory=ValidationState)

    def reset_all(self):
        """Reset all state to defaults."""
        self.zoom_pan.reset()
        self.grid = GridState()
        self.selection.clear()
        self.hover.clear()
        self.drag.end_drag()
        self.resize.end_resize()
        self.connection.end_connection()
        self.validation.clear()

    def reset_interaction_state(self):
        """Reset transient interaction state (hover, drag, resize, selection)."""
        self.selection.clear()
        self.hover.clear()
        self.drag.end_drag()
        self.resize.end_resize()
        self.connection.end_connection()
