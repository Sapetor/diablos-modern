"""Diagram Validation System
Validates block diagrams for integrity errors before simulation.
"""

import logging
from typing import List, Set
from enum import Enum

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    ERROR = "error"      # Prevents simulation
    WARNING = "warning"  # Simulation can run but may have issues
    INFO = "info"        # Informational message


class ValidationError:
    """Represents a validation error or warning."""

    def __init__(self, severity: ErrorSeverity, message: str,
                 blocks: List = None, connections: List = None,
                 suggestion: str = None):
        """
        Initialize a validation error.

        Args:
            severity: Error severity level
            message: Human-readable error message
            blocks: List of blocks involved in the error
            connections: List of connections involved in the error
            suggestion: Suggested fix for the error
        """
        self.severity = severity
        self.message = message
        self.blocks = blocks or []
        self.connections = connections or []
        self.suggestion = suggestion

    def __str__(self):
        return f"[{self.severity.value.upper()}] {self.message}"


class DiagramValidator:
    """Validates block diagrams for common errors and issues."""

    def __init__(self, dsim):
        """
        Initialize the validator.

        Args:
            dsim: DSim instance containing blocks and connections
        """
        self.dsim = dsim
        self.errors = []

    def validate(self) -> List[ValidationError]:
        """
        Run all validation checks on the diagram.

        Returns:
            List of ValidationError objects
        """
        self.errors = []

        # Build connection maps once for efficiency (O(n) instead of O(n²))
        connection_maps = self._build_connection_maps()

        # Run all validation checks with pre-built maps
        self._check_disconnected_inputs(connection_maps)
        self._check_disconnected_outputs(connection_maps)
        self._check_isolated_blocks(connection_maps)
        self._check_invalid_connections(connection_maps)
        self._check_duplicate_connections(connection_maps)
        self._check_goto_from_tags()

        logger.info(f"Validation complete: {len(self.errors)} issues found")
        return self.errors

    def _build_connection_maps(self) -> dict:
        """
        Build connection maps once to avoid O(n²) complexity.

        Returns:
            Dictionary containing:
                - input_connections: {(block_name, port_idx): [connections]}
                - output_connections: {(block_name, port_idx): [connections]}
                - connected_blocks: set of block names with any connection
                - valid_block_names: set of all valid block names
        """
        input_connections = {}
        output_connections = {}
        connected_blocks = set()

        for line in self.dsim.line_list:
            if getattr(line, "hidden", False):
                continue
            # Track input connections
            input_key = (line.dstblock, line.dstport)
            if input_key not in input_connections:
                input_connections[input_key] = []
            input_connections[input_key].append(line)

            # Track output connections
            output_key = (line.srcblock, line.srcport)
            if output_key not in output_connections:
                output_connections[output_key] = []
            output_connections[output_key].append(line)

            # Track connected blocks
            connected_blocks.add(line.srcblock)
            connected_blocks.add(line.dstblock)

        return {
            'input_connections': input_connections,
            'output_connections': output_connections,
            'connected_blocks': connected_blocks,
            'valid_block_names': {block.name for block in self.dsim.blocks_list}
        }

    def _check_disconnected_inputs(self, connection_maps: dict) -> None:
        """Check for input ports that have no connections."""
        input_connections = connection_maps['input_connections']

        for block in self.dsim.blocks_list:
            # Skip blocks that don't require inputs (use block property if available)
            if hasattr(block, 'block_instance') and block.block_instance and hasattr(block.block_instance, 'requires_inputs'):
                if not block.block_instance.requires_inputs:
                    continue
            elif hasattr(block, 'category') and block.category == 'Sources':
                continue

            # Get optional inputs list (if the block specifies any)
            optional_inputs = set()
            if hasattr(block, 'block_instance') and block.block_instance:
                if hasattr(block.block_instance, 'optional_inputs'):
                    optional_inputs = set(block.block_instance.optional_inputs)

            # Check for disconnected inputs using pre-built map
            for i in range(block.in_ports):
                if i in optional_inputs:
                    continue  # Skip optional inputs
                if (block.name, i) not in input_connections:
                    error = ValidationError(
                        severity=ErrorSeverity.ERROR,
                        message=f"Block '{block.username or block.name}' has disconnected input port {i+1}",
                        blocks=[block],
                        suggestion=f"Connect an output to input port {i+1} or remove the block"
                    )
                    self.errors.append(error)

    def _check_disconnected_outputs(self, connection_maps: dict) -> None:
        """Check for output ports that have no connections."""
        output_connections = connection_maps['output_connections']

        for block in self.dsim.blocks_list:
            # Skip blocks that don't require outputs to be connected (use block property if available)
            if hasattr(block, 'block_instance') and block.block_instance and hasattr(block.block_instance, 'requires_outputs'):
                if not block.block_instance.requires_outputs:
                    continue
            elif hasattr(block, 'category') and block.category in ['Sinks', 'Other']:
                continue

            # Get optional outputs (ports that don't need to be connected)
            optional_outputs = set()
            if hasattr(block, 'block_instance') and block.block_instance:
                if hasattr(block.block_instance, 'optional_outputs'):
                    optional_outputs = set(block.block_instance.optional_outputs)

            # Check for disconnected outputs using pre-built map
            for i in range(block.out_ports):
                # Skip optional output ports
                if i in optional_outputs:
                    continue
                if (block.name, i) not in output_connections:
                    error = ValidationError(
                        severity=ErrorSeverity.WARNING,
                        message=f"Block '{block.username or block.name}' has disconnected output port {i+1}",
                        blocks=[block],
                        suggestion=f"Connect output port {i+1} to another block or add a sink"
                    )
                    self.errors.append(error)

    def _check_isolated_blocks(self, connection_maps: dict) -> None:
        """Check for blocks with no connections at all."""
        connected_blocks = connection_maps['connected_blocks']

        for block in self.dsim.blocks_list:
            # Block is isolated if it's not in the connected blocks set
            if block.name not in connected_blocks:
                error = ValidationError(
                    severity=ErrorSeverity.ERROR,
                    message=f"Block '{block.username or block.name}' is not connected to anything",
                    blocks=[block],
                    suggestion="Connect this block to the diagram or remove it"
                )
                self.errors.append(error)

    def _check_goto_from_tags(self) -> None:
        """Validate tag usage for Goto/From blocks (duplicates, missing links)."""
        goto_tags = {}
        from_tags = {}
        for block in self.dsim.blocks_list:
            if block.block_fn == 'Goto':
                tag = str(block.params.get('tag', '')).strip()
                goto_tags.setdefault(tag, []).append(block)
            elif block.block_fn == 'From':
                tag = str(block.params.get('tag', '')).strip()
                from_tags.setdefault(tag, []).append(block)

        # Multiple Goto with same tag -> warning (ambiguous source)
        for tag, gotos in goto_tags.items():
            if len(gotos) > 1:
                self.errors.append(
                    ValidationError(
                        severity=ErrorSeverity.WARNING,
                        message=f"Multiple Goto blocks share tag '{tag or '(empty)'}'",
                        blocks=gotos,
                        suggestion="Use unique tags or remove duplicates to avoid ambiguity"
                    )
                )

        # From without matching Goto -> error
        for tag, frs in from_tags.items():
            if tag not in goto_tags:
                self.errors.append(
                    ValidationError(
                        severity=ErrorSeverity.ERROR,
                        message=f"From tag '{tag or '(empty)'}' has no matching Goto",
                        blocks=frs,
                        suggestion="Add a Goto with the same tag or update the tag"
                    )
                )

        # Goto with no From -> warning (unused)
        for tag, gotos in goto_tags.items():
            if tag not in from_tags:
                self.errors.append(
                    ValidationError(
                        severity=ErrorSeverity.WARNING,
                        message=f"Goto tag '{tag or '(empty)'}' is unused (no From)",
                        blocks=gotos,
                        suggestion="Add a From with the same tag or remove the Goto"
                    )
                )

    def _check_invalid_connections(self, connection_maps: dict) -> None:
        """Check for connections with invalid block references."""
        valid_block_names = connection_maps['valid_block_names']

        for line in self.dsim.line_list:
            if getattr(line, "hidden", False):
                continue
            if line.srcblock not in valid_block_names:
                error = ValidationError(
                    severity=ErrorSeverity.ERROR,
                    message=f"Connection '{line.name}' references non-existent source block '{line.srcblock}'",
                    connections=[line],
                    suggestion="Delete this invalid connection"
                )
                self.errors.append(error)

            if line.dstblock not in valid_block_names:
                error = ValidationError(
                    severity=ErrorSeverity.ERROR,
                    message=f"Connection '{line.name}' references non-existent destination block '{line.dstblock}'",
                    connections=[line],
                    suggestion="Delete this invalid connection"
                )
                self.errors.append(error)

    def _check_duplicate_connections(self, connection_maps: dict) -> None:
        """Check for multiple connections to the same input port."""
        input_connections = connection_maps['input_connections']

        # Find duplicates
        for (block_name, port_idx), connections in input_connections.items():
            if len(connections) > 1:
                # Find the block
                block = None
                for b in self.dsim.blocks_list:
                    if b.name == block_name:
                        block = b
                        break

                error = ValidationError(
                    severity=ErrorSeverity.ERROR,
                    message=f"Block '{block.username if block else block_name}' input port {port_idx+1} has {len(connections)} connections",
                    blocks=[block] if block else [],
                    connections=connections,
                    suggestion=f"Remove all but one connection to input port {port_idx+1}"
                )
                self.errors.append(error)

    def has_errors(self) -> bool:
        """Check if there are any errors (not warnings)."""
        return any(e.severity == ErrorSeverity.ERROR for e in self.errors)

    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return any(e.severity == ErrorSeverity.WARNING for e in self.errors)

    def get_errors_by_severity(self, severity: ErrorSeverity) -> List[ValidationError]:
        """Get all errors of a specific severity."""
        return [e for e in self.errors if e.severity == severity]

    def get_blocks_with_errors(self) -> Set:
        """Get set of all blocks that have errors."""
        blocks = set()
        for error in self.errors:
            if error.severity == ErrorSeverity.ERROR:
                blocks.update(error.blocks)
        return blocks

    def get_blocks_with_warnings(self) -> Set:
        """Get set of all blocks that have warnings."""
        blocks = set()
        for error in self.errors:
            if error.severity == ErrorSeverity.WARNING:
                blocks.update(error.blocks)
        return blocks
