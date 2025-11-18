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

        # Run all validation checks
        self._check_disconnected_inputs()
        self._check_disconnected_outputs()
        self._check_isolated_blocks()
        self._check_invalid_connections()
        self._check_duplicate_connections()

        logger.info(f"Validation complete: {len(self.errors)} issues found")
        return self.errors

    def _check_disconnected_inputs(self):
        """Check for input ports that have no connections."""
        for block in self.dsim.blocks_list:
            # Skip blocks that don't require inputs (sources)
            if hasattr(block, 'category') and block.category == 'Sources':
                continue

            # Get connected input ports
            connected_inputs = set()
            for line in self.dsim.line_list:
                if line.dstblock == block.name:
                    connected_inputs.add(line.dstport)

            # Check for disconnected inputs
            for i in range(block.in_ports):
                if i not in connected_inputs:
                    error = ValidationError(
                        severity=ErrorSeverity.ERROR,
                        message=f"Block '{block.username or block.name}' has disconnected input port {i+1}",
                        blocks=[block],
                        suggestion=f"Connect an output to input port {i+1} or remove the block"
                    )
                    self.errors.append(error)

    def _check_disconnected_outputs(self):
        """Check for output ports that have no connections."""
        for block in self.dsim.blocks_list:
            # Skip blocks that don't require outputs to be connected (sinks)
            if hasattr(block, 'category') and block.category in ['Sinks', 'Other']:
                continue

            # Get connected output ports
            connected_outputs = set()
            for line in self.dsim.line_list:
                if line.srcblock == block.name:
                    connected_outputs.add(line.srcport)

            # Check for disconnected outputs
            for i in range(block.out_ports):
                if i not in connected_outputs:
                    error = ValidationError(
                        severity=ErrorSeverity.WARNING,
                        message=f"Block '{block.username or block.name}' has disconnected output port {i+1}",
                        blocks=[block],
                        suggestion=f"Connect output port {i+1} to another block or add a sink"
                    )
                    self.errors.append(error)

    def _check_isolated_blocks(self):
        """Check for blocks with no connections at all."""
        for block in self.dsim.blocks_list:
            has_input = False
            has_output = False

            for line in self.dsim.line_list:
                if line.srcblock == block.name:
                    has_output = True
                if line.dstblock == block.name:
                    has_input = True

            # Block is isolated if it has no connections
            if not has_input and not has_output:
                error = ValidationError(
                    severity=ErrorSeverity.ERROR,
                    message=f"Block '{block.username or block.name}' is not connected to anything",
                    blocks=[block],
                    suggestion="Connect this block to the diagram or remove it"
                )
                self.errors.append(error)

    def _check_invalid_connections(self):
        """Check for connections with invalid block references."""
        valid_block_names = {block.name for block in self.dsim.blocks_list}

        for line in self.dsim.line_list:
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

    def _check_duplicate_connections(self):
        """Check for multiple connections to the same input port."""
        input_connections = {}  # (block_name, port_index) -> [connections]

        for line in self.dsim.line_list:
            key = (line.dstblock, line.dstport)
            if key not in input_connections:
                input_connections[key] = []
            input_connections[key].append(line)

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
