"""Diagram graph analysis — pure connectivity / topology queries.

Extracted from :class:`~lib.engine.simulation_engine.SimulationEngine`: every
function here operates purely on an explicit block list and/or line (connection)
list, carrying no engine state. The engine keeps thin methods that select the
right active/model lists and delegate here, so the public API and all hot call
sites are unchanged — only the traversal logic moved, and it is now unit-tested
against plain stand-in blocks/lines instead of a fully initialized engine.

A "line" is any object with ``srcblock``/``srcport``/``dstblock``/``dstport``;
a "block" here needs ``name``/``hierarchy``/``in_ports``/``out_ports`` and,
for the integrity check, an optional ``block_instance``.
"""

import logging

import numpy as np

from lib.engine.topo import kahn_topological_order

logger = logging.getLogger(__name__)


def get_neighbors(block_name, lines):
    """Return ``(inputs, outputs)`` connection-dict lists for ``block_name``."""
    inputs = []
    outputs = []
    for line in lines:
        if line.dstblock == block_name:
            inputs.append(
                {"srcblock": line.srcblock, "srcport": line.srcport, "dstport": line.dstport}
            )
        if line.srcblock == block_name:
            outputs.append(
                {"dstblock": line.dstblock, "srcport": line.srcport, "dstport": line.dstport}
            )
    return inputs, outputs


def get_outputs(block_name, lines):
    """Return the list of output connections leaving ``block_name``."""
    outputs = []
    for line in lines:
        if line.srcblock == block_name:
            outputs.append(
                {"dstblock": line.dstblock, "srcport": line.srcport, "dstport": line.dstport}
            )
    return outputs


def get_max_hierarchy(blocks):
    """Return the maximum ``hierarchy`` across ``blocks`` (-1 if empty)."""
    max_h = -1
    for block in blocks:
        if block.hierarchy > max_h:
            max_h = block.hierarchy
    return max_h


def children_recognition(block_name, children_list, lines):
    """Recursively accumulate all downstream block names of ``block_name``.

    Mutates and returns ``children_list`` (matching the original in-place API).
    """
    outputs = get_outputs(block_name, lines)
    for output in outputs:
        child_name = output["dstblock"]
        if child_name not in children_list:
            children_list.append(child_name)
            children_recognition(child_name, children_list, lines)
    return children_list


def is_child_of(block_name, children_list):
    """Return ``(is_child, matching_connections)`` for ``block_name`` within a
    list of connection dicts (was ``SimulationEngine._children_recognition``)."""
    child_ports = [child for child in children_list if child.get("dstblock") == block_name]
    if not child_ports:
        return False, []
    return True, child_ports


def detect_algebraic_loops(uncomputed_blocks, lines, memory_blocks):
    """Detect an algebraic loop among ``uncomputed_blocks`` via Kahn's algorithm.

    Returns ``(is_algebraic, cycle_nodes)``. A cycle broken by a memory block
    (transfer function, integrator, ...) is *not* an algebraic loop, so cycles
    that include any name in ``memory_blocks`` are reported as clear.
    """
    if len(uncomputed_blocks) == 0:
        return False, []

    logger.debug("Checking for algebraic loops...")
    logger.debug(f"Uncomputed blocks: {[b.name for b in uncomputed_blocks]}")

    uncomputed_block_names = {block.name for block in uncomputed_blocks}

    # Build the graph only with uncomputed blocks.
    graph = {block.name: [] for block in uncomputed_blocks}
    for block in uncomputed_blocks:
        children = get_outputs(block.name, lines)
        for child_info in children:
            child_name = child_info["dstblock"]
            if child_name in uncomputed_block_names:
                graph[block.name].append(child_name)

    # Topological sort (Kahn). Any node left in a cycle has unresolved
    # dependencies -> a non-empty `cycle_nodes` means an algebraic loop.
    _order, cycle_nodes = kahn_topological_order((b.name for b in uncomputed_blocks), graph)

    if cycle_nodes:
        has_memory_block = any(node in memory_blocks for node in cycle_nodes)
        if not has_memory_block:
            logger.error("ALGEBRAIC LOOP DETECTED")
            logger.error(f"Blocks involved: {cycle_nodes}")
            return True, cycle_nodes

    return False, []


def check_diagram_integrity(blocks, lines):
    """Verify that every block's required input/output ports are connected.

    Returns ``True`` if the diagram is valid, ``False`` (with logged errors) if
    any required port is unlinked or an input port has multiple wires landing
    on it. Optional ports (per the block instance's ``optional_inputs`` /
    ``optional_outputs`` / ``requires_outputs``) are exempt.
    """
    logger.debug("Checking diagram integrity")
    error_trigger = False

    for block in blocks:
        inputs, outputs = get_neighbors(block.name, lines)

        # Optional inputs/outputs from the block instance (if available).
        optional_inputs = set()
        if hasattr(block, "block_instance") and block.block_instance:
            if hasattr(block.block_instance, "optional_inputs"):
                optional_inputs = set(block.block_instance.optional_inputs)

        optional_outputs = set()
        if hasattr(block, "block_instance") and block.block_instance:
            if hasattr(block.block_instance, "optional_outputs"):
                optional_outputs = set(block.block_instance.optional_outputs)
            # Also honor the requires_outputs property.
            if hasattr(block.block_instance, "requires_outputs"):
                if not block.block_instance.requires_outputs:
                    optional_outputs = set(range(block.out_ports))

        # Reject multiple wires landing on the same input port: propagate_outputs
        # overwrites input_queue[dstport] per connection, so extra wires would
        # silently last-write-win with no error surfaced to the user.
        dst_counts = {}
        for tupla in inputs:
            dst_counts[tupla["dstport"]] = dst_counts.get(tupla["dstport"], 0) + 1
        duplicated = sorted(p for p, c in dst_counts.items() if c > 1)
        if duplicated:
            logger.error(
                f"ERROR. MULTIPLE CONNECTIONS INTO SAME INPUT PORT: "
                f"{block.name} PORT(S): {duplicated}"
            )
            error_trigger = True

        # Check input ports.
        required_in_ports = block.in_ports - len(optional_inputs)
        connected_required_inputs = sum(1 for t in inputs if t["dstport"] not in optional_inputs)

        if required_in_ports == 1 and connected_required_inputs < 1:
            logger.error(f"ERROR. UNLINKED INPUT IN BLOCK: {block.name}")
            error_trigger = True
        elif required_in_ports > 1 or (block.in_ports > 1 and required_in_ports > 0):
            in_vector = np.zeros(block.in_ports)
            for tupla in inputs:
                in_vector[tupla["dstport"]] += 1
            unlinked = [
                i for i in range(block.in_ports) if in_vector[i] == 0 and i not in optional_inputs
            ]
            if len(unlinked) > 0:
                logger.error(f"ERROR. UNLINKED INPUT(S) IN BLOCK: {block.name} PORT(S): {unlinked}")
                error_trigger = True

        # Check output ports.
        required_out_ports = block.out_ports - len(optional_outputs)
        connected_required_outputs = sum(1 for t in outputs if t["srcport"] not in optional_outputs)

        if required_out_ports == 1 and connected_required_outputs < 1:
            logger.error(f"ERROR. UNLINKED OUTPUT PORT: {block.name}")
            error_trigger = True
        elif required_out_ports > 1 or (block.out_ports > 1 and required_out_ports > 0):
            out_vector = np.zeros(block.out_ports)
            for tupla in outputs:
                out_vector[tupla["srcport"]] += 1
            unlinked = [
                i
                for i in range(block.out_ports)
                if out_vector[i] == 0 and i not in optional_outputs
            ]
            if len(unlinked) > 0:
                logger.error(
                    f"ERROR. UNLINKED OUTPUT(S) IN BLOCK: {block.name} PORT(S): {unlinked}"
                )
                error_trigger = True

    if error_trigger:
        logger.error("Diagram integrity check failed.")
        return False
    logger.debug("NO ISSUES FOUND IN DIAGRAM")
    return True
