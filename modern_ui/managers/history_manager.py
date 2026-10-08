import collections
import copy
import logging
from PyQt6.QtCore import QPoint, QRect
from PyQt6.QtGui import QColor

# Import DBlock/DSim dependencies for restoration
# Note: In a cleaner architecture we would use factory methods, but for now we follow existing logic
from lib.simulation.block import DBlock

logger = logging.getLogger(__name__)


class HistoryManager:
    """
    Manages the Undo/Redo stack and state capture/restore for the ModernCanvas.
    """

    def __init__(self, canvas):
        self.canvas = canvas
        self.dsim = canvas.dsim

        # Undo/Redo stacks — deque(maxlen) gives O(1) eviction at front.
        # The redo stack is capped like the undo stack: each entry deep-copies
        # every block's params, so an uncapped redo history was an unbounded
        # memory hold for large diagrams.
        self.max_undo_steps = 50
        self.undo_stack = collections.deque(maxlen=self.max_undo_steps)
        self.redo_stack = collections.deque(maxlen=self.max_undo_steps)

    def capture_snapshot(self):
        """Snapshot the current diagram so it can be pushed later.

        Use this at the *start* of a drag-style gesture and hand the result to
        ``push_snapshot`` once the gesture turns out to have changed something,
        so the undo entry holds the pre-gesture state.
        """
        return self._capture_state()

    def push_snapshot(self, state, description="Action"):
        """Push a previously captured snapshot onto the undo stack."""
        if not state:
            return
        self.undo_stack.append({"state": state, "description": description})
        self.redo_stack.clear()
        logger.debug(f"Pushed to undo stack: {description} (stack size: {len(self.undo_stack)})")

    def push_undo(self, description="Action"):
        """Push current state to undo stack."""
        try:
            state = self._capture_state()
            if state:
                self.undo_stack.append({"state": state, "description": description})

                # Limit stack size: deque(maxlen) auto-evicts oldest entry — no manual pop(0) needed

                # Clear redo stack when new action is performed
                self.redo_stack.clear()

                logger.debug(
                    f"Pushed to undo stack: {description} (stack size: {len(self.undo_stack)})"
                )

        except Exception:
            # Never swallow silently: a failed push means the *next* undo will
            # restore the wrong state, so the traceback has to reach the log.
            logger.warning("Error pushing undo snapshot", exc_info=True)

    # ------------------------------------------------------------------
    # Scope handling
    # ------------------------------------------------------------------
    def _current_scope(self):
        """Names of the subsystems entered to reach the current scope ([] = top)."""
        try:
            path = self.dsim.get_current_path()
        except Exception:
            return []
        if isinstance(path, list):
            return [str(p) for p in path[1:]]
        return []

    def _navigate_to_scope(self, target):
        """Move the canvas to the scope named by ``target`` (a list of names).

        Snapshots are taken in whatever scope was current, so applying one while
        the user is somewhere else would graft that scope's blocks into the wrong
        list. Instead undo/redo first navigates to the snapshot's own scope (exit
        to the common ancestor, then enter by name). Returns False when the path
        no longer resolves.
        """
        target = list(target or [])
        current = self._current_scope()
        if current == target:
            return True
        common = 0
        while common < min(len(current), len(target)) and current[common] == target[common]:
            common += 1
        try:
            for _ in range(len(current) - common):
                self.dsim.exit_subsystem()
            for name in target[common:]:
                sub = next(
                    (
                        b
                        for b in self.dsim.blocks_list
                        if b.name == name and hasattr(b, "sub_blocks")
                    ),
                    None,
                )
                if sub is None:
                    logger.warning(f"Undo: subsystem '{name}' not found in scope")
                    return False
                self.dsim.enter_subsystem(sub)
        except Exception:
            logger.warning("Undo: could not navigate to snapshot scope", exc_info=True)
            return False
        scope_signal = getattr(self.canvas, "scope_changed", None)
        if scope_signal is not None:
            scope_signal.emit(self.dsim.get_current_path())
        return True

    def _apply_entry(self, entry, other_stack, other_description):
        """Apply a stack entry; on success record the inverse on ``other_stack``.

        Returns True when the entry was applied. The entry is only popped by the
        caller after this succeeds, and ``_restore_state`` swaps atomically, so a
        failure leaves both stacks and the diagram untouched.
        """
        state = entry["state"]
        if not self._navigate_to_scope(state.get("scope", [])):
            return False
        current_state = self._capture_state()
        if not self._restore_state(state):
            return False
        if current_state:
            other_stack.append({"state": current_state, "description": other_description})
        self.dsim.dirty = True
        return True

    def undo(self):
        """Undo the last action. Returns True if something was undone."""
        try:
            if not self.undo_stack:
                logger.info("Nothing to undo")
                return False
            item = self.undo_stack[-1]
            if self._apply_entry(item, self.redo_stack, "Redo"):
                self.undo_stack.pop()
                logger.info(f"Undone: {item['description']}")
                return True
            logger.error("Failed to undo")
        except Exception as e:
            logger.error(f"Error in undo: {str(e)}")
        return False

    def redo(self):
        """Redo the last undone action. Returns True if something was redone."""
        try:
            if not self.redo_stack:
                logger.info("Nothing to redo")
                return False
            item = self.redo_stack[-1]
            if self._apply_entry(item, self.undo_stack, "Undo"):
                self.redo_stack.pop()
                logger.info(f"Redone: {item['description']}")
                return True
            logger.error("Failed to redo")
        except Exception as e:
            logger.error(f"Error in redo: {str(e)}")
        return False

    def _capture_state(self):
        """Capture current diagram state (Snapshot)."""
        try:
            state = {"blocks": [], "lines": [], "scope": self._current_scope()}

            # Capture all blocks
            for block in self.dsim.blocks_list:
                block_data = {
                    "name": block.name,
                    "block_fn": block.block_fn,
                    "coords": (block.left, block.top, block.width, block.height_base),
                    "color": block.b_color.name()
                    if hasattr(block.b_color, "name")
                    else str(block.b_color),
                    "category": getattr(block, "category", "Other"),
                    "in_ports": block.in_ports,
                    "out_ports": block.out_ports,
                    "b_type": block.b_type,
                    "io_edit": block.io_edit,
                    "fn_name": block.fn_name,
                    "params": copy.deepcopy(block.params)
                    if hasattr(block, "params") and block.params
                    else {},
                    "external": block.external,
                    "selected": block.selected,
                    "username": getattr(block, "username", ""),
                    "flipped": bool(getattr(block, "flipped", False)),
                }
                # Subsystems cannot be rebuilt from these scalars: restoring
                # them as a plain DBlock would silently drop sub_blocks,
                # sub_lines, ports and the mask. Snapshot the object itself so
                # undo/redo round-trips a whole (possibly masked) subsystem.
                # Inport/Outport are not in menu_blocks either, so they cannot be
                # rebuilt from block_fn and are snapshotted as objects too.
                if block.block_fn in ("Subsystem", "Inport", "Outport") or hasattr(
                    block, "sub_blocks"
                ):
                    try:
                        block_data["snapshot"] = copy.deepcopy(block)
                    except Exception as e:
                        logger.error(f"Could not snapshot subsystem {block.name}: {e}")
                state["blocks"].append(block_data)

            # Capture all connections
            for line in self.dsim.line_list:
                line_data = {
                    "name": line.name,
                    "srcblock": line.srcblock,
                    "srcport": line.srcport,
                    "dstblock": line.dstblock,
                    "dstport": line.dstport,
                    "selected": line.selected,
                    "routing_mode": line.routing_mode
                    if hasattr(line, "routing_mode")
                    else "bezier",
                    "label": line.label if hasattr(line, "label") else "",
                    # Custom routing (manual bends / auto-routed paths) must
                    # survive undo/redo, so snapshot the waypoints too.
                    "points": [(p.x(), p.y()) for p in getattr(line, "points", [])],
                    "modified": bool(getattr(line, "modified", False)),
                    "auto_routed": bool(getattr(line, "auto_routed", False)),
                }
                state["lines"].append(line_data)

            return state

        except Exception:
            logger.warning("Error capturing diagram state for undo", exc_info=True)
            return None

    def _restore_state(self, state):
        """Restore diagram state from snapshot.

        Everything is rebuilt into temporary lists first and swapped into the
        live (in-place) lists only once the whole snapshot has been built, so a
        failure part-way leaves the current diagram untouched.
        """
        try:
            if not state:
                return False

            # block_fn -> block_class via the menu_blocks registry. Naive
            # module/class-name guessing from block_fn fails when the two
            # diverge (e.g. "TranFn" -> TransferFunctionBlock), leaving the
            # restored DBlock without a block_instance.
            class_by_fn = {mb.block_fn: mb.block_class for mb in self.dsim.menu_blocks}

            new_blocks = []
            for block_data in state["blocks"]:
                # A block that cannot be rebuilt aborts the restore: dropping it
                # silently would turn an undo into data loss.
                new_blocks.append(self._rebuild_block(block_data, class_by_fn))

            new_lines = self._rebuild_lines(state["lines"], new_blocks)

            # Commit: mutate in place so navigation-stack references stay valid.
            self.dsim.blocks_list[:] = new_blocks
            self.dsim.line_list[:] = new_lines

            # Clear hover state (old objects are now invalid) through the owner.
            self.canvas.interaction_manager.clear_hover()

            # Clear validation errors when state is restored (undo/redo)
            if hasattr(self.canvas, "clear_validation"):
                self.canvas.clear_validation()

            if hasattr(self.canvas, "update"):
                self.canvas.update()
            return True

        except Exception as e:
            logger.error(f"Error restoring state: {str(e)}")
            return False

    def _rebuild_block(self, block_data, class_by_fn):
        """Build one block from its snapshot entry (without touching the diagram)."""
        snapshot = block_data.get("snapshot")
        if snapshot is not None:
            # Subsystem / Inport / Outport: restore the snapshotted object
            # wholesale (contents, ports and mask included). Deep-copy again so
            # a later undo/redo of the same entry is independent.
            restored = copy.deepcopy(snapshot)
            restored.selected = block_data.get("selected", False)
            return restored

        coords = QRect(*block_data["coords"])

        block_fn = block_data["block_fn"]
        block_class = class_by_fn.get(block_fn)
        if block_class is None:
            logger.warning(
                f"Restore: block_class not found for '{block_fn}' in menu_blocks; "
                f"restored block will have no block_instance."
            )

        # Extract sid from name (e.g., "step0" -> 0)
        name = block_data["name"]
        sid = int(name[len(block_fn) :]) if len(name) > len(block_fn) else 0

        block = DBlock(
            block_data["block_fn"],
            sid,
            coords,
            QColor(block_data["color"]),
            block_data["in_ports"],
            block_data["out_ports"],
            block_data["b_type"],
            block_data["io_edit"],
            block_data["fn_name"],
            block_data["params"],
            block_data["external"],
            username=block_data.get("username", ""),
            block_class=block_class,
            category=block_data.get("category", "Other"),
        )

        # Restore selection state and original name
        block.selected = block_data.get("selected", False)
        block.name = name
        # The flipped setter re-lays the ports, so set it after construction.
        block.flipped = bool(block_data.get("flipped", False))
        return block

    def _rebuild_lines(self, lines_data, blocks):
        """Build connection lines for ``blocks`` (not yet attached to the diagram)."""
        from lib.simulation.connection import DLine

        new_lines = []
        block_by_name = {b.name: b for b in blocks}
        for line_data in lines_data:
            try:
                src_block = block_by_name.get(line_data["srcblock"])
                dst_block = block_by_name.get(line_data["dstblock"])

                if not (src_block and dst_block):
                    continue

                src_port = line_data["srcport"]
                dst_port = line_data["dstport"]

                # Bounds-check ports: the rebuilt block may have fewer ports
                # than the snapshot recorded. Skip the bad connection rather
                # than raising IndexError and aborting the whole restore.
                if src_port >= len(src_block.out_coords):
                    logger.warning(
                        f"Restore: srcport {src_port} out of range for "
                        f"'{src_block.name}' ({len(src_block.out_coords)} out ports); skipping connection"
                    )
                    continue
                if dst_port >= len(dst_block.in_coords):
                    logger.warning(
                        f"Restore: dstport {dst_port} out of range for "
                        f"'{dst_block.name}' ({len(dst_block.in_coords)} in ports); skipping connection"
                    )
                    continue

                src_port_pos = src_block.out_coords[src_port]
                dst_port_pos = dst_block.in_coords[dst_port]

                name = line_data["name"]
                try:
                    sid = int(name[len("Line") :])
                except ValueError:
                    sid = len(new_lines)
                line = DLine(
                    sid,
                    srcblock=line_data["srcblock"],
                    srcport=src_port,
                    dstblock=line_data["dstblock"],
                    dstport=dst_port,
                    points=(src_port_pos, dst_port_pos),
                )
                line.color = QColor(255, 0, 0)
                line.selected = line_data.get("selected", False)
                line.name = name
                # Restore routing mode and label
                if "routing_mode" in line_data:
                    line.routing_mode = line_data["routing_mode"]
                if "label" in line_data:
                    line.label = line_data["label"]
                line.auto_routed = bool(line_data.get("auto_routed", False))
                saved_points = line_data.get("points") or []
                if line_data.get("modified") and len(saved_points) > 2:
                    # Replay the snapshotted waypoints between the rebuilt
                    # ports so bends come back exactly.
                    pts = [QPoint(int(x), int(y)) for x, y in saved_points]
                    pts[0] = QPoint(src_port_pos)
                    pts[-1] = QPoint(dst_port_pos)
                    line.modified = True
                    line.path, line.points, line.segments = line.create_trajectory(
                        pts[0], pts[-1], blocks, points=pts
                    )
                else:
                    # Recalculate the default path for the restored routing
                    # mode against the rebuilt blocks.
                    line.reroute(blocks)
                new_lines.append(line)

            except Exception as e:
                logger.error(
                    f"Error restoring connection {line_data.get('name', 'unknown')}: {str(e)}"
                )
                continue
        return new_lines
