"""UDP Send sink block: send the input values to a UDP address each step it
runs (set ``sampling_time`` to send at a fixed rate instead)."""

import logging
import socket

from blocks import _hardware_io as hw
from blocks.base_block import BaseBlock

logger = logging.getLogger(__name__)


class UDPSendBlock(BaseBlock):
    @property
    def block_name(self):
        return "UDPSend"

    @property
    def display_name(self):
        return "UDP Send"

    @property
    def category(self):
        return "Sinks"

    @property
    def color(self):
        return "red"

    @property
    def requires_outputs(self):
        return False

    @property
    def doc(self):
        return (
            "Sends the input values to a UDP address.\n\n"
            "Parameters:\n"
            "- host, port: where to send (127.0.0.1 = this computer).\n"
            "- format: 'text' (comma-separated line) or 'float64' (little-endian "
            "doubles).\n"
            "- sampling_time: send period in seconds; -1 sends on every step.\n\n"
            "Sending never blocks the simulation; a message the network drops is "
            "simply lost (UDP)."
        )

    @property
    def params(self):
        return {
            "host": {"type": "string", "default": "127.0.0.1", "doc": "Destination address."},
            "port": {"type": "int", "default": 5006, "doc": "Destination UDP port."},
            "format": {
                "type": "choice",
                "default": "text",
                "options": ["text", "float64"],
                "doc": "Message format: text numbers or little-endian float64.",
            },
            "sampling_time": {
                "type": "float",
                "default": -1.0,
                "doc": "Send period in seconds (-1 = every step).",
            },
        }

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return []

    def draw_icon(self, block_rect):
        """Waves leaving an antenna."""
        from PyQt6.QtGui import QPainterPath

        path = QPainterPath()
        path.moveTo(0.25, 0.85)
        path.lineTo(0.25, 0.40)
        path.moveTo(0.15, 0.30)
        path.lineTo(0.25, 0.40)
        path.lineTo(0.35, 0.30)
        for r in (0.10, 0.20):  # right-facing arcs: the signal travels left to right
            path.moveTo(0.45, 0.45 - r)
            path.arcTo(0.45 - r, 0.45 - r, 2 * r, 2 * r, 90, -180)
        return path

    def execute(self, time, inputs, params, **kwargs):
        if params.get("_init_start_", True) or hw.current(params) is None:
            params["_init_start_"] = False
            try:
                hw.reopen(params, lambda: socket.socket(socket.AF_INET, socket.SOCK_DGRAM))
            except OSError as exc:
                return {"E": True, "error": f"UDP Send: cannot create socket: {exc}"}
        if 0 not in inputs:
            return {"E": False}
        address = (str(params.get("host", "127.0.0.1")), int(params.get("port", 5006)))
        try:
            hw.current(params).sendto(
                hw.encode(inputs[0], str(params.get("format", "text"))), address
            )
        except OSError as exc:
            # Unreachable host / full buffer: drop this message, keep simulating.
            logger.debug("UDP Send to %s failed: %s", address, exc)
        return {"E": False}
