"""UDP Receive source block: read numbers sent to a UDP port (lab rigs, other
programs). Non-blocking: every step drains the datagrams that arrived and
outputs the most recent one; between packets the last value is held."""

import logging
import socket

import numpy as np

from blocks import _hardware_io as hw
from blocks.base_block import BaseBlock

logger = logging.getLogger(__name__)


class UDPReceiveBlock(BaseBlock):
    @property
    def block_name(self):
        return "UDPReceive"

    @property
    def display_name(self):
        return "UDP Receive"

    @property
    def category(self):
        return "Sources"

    @property
    def b_type(self):
        return 0

    @property
    def color(self):
        return "mediumseagreen"

    @property
    def requires_inputs(self):
        return False

    @property
    def doc(self):
        return (
            "Receives numbers on a UDP port and outputs the latest message.\n\n"
            "Parameters:\n"
            "- host: local address to listen on (127.0.0.1 = this computer only, "
            "0.0.0.0 = all network interfaces).\n"
            "- port: UDP port.\n"
            "- format: 'text' (numbers separated by commas or spaces) or 'float64' "
            "(little-endian doubles).\n"
            "- width: number of values in the output vector.\n"
            "- initial: output before the first message arrives.\n\n"
            "Outputs:\n"
            "- data: the latest values (held between messages).\n"
            "- fresh: 1 on a step where a new message arrived, else 0.\n\n"
            "Tick 'Run in real-time' in Simulation Settings so the simulation keeps "
            "pace with the hardware."
        )

    @property
    def params(self):
        return {
            "host": {
                "type": "string",
                "default": "127.0.0.1",
                "doc": "Local address to listen on.",
            },
            "port": {"type": "int", "default": 5005, "doc": "UDP port to listen on."},
            "format": {
                "type": "choice",
                "default": "text",
                "options": ["text", "float64"],
                "doc": "Message format: text numbers or little-endian float64.",
            },
            "width": {"type": "int", "default": 1, "doc": "Number of values in the output."},
            "initial": {"type": "float", "default": 0.0, "doc": "Output before the first message."},
        }

    @property
    def inputs(self):
        return []

    @property
    def outputs(self):
        return [{"name": "data", "type": "any"}, {"name": "fresh", "type": "float"}]

    def draw_icon(self, block_rect):
        """Waves arriving at an antenna."""
        from PyQt6.QtGui import QPainterPath

        path = QPainterPath()
        path.moveTo(0.75, 0.85)
        path.lineTo(0.75, 0.40)
        path.moveTo(0.65, 0.30)
        path.lineTo(0.75, 0.40)
        path.lineTo(0.85, 0.30)
        for r in (0.10, 0.20):  # right-facing arcs: the signal travels left to right
            path.moveTo(0.40, 0.45 - r)
            path.arcTo(0.40 - r, 0.45 - r, 2 * r, 2 * r, 90, -180)
        return path

    def _open(self, params):
        def opener():
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((str(params.get("host", "127.0.0.1")), int(params.get("port", 5005))))
            sock.setblocking(False)
            return sock

        return hw.reopen(params, opener)

    def execute(self, time, inputs, params, **kwargs):
        width = max(1, int(params.get("width", 1)))
        if params.get("_init_start_", True) or hw.current(params) is None:
            params["_init_start_"] = False
            params["_held_"] = np.full(width, float(params.get("initial", 0.0)))
            try:
                self._open(params)
            except OSError as exc:
                return {
                    "E": True,
                    "error": f"UDP Receive: cannot listen on port {params.get('port')}: {exc}",
                }
        sock = hw.current(params)
        latest = None
        while True:
            try:
                payload = sock.recv(65535)
            except (BlockingIOError, InterruptedError):
                break
            except OSError as exc:  # e.g. ICMP port unreachable on Windows
                logger.debug("UDP Receive: recv failed: %s", exc)
                break
            values = hw.parse(payload, str(params.get("format", "text")))
            if values is not None:
                latest = values
        held = params.get("_held_")
        if held is None or held.size != width:
            held = np.full(width, float(params.get("initial", 0.0)))
        params["_held_"] = hw.fit_width(latest, width, held)
        return {
            0: params["_held_"].copy(),
            1: np.atleast_1d(1.0 if latest is not None else 0.0),
            "E": False,
        }
