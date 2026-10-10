"""Serial Read source block: read newline-terminated lines of numbers from a
serial port (Arduino ``Serial.println``). Needs the optional ``pyserial``
package; without it the block reports a clear error instead of failing to
load. Non-blocking: each step reads what has arrived and outputs the last
complete line; between lines the last value is held."""

import logging

import numpy as np

from blocks import _hardware_io as hw
from blocks.base_block import BaseBlock

logger = logging.getLogger(__name__)

PYSERIAL_MISSING = (
    "{block} needs the pyserial package, which is not installed. "
    "Install it with: pip install pyserial"
)


def open_serial(params, block):
    """Open the configured port (non-blocking). Returns (port, error)."""
    try:
        import serial  # pyserial: optional dependency
    except ImportError:
        return None, PYSERIAL_MISSING.format(block=block)
    try:
        port = serial.Serial(
            str(params.get("port", "")),
            int(params.get("baudrate", 115200)),
            timeout=0,
            write_timeout=0,
        )
    except (serial.SerialException, ValueError, OSError) as exc:
        return None, f"{block}: cannot open serial port {params.get('port')!r}: {exc}"
    return port, None


class SerialReadBlock(BaseBlock):
    @property
    def block_name(self):
        return "SerialRead"

    @property
    def display_name(self):
        return "Serial Read"

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
            "Reads lines of numbers from a serial port (USB/serial devices such as "
            "an Arduino) and outputs the latest line.\n\n"
            "Each line holds numbers separated by commas or spaces and ends with a "
            "newline (what Serial.println produces). Requires the pyserial package "
            "(pip install pyserial).\n\n"
            "Parameters:\n"
            "- port: device name, e.g. COM3 (Windows) or /dev/ttyACM0 (Linux).\n"
            "- baudrate: must match the device.\n"
            "- width: number of values in the output vector.\n"
            "- initial: output before the first line arrives.\n\n"
            "Outputs:\n"
            "- data: the latest values (held between lines).\n"
            "- fresh: 1 on a step where a new line arrived, else 0.\n\n"
            "Tick 'Run in real-time' in Simulation Settings so the simulation keeps "
            "pace with the hardware."
        )

    @property
    def params(self):
        return {
            "port": {
                "type": "string",
                "default": "COM3",
                "doc": "Serial device, e.g. COM3 or /dev/ttyACM0.",
            },
            "baudrate": {
                "type": "int",
                "default": 115200,
                "doc": "Baud rate (must match the device).",
            },
            "width": {"type": "int", "default": 1, "doc": "Number of values in the output."},
            "initial": {"type": "float", "default": 0.0, "doc": "Output before the first line."},
        }

    @property
    def inputs(self):
        return []

    @property
    def outputs(self):
        return [{"name": "data", "type": "any"}, {"name": "fresh", "type": "float"}]

    def draw_icon(self, block_rect):
        """A plug with an incoming arrow."""
        from PyQt6.QtGui import QPainterPath

        path = QPainterPath()
        path.addRoundedRect(0.45, 0.35, 0.30, 0.30, 0.05, 0.05)
        path.moveTo(0.75, 0.43)
        path.lineTo(0.85, 0.43)
        path.moveTo(0.75, 0.57)
        path.lineTo(0.85, 0.57)
        path.moveTo(0.15, 0.50)
        path.lineTo(0.40, 0.50)
        path.moveTo(0.32, 0.42)
        path.lineTo(0.40, 0.50)
        path.lineTo(0.32, 0.58)
        return path

    def execute(self, time, inputs, params, **kwargs):
        width = max(1, int(params.get("width", 1)))
        if params.get("_init_start_", True) or hw.current(params) is None:
            params["_init_start_"] = False
            params["_held_"] = np.full(width, float(params.get("initial", 0.0)))
            params["_buffer_"] = b""
            hw.release(params.pop("_io_handle_", None))
            port, error = open_serial(params, "Serial Read")
            if error:
                return {"E": True, "error": error}
            params["_io_handle_"] = hw.register(port)
        port = hw.current(params)
        try:
            waiting = int(getattr(port, "in_waiting", 0) or 0)
            chunk = port.read(waiting) if waiting else b""
        except Exception as exc:  # noqa: BLE001 -- device unplugged mid-run
            return {"E": True, "error": f"Serial Read: read failed: {exc}"}
        buffer = params.get("_buffer_", b"") + bytes(chunk)
        *lines, buffer = buffer.split(b"\n")
        params["_buffer_"] = buffer[-4096:]  # a device that never sends \n cannot grow it forever
        latest = None
        for line in lines:
            values = hw.parse(line, "text")
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
