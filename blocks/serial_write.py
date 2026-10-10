"""Serial Write sink block: write the input values to a serial port as one
comma-separated, newline-terminated text line per step it runs (set
``sampling_time`` for a fixed rate). Needs the optional ``pyserial`` package."""

import logging

from blocks import _hardware_io as hw
from blocks.base_block import BaseBlock
from blocks.serial_read import open_serial

logger = logging.getLogger(__name__)


class SerialWriteBlock(BaseBlock):
    @property
    def block_name(self):
        return "SerialWrite"

    @property
    def display_name(self):
        return "Serial Write"

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
            "Writes the input values to a serial port as one text line per step: "
            "numbers separated by commas, ending with a newline (read it on an "
            "Arduino with Serial.parseFloat or readStringUntil('\\n')). Requires the "
            "pyserial package (pip install pyserial).\n\n"
            "Parameters:\n"
            "- port: device name, e.g. COM3 (Windows) or /dev/ttyACM0 (Linux).\n"
            "- baudrate: must match the device.\n"
            "- sampling_time: write period in seconds; -1 writes on every step.\n\n"
            "Writing never blocks the simulation; if the device cannot keep up, "
            "lines are dropped."
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
            "sampling_time": {
                "type": "float",
                "default": -1.0,
                "doc": "Write period in seconds (-1 = every step).",
            },
        }

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return []

    def draw_icon(self, block_rect):
        """A plug with an outgoing arrow."""
        from PyQt6.QtGui import QPainterPath

        path = QPainterPath()
        path.addRoundedRect(0.15, 0.35, 0.30, 0.30, 0.05, 0.05)
        path.moveTo(0.05, 0.43)
        path.lineTo(0.15, 0.43)
        path.moveTo(0.05, 0.57)
        path.lineTo(0.15, 0.57)
        path.moveTo(0.55, 0.50)
        path.lineTo(0.85, 0.50)
        path.moveTo(0.77, 0.42)
        path.lineTo(0.85, 0.50)
        path.lineTo(0.77, 0.58)
        return path

    def execute(self, time, inputs, params, **kwargs):
        if params.get("_init_start_", True) or hw.current(params) is None:
            params["_init_start_"] = False
            hw.release(params.pop("_io_handle_", None))
            port, error = open_serial(params, "Serial Write")
            if error:
                return {"E": True, "error": error}
            params["_io_handle_"] = hw.register(port)
        if 0 not in inputs:
            return {"E": False}
        try:
            hw.current(params).write(hw.encode(inputs[0], "text"))
        except Exception as exc:  # noqa: BLE001 -- write timeout / unplugged device
            logger.debug("Serial Write failed: %s", exc)
        return {"E": False}
