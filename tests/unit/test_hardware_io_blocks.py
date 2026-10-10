"""UDP and serial hardware-I/O blocks.

UDP tests use localhost only (an ephemeral port); serial tests inject a fake
``serial`` module, so no real device or network port is ever opened.
"""

import socket
import struct
import sys
import time
import types

import numpy as np
import pytest

from blocks import _hardware_io as hw
from blocks.serial_read import SerialReadBlock
from blocks.serial_write import SerialWriteBlock
from blocks.udp_receive import UDPReceiveBlock
from blocks.udp_send import UDPSendBlock


def _p(**params):
    return {"_init_start_": True, **params}


@pytest.fixture
def receiver():
    block, params = UDPReceiveBlock(), _p(host="127.0.0.1", port=0, width=2, initial=-1.0)
    first = block.execute(0.0, {}, params)
    port = hw.current(params).getsockname()[1]
    yield block, params, port, first
    hw.release(params.get("_io_handle_"))


def _send(port, payload):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.sendto(payload, ("127.0.0.1", port))


def _poll(block, params, t=0.1):
    for _ in range(50):  # loopback delivery is fast but not instantaneous
        out = block.execute(t, {}, params)
        if out[1][0] == 1.0:
            return out
        time.sleep(0.01)
    return out


@pytest.mark.unit
class TestHardwareIoHelpers:
    def test_parse_and_encode(self):
        np.testing.assert_allclose(hw.parse(b"1.5, 2;3 4\n", "text"), [1.5, 2, 3, 4])
        assert hw.parse(b"1.5,abc", "text") is None  # malformed -> dropped whole
        assert hw.parse(b"   ", "text") is None
        np.testing.assert_allclose(hw.parse(struct.pack("<2d", 1.0, -2.5), "float64"), [1.0, -2.5])
        assert hw.encode([1.0, 2.5], "text") == b"1.0,2.5\n"
        assert hw.encode(np.array([[3.0]]), "float64") == struct.pack("<d", 3.0)

    def test_fit_width_pads_with_held_values(self):
        held = np.array([9.0, 9.0, 9.0])
        np.testing.assert_allclose(hw.fit_width(np.array([1.0]), 3, held), [1.0, 9.0, 9.0])
        np.testing.assert_allclose(hw.fit_width(np.array([1.0, 2, 3, 4]), 3, held), [1, 2, 3])
        np.testing.assert_allclose(hw.fit_width(None, 3, held), held)

    def test_handles_are_closed_on_reopen(self):
        closed = []
        params = {}
        hw.reopen(params, lambda: types.SimpleNamespace(close=lambda: closed.append(1)))
        hw.reopen(params, lambda: types.SimpleNamespace(close=lambda: closed.append(2)))
        assert closed == [1]
        hw.release(params["_io_handle_"])
        assert closed == [1, 2]


@pytest.mark.unit
class TestUdp:
    def test_initial_value_until_a_message_arrives(self, receiver):
        block, params, port, first = receiver
        np.testing.assert_allclose(first[0], [-1.0, -1.0])
        assert first[1][0] == 0.0

    def test_text_message_latest_wins_and_is_held(self, receiver):
        block, params, port, _ = receiver
        _send(port, b"1,2")
        _send(port, b"3 4\n")
        time.sleep(0.05)
        out = _poll(block, params)
        np.testing.assert_allclose(out[0], [3.0, 4.0])
        later = block.execute(0.2, {}, params)
        np.testing.assert_allclose(later[0], [3.0, 4.0])
        assert later[1][0] == 0.0

    def test_float64_and_send_block_round_trip(self, receiver):
        block, params, port, _ = receiver
        params["format"] = "float64"
        sender, sparams = UDPSendBlock(), _p(host="127.0.0.1", port=port, format="float64")
        try:
            assert sender.execute(0.0, {0: np.array([7.5, -1.25])}, sparams) == {"E": False}
            out = _poll(block, params)
        finally:
            hw.release(sparams.get("_io_handle_"))
        np.testing.assert_allclose(out[0], [7.5, -1.25])

    def test_rerun_rebinds_and_port_in_use_is_reported(self, receiver):
        block, params, port, _ = receiver
        old = params["_io_handle_"]
        params["_init_start_"] = True  # engine reset for the next run
        block.execute(0.0, {}, params)
        assert hw.get(old) is None and params["_io_handle_"] != old
        # A non-reusable socket holding a port makes binding fail with a clear error.
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as hog:
            if sys.platform == "win32":
                hog.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            hog.bind(("127.0.0.1", 0))
            busy = hog.getsockname()[1]
            p2 = _p(host="127.0.0.1", port=busy)
            out = UDPReceiveBlock().execute(0.0, {}, p2)
            hw.release(p2.get("_io_handle_"))
        # Linux lets SO_REUSEADDR sockets share a UDP port, so only Windows
        # (SO_EXCLUSIVEADDRUSE) is guaranteed to refuse the second bind.
        if sys.platform == "win32":
            assert out["E"] and "cannot listen on port" in out["error"]


class _FakeSerialPort:
    def __init__(self, port, baudrate, timeout=None, write_timeout=None):
        if port == "BAD":
            raise _FakeSerialModule.SerialException("no such port")
        self.port, self.baudrate = port, baudrate
        self.incoming, self.written, self.closed = b"", [], False

    @property
    def in_waiting(self):
        return len(self.incoming)

    def read(self, n):
        data, self.incoming = self.incoming[:n], self.incoming[n:]
        return data

    def write(self, data):
        self.written.append(data)

    def close(self):
        self.closed = True


class _FakeSerialModule(types.ModuleType):
    class SerialException(Exception):
        pass

    Serial = _FakeSerialPort


@pytest.fixture
def fake_serial(monkeypatch):
    monkeypatch.setitem(sys.modules, "serial", _FakeSerialModule("serial"))


@pytest.mark.unit
class TestSerial:
    def test_read_lines_hold_and_partial_lines(self, fake_serial):
        block, params = SerialReadBlock(), _p(port="COM9", baudrate=9600, width=2)
        assert block.execute(0.0, {}, params)[1][0] == 0.0
        dev = hw.current(params)
        assert (dev.port, dev.baudrate) == ("COM9", 9600)
        dev.incoming = b"1,2\n3,4\n5,"  # last line incomplete
        out = block.execute(0.1, {}, params)
        np.testing.assert_allclose(out[0], [3.0, 4.0])
        assert out[1][0] == 1.0
        dev.incoming = b"6\r\n"
        np.testing.assert_allclose(block.execute(0.2, {}, params)[0], [5.0, 6.0])
        held = block.execute(0.3, {}, params)
        np.testing.assert_allclose(held[0], [5.0, 6.0])
        assert held[1][0] == 0.0
        hw.release(params["_io_handle_"])
        assert dev.closed

    def test_write_lines(self, fake_serial):
        block, params = SerialWriteBlock(), _p(port="COM9")
        block.execute(0.0, {0: np.array([1.0, 2.5])}, params)
        block.execute(0.1, {0: 3.0}, params)
        assert hw.current(params).written == [b"1.0,2.5\n", b"3.0\n"]
        hw.release(params["_io_handle_"])

    def test_bad_port_is_reported(self, fake_serial):
        out = SerialReadBlock().execute(0.0, {}, _p(port="BAD"))
        assert out["E"] and "cannot open serial port 'BAD'" in out["error"]

    def test_missing_pyserial_gives_a_clear_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "serial", None)  # makes "import serial" fail
        for block in (SerialReadBlock(), SerialWriteBlock()):
            out = block.execute(0.0, {0: 1.0}, _p(port="COM3"))
            assert out["E"] and "pip install pyserial" in out["error"]


@pytest.mark.integration
def test_headless_run_sends_at_the_sampling_rate(tmp_path):
    from lib.cli import run_diagram
    from lib.diagram_builder import DiagramBuilder

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.settimeout(0.5)
        port = listener.getsockname()[1]

        b = DiagramBuilder(sim_time=1.0, sim_dt=0.01)
        sine = b.add_block(
            "Sine",
            x=100,
            y=100,
            name="sine",
            params={"amplitude": 2.0, "omega": 6.0, "init_angle": 0.0},
        )
        udp = b.add_block(
            "UDPSend",
            x=300,
            y=100,
            name="udp",
            params={"host": "127.0.0.1", "port": port, "format": "text", "sampling_time": 0.1},
            in_ports=1,
            out_ports=0,
        )
        b.connect(sine, 0, udp, 0)
        path = tmp_path / "udp.diablos"
        b.save(str(path))
        run_diagram(str(path))

        values = []
        try:
            while True:
                values.append(float(listener.recv(1024)))
        except socket.timeout:
            pass
    assert 10 <= len(values) <= 12  # every 0.1 s over 1 s, not every 0.01 s step
    assert max(abs(v) for v in values) <= 2.0 + 1e-9
    hw.release_all()
