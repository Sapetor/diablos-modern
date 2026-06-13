"""
Unit tests for the Export block output FORMAT feature.

Covers:
- The ``format`` parameter on :class:`blocks.export.ExportBlock`.
- :meth:`lib.lib.DSim.export_data` branching on that format and writing a
  readable npz / csv / mat file with the expected time + signal data.
"""

import os

import numpy as np
import pytest


def _populated_export_block(values, str_name='default', fmt='npz'):
    """Run an ExportBlock over ``values`` and return it with realistic params.

    ``values`` is a list of per-timestep inputs (scalars or 1-D arrays). The
    returned block has the same ``params`` layout the simulation engine leaves
    behind: ``vector``, ``vec_dim``, ``vec_labels`` and the chosen ``format``.
    """
    from blocks.export import ExportBlock

    block = ExportBlock()
    params = {
        '_init_start_': True,
        'str_name': str_name,
        'format': fmt,
        '_name_': 'Export0',
    }
    for i, v in enumerate(values):
        block.execute(time=i * 0.1, inputs={0: v}, params=params)
    return params


class _StubBlock:
    """Minimal stand-in for a simulation block carrying Export params."""

    def __init__(self, params):
        self.block_fn = 'Export'
        self.params = params


class _StubDSim:
    """Minimal object exposing the attributes DSim.export_data() touches."""

    def __init__(self, blocks_list, timeline, filename):
        self.blocks_list = blocks_list
        self.timeline = timeline
        self.filename = filename

    export_data = None  # bound below


# Bind the real method onto the stub so we exercise production code unchanged.
from lib.lib import DSim  # noqa: E402

_StubDSim.export_data = DSim.export_data


@pytest.mark.unit
class TestExportBlockFormatParam:
    """The Export block advertises a selectable output format."""

    def test_format_param_present_with_choices(self):
        from blocks.export import ExportBlock
        block = ExportBlock()
        assert 'format' in block.params, "Export should expose a 'format' param"
        spec = block.params['format']
        assert spec['default'] == 'npz', "Default must preserve npz behavior"
        assert spec['choices'] == ['npz', 'csv', 'mat'], "Should offer npz/csv/mat"

    def test_default_format_is_npz(self):
        # When no format is passed in params, execute must still work and the
        # default should leave npz as the effective behavior downstream.
        params = _populated_export_block([1.0, 2.0, 3.0])
        params.pop('format', None)
        assert params['vec_dim'] == 1
        assert len(params['vector']) == 3


@pytest.mark.unit
class TestExportDataFormats:
    """DSim.export_data writes a readable file per selected format."""

    def _run_export(self, tmp_path, monkeypatch, values, fmt):
        monkeypatch.chdir(tmp_path)
        params = _populated_export_block(values, str_name='x,y', fmt=fmt)
        timeline = np.arange(len(values)) * 0.1
        dsim = _StubDSim(
            blocks_list=[_StubBlock(params)],
            timeline=timeline,
            filename='mydiagram.dat',
        )
        dsim.export_data()
        return params, timeline

    def test_npz_roundtrip(self, tmp_path, monkeypatch):
        values = [np.array([1.0, 10.0]),
                  np.array([2.0, 20.0]),
                  np.array([3.0, 30.0])]
        params, timeline = self._run_export(tmp_path, monkeypatch, values, 'npz')

        out = os.path.join(str(tmp_path), 'saves', 'mydiagram.npz')
        assert os.path.exists(out), "npz file should be written"
        loaded = np.load(out)
        assert np.allclose(loaded['t'], timeline)
        assert np.allclose(loaded['x'], [1.0, 2.0, 3.0])
        assert np.allclose(loaded['y'], [10.0, 20.0, 30.0])
        loaded.close()

    def test_csv_roundtrip(self, tmp_path, monkeypatch):
        values = [np.array([1.0, 10.0]),
                  np.array([2.0, 20.0]),
                  np.array([3.0, 30.0])]
        params, timeline = self._run_export(tmp_path, monkeypatch, values, 'csv')

        out = os.path.join(str(tmp_path), 'saves', 'mydiagram.csv')
        assert os.path.exists(out), "csv file should be written"

        # Header must mirror the columns: t + the vec labels.
        with open(out, 'r') as fh:
            header = fh.readline().strip()
        assert header == 't,x,y', f"Unexpected header: {header!r}"

        data = np.loadtxt(out, delimiter=',', skiprows=1)
        assert np.allclose(data[:, 0], timeline)
        assert np.allclose(data[:, 1], [1.0, 2.0, 3.0])
        assert np.allclose(data[:, 2], [10.0, 20.0, 30.0])

    def test_mat_roundtrip(self, tmp_path, monkeypatch):
        values = [np.array([1.0, 10.0]),
                  np.array([2.0, 20.0]),
                  np.array([3.0, 30.0])]
        params, timeline = self._run_export(tmp_path, monkeypatch, values, 'mat')

        out = os.path.join(str(tmp_path), 'saves', 'mydiagram.mat')
        assert os.path.exists(out), "mat file should be written"

        from scipy.io import loadmat
        loaded = loadmat(out)
        assert np.allclose(np.ravel(loaded['t']), timeline)
        assert np.allclose(np.ravel(loaded['x']), [1.0, 2.0, 3.0])
        assert np.allclose(np.ravel(loaded['y']), [10.0, 20.0, 30.0])

    def test_scalar_signal_csv(self, tmp_path, monkeypatch):
        # vec_dim == 1 path: single labelled column.
        values = [1.0, 2.0, 3.0, 4.0]
        monkeypatch.chdir(tmp_path)
        params = _populated_export_block(values, str_name='sig', fmt='csv')
        timeline = np.arange(len(values)) * 0.1
        dsim = _StubDSim([_StubBlock(params)], timeline, 'scalar.dat')
        dsim.export_data()

        out = os.path.join(str(tmp_path), 'saves', 'scalar.csv')
        with open(out, 'r') as fh:
            header = fh.readline().strip()
        assert header == 't,sig', f"Unexpected header: {header!r}"
        data = np.loadtxt(out, delimiter=',', skiprows=1)
        assert np.allclose(data[:, 0], timeline)
        assert np.allclose(data[:, 1], [1.0, 2.0, 3.0, 4.0])

    def test_unknown_format_falls_back_to_npz(self, tmp_path, monkeypatch):
        values = [1.0, 2.0]
        monkeypatch.chdir(tmp_path)
        params = _populated_export_block(values, str_name='sig', fmt='bogus')
        timeline = np.arange(len(values)) * 0.1
        dsim = _StubDSim([_StubBlock(params)], timeline, 'fallback.dat')
        dsim.export_data()

        assert os.path.exists(os.path.join(str(tmp_path), 'saves', 'fallback.npz'))
