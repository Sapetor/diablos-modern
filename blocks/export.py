from blocks.base_block import BaseBlock
import numpy as np


class ExportBlock(BaseBlock):
    def __init__(self):
        super().__init__()

    @property
    def block_name(self):
        return "Export"

    @property
    def category(self):
        return "Sinks"

    @property
    def b_type(self):
        """Sink block - consumes output without producing further output."""
        return 3

    @property
    def color(self):
        return "red"

    @property
    def params(self):
        return {
            "str_name": {"default": "default", "type": "string"},
            "_init_start_": {"default": True, "type": "bool"},
        }

    @property
    def doc(self):
        return (
            "Data Export."
            "\n\nSaves simulation data to a file (e.g., .npz, .mat, .csv)."
            "\n\nParameters:"
            "\n- Filename: Destination file path."
            "\n- Variable Name: Name of variable in saved file."
            "\n\nUsage:"
            "\nSave results for post-processing in Python or other tools."
        )

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return []

    def draw_icon(self, block_rect):
        """Draw export/file icon in normalized 0-1 coordinates."""
        from PyQt5.QtGui import QPainterPath
        path = QPainterPath()
        # Box
        path.moveTo(0.2, 0.2)
        path.lineTo(0.8, 0.2)
        path.lineTo(0.8, 0.8)
        path.lineTo(0.2, 0.8)
        path.lineTo(0.2, 0.2)
        # Arrow out
        path.moveTo(0.5, 0.5)
        path.lineTo(1.0, 0.5)
        path.moveTo(0.8, 0.3)
        path.lineTo(1.0, 0.5)
        path.lineTo(0.8, 0.7)
        return path

    def execute(self, time, inputs, params, **kwargs):
        """
        Save and export block signals.
        """
        # To prevent saving data in wrong iterations (RK45 integration)
        if '_skip_' in params.keys() and params['_skip_']:
            params['_skip_'] = False
            return {0: np.array([0.0]), 'E': False}

        # Guard against a missing/None input on port 0. A connected sink is
        # normally wired before execute(), but a dangling port would otherwise
        # raise an unguarded KeyError (line below) or feed None into concatenate.
        if inputs.get(0) is None:
            return {'E': True, 'error': 'Export has no input'}

        # The fast-path replay loop writes a Python list to params['vector']
        # after each simulation. If we land here on a later sim with a stale
        # non-ndarray vector, we cannot safely concatenate; force re-init.
        prev_vec = params.get('vector', None)
        if not isinstance(prev_vec, np.ndarray):
            params['_init_start_'] = True

        # Initialization of the saving vector
        if params.get('_init_start_', True):
            aux_vector = np.array([inputs[0]])
            try:
                params['vec_dim'] = len(inputs[0])
            except TypeError:
                params['vec_dim'] = 1

            labels = params['str_name']
            if labels == 'default':
                labels = params['_name_'] + '-0'
            labels = labels.replace(' ', '').split(',')
            if len(labels) < params['vec_dim']:
                for i in range(len(labels), params['vec_dim']):
                    labels.append(params['_name_'] + '-' + str(i))
            elif len(labels) > params['vec_dim']:
                labels = labels[:params['vec_dim']]
            if len(labels) == params['vec_dim'] == 1:
                labels = labels[0]
            params['vec_labels'] = labels
            params['_init_start_'] = False
            # Amortized-O(1) append via a geometrically-grown row buffer; was
            # np.concatenate of the full history each step (O(n^2)). Runtime-only
            # keys (trailing underscore => excluded from save).
            _row = np.asarray(inputs[0], dtype=float)
            _buf = np.empty((8,) + _row.shape, dtype=float)
            _buf[0] = _row
            params['_export_buf_'] = _buf
            params['_export_len_'] = 1
            params['vector'] = _buf[:1]
        else:
            try:
                new_dim = len(inputs[0])
            except TypeError:
                new_dim = 1
            _row = np.asarray(inputs[0], dtype=float)
            # If the per-sample dimension changed since init (e.g. the upstream
            # signal width varies mid-run), re-initialize instead of producing a
            # ragged buffer that breaks the (N, vec_dim) layout export_data() expects.
            if new_dim != params.get('vec_dim', new_dim):
                _buf = np.empty((8,) + _row.shape, dtype=float)
                _buf[0] = _row
                params['_export_buf_'] = _buf
                params['_export_len_'] = 1
                params['vec_dim'] = new_dim
                params['vector'] = _buf[:1]
            else:
                _buf = params.get('_export_buf_')
                _len = int(params.get('_export_len_', 0))
                # Rebuild if missing/stale (e.g. a prior fast-path run wrote a list).
                if (not isinstance(_buf, np.ndarray) or _len < 1
                        or _len > _buf.shape[0] or _buf.shape[1:] != _row.shape):
                    _buf = np.asarray(params.get('vector', np.array([inputs[0]])), dtype=float).copy()
                    _len = _buf.shape[0]
                if _len >= _buf.shape[0]:
                    _grown = np.empty((max(_len + 1, _buf.shape[0] * 2),) + _buf.shape[1:], dtype=float)
                    _grown[:_len] = _buf[:_len]
                    _buf = _grown
                _buf[_len] = _row
                _len += 1
                params['_export_buf_'] = _buf
                params['_export_len_'] = _len
                params['vector'] = _buf[:_len]
        return {0: np.array([0.0]), 'E': False}
